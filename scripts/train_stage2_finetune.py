"""Stage 2 of the two-stage fine-tuning experiment for mushroom's data-scarce classes.

Loads a Stage 1 checkpoint (trained normally on all 169 classes — see
configs/mushroom_twostage_stage1.yaml), freezes everything except the last
residual block + classifier head, then fine-tunes for a few epochs on
*only* the 26 data-scarce classes (notebooks/08_mushroom_confusion_diagnosis.ipynb,
list b) at a low learning rate. Evaluated every epoch on the FULL
validation set (all 169 classes) so we can directly see whether this
helps the scarce classes without catastrophically forgetting the other 143
— both overall and scarce-only macro-F1 are printed and logged.

Run from the project root:

    python scripts/train_stage2_finetune.py --config configs/mushroom_twostage_stage2.yaml
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

from src.data.config import load_config
from src.data.dataloader import build_dataloader, build_pipeline
from src.data.dataset import ImageClassificationDataset
from src.data.label_map import invert_label_map
from src.evaluation.metrics import build_evaluation_report, macro_f1_score
from src.models.checkpoint import load_model_from_checkpoint, save_checkpoint
from src.training.optimizers import build_optimizer
from src.training.tracking import (
    log_epoch_metrics,
    log_epoch_metrics_mlflow,
    log_evaluation_report,
    log_mlflow_evaluation_report,
    log_mlflow_model,
    log_run_duration,
    mlflow_run,
    wandb_run,
)
from src.utils.run_metadata import collect_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the stage-2 config YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging/registry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_type = config["dataset_type"]
    model_config = config["model"]
    training_config = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipeline = build_pipeline(config)
    label_map = pipeline["label_map"]
    num_classes = len(label_map)
    idx_to_label = invert_label_map(label_map)

    with open(config["scarce_species_path"], encoding="utf-8") as f:
        scarce_labels = set(json.load(f))
    scarce_indices = [label_map[l] for l in scarce_labels if l in label_map]
    print(f"Stage 2: fine-tuning on {len(scarce_indices)} scarce classes only")

    model, stage1_checkpoint = load_model_from_checkpoint(
        config["stage1_checkpoint"], map_location=device
    )
    model.to(device)
    model.freeze_except_last_block()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Stage 1 checkpoint epoch: {stage1_checkpoint['epoch']}")
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable / total:.1%})")

    train_df = pipeline["frames"]["train"]
    scarce_train_df = train_df[train_df["label"].isin(scarce_labels)].reset_index(drop=True)
    scarce_train_dataset = ImageClassificationDataset(
        scarce_train_df, label_map, pipeline["train_dataset"].transform
    )
    print(f"Stage 2 training set: {len(scarce_train_dataset)} images (scarce classes only)")

    train_loader = build_dataloader(
        scarce_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    val_loader = build_dataloader(
        pipeline["val_dataset"],
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        training_config["optimizer"],
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=training_config["lr"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )

    epochs = training_config["epochs"]
    early_stopping_patience = training_config.get("early_stopping_patience", 3)
    checkpoint_stem = config["checkpoint_name"]
    checkpoint_path = Path("outputs/checkpoints") / f"{checkpoint_stem}.pt"
    best_checkpoint_path = Path("outputs/checkpoints") / f"{checkpoint_stem}_best.pt"

    run_metadata = collect_run_metadata(args.config, config.get("dataset_root"))

    def save(path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            model,
            path,
            architecture=model_config["name"],
            dataset_type=dataset_type,
            label_map_path=config["label_map_path"],
            epoch=epoch,
            optimizer=optimizer,
            run_metadata=run_metadata,
        )

    run_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
    run_name = f"{dataset_type}_{model_config['name']}_stage2_{run_id}"
    run_params = {
        "model": model_config["name"],
        "dataset": dataset_type,
        "batch_size": config["batch_size"],
        "lr": training_config["lr"],
        "epochs": epochs,
        "stage1_checkpoint": config["stage1_checkpoint"],
        "scarce_train_size": len(scarce_train_dataset),
        "trainable_param_fraction": trainable / total,
    }
    tracking_context = (
        nullcontext() if args.no_wandb else wandb_run("visionlab", run_name, run_params)
    )
    mlflow_context = (
        nullcontext()
        if args.no_mlflow
        else mlflow_run("visionlab", run_name, run_params, run_metadata=run_metadata)
    )

    run_start = time.perf_counter()
    best_val_macro_f1 = -1.0
    best_y_true = None
    best_y_pred = None
    epochs_without_improvement = 0

    with tracking_context, mlflow_context:
        for epoch in range(epochs):
            epoch_start = time.perf_counter()

            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                train_correct += (outputs.argmax(dim=1) == labels).sum().item()
                train_total += images.size(0)

            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images.to(device))
                    y_pred.extend(outputs.argmax(dim=1).tolist())
                    y_true.extend(labels.tolist())

            overall_macro_f1 = macro_f1_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=range(num_classes), zero_division=0
            )
            scarce_macro_f1 = float(sum(f1[i] for i in scarce_indices) / len(scarce_indices))
            elapsed = time.perf_counter() - epoch_start

            print(
                f"epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss / train_total:.4f} train_acc={train_correct / train_total:.4f} | "
                f"val_macro_f1_overall={overall_macro_f1:.4f} "
                f"val_macro_f1_scarce_only={scarce_macro_f1:.4f} | {elapsed:.1f}s"
            )

            epoch_train_metrics = {
                "loss": train_loss / train_total,
                "accuracy": train_correct / train_total,
            }
            epoch_val_metrics = {"loss": 0.0, "accuracy": 0.0}
            epoch_extra_metrics = {
                "val_macro_f1_overall": overall_macro_f1,
                "val_macro_f1_scarce_only": scarce_macro_f1,
                "epoch_duration_seconds": elapsed,
            }
            if not args.no_wandb:
                log_epoch_metrics(
                    epoch, epoch_train_metrics, epoch_val_metrics, extra_metrics=epoch_extra_metrics
                )
            if not args.no_mlflow:
                log_epoch_metrics_mlflow(
                    epoch, epoch_train_metrics, epoch_val_metrics, extra_metrics=epoch_extra_metrics
                )

            save(checkpoint_path, epoch)

            if overall_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = overall_macro_f1
                best_y_true, best_y_pred = y_true, y_pred
                epochs_without_improvement = 0
                save(best_checkpoint_path, epoch)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"\nEarly stopping: overall val_macro_f1 hasn't improved for "
                        f"{early_stopping_patience} epochs (best={best_val_macro_f1:.4f})"
                    )
                    break

        run_duration = time.perf_counter() - run_start
        if best_y_true is not None:
            report = build_evaluation_report(best_y_true, best_y_pred, idx_to_label)
            report_path = Path("outputs/reports") / f"{checkpoint_stem}_eval_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\nOverall macro-F1 (best epoch): {report['macro_f1']:.4f}")
            print("Worst 10 classes (by F1):")
            for row in report["worst_classes"]:
                print(
                    f"  {row['label']:<40} precision={row['precision']:.3f} "
                    f"recall={row['recall']:.3f} f1={row['f1']:.3f} support={row['support']}"
                )
            print(f"Full report saved to {report_path}")
            if not args.no_wandb:
                log_evaluation_report(report)
            if not args.no_mlflow:
                log_mlflow_evaluation_report(report)
                log_mlflow_model(
                    model,
                    best_checkpoint_path,
                    registered_model_name=f"{dataset_type}_{model_config['name']}_stage2",
                    run_metadata=run_metadata,
                )

        if not args.no_wandb:
            log_run_duration(run_duration)
        if not args.no_mlflow and best_y_true is not None:
            mlflow.log_metric("run_duration_seconds", run_duration)
        print(f"\nTotal run duration: {run_duration:.1f}s")

    print(f"Best checkpoint (overall val_macro_f1={best_val_macro_f1:.4f}) saved to {best_checkpoint_path}")


if __name__ == "__main__":
    main()
