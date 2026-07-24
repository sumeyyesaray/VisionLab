"""Baseline training script — wires config, pipeline, model, and engine together.

Run from the project root:

    .venv\\Scripts\\python.exe scripts\\train_baseline.py --config configs/mushroom.yaml

Wrapped in `if __name__ == "__main__":` because the configs use
`num_workers > 0` — on Windows, multiprocessing's "spawn" start method
requires this guard (see scripts/check_pipeline.py).

Logs to Weights & Biases (wandb.ai) by default — disable with --no-wandb.
Saves a resumable checkpoint (model + optimizer + scaler state) after
every epoch; continue an interrupted run with --resume <path>.
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import mlflow
import torch
from torch.utils.data import Subset

from src.data.config import load_config
from src.data.dataloader import build_dataloader, build_pipeline
from src.data.dataset import ImageClassificationDataset
from src.data.label_map import invert_label_map
from src.data.sampling import subsample_per_class
from src.data.transforms import ScarcityAwareTransform, build_train_transform
from src.evaluation.metrics import build_evaluation_report, macro_f1_score
from src.models.checkpoint import load_checkpoint, save_checkpoint
from src.models.registry import build_model
from src.training.engine import evaluate, train_one_epoch
from src.training.losses import build_loss, compute_class_weights
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
    parser.add_argument("--config", required=True, help="Path to a dataset config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate train/val to the first N images (quick smoke test)",
    )
    parser.add_argument(
        "--subset-per-class",
        type=int,
        default=None,
        help="Stratified subset: up to N training images per class",
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging/registry")
    return parser.parse_args()


def build_datasets(pipeline: dict, args: argparse.Namespace):
    train_dataset = pipeline["train_dataset"]
    val_dataset = pipeline["val_dataset"]

    if args.subset_per_class is not None:
        subset_df = subsample_per_class(pipeline["frames"]["train"], args.subset_per_class)
        train_dataset = ImageClassificationDataset(
            subset_df, pipeline["label_map"], train_dataset.transform
        )
    elif args.limit is not None:
        train_dataset = Subset(train_dataset, range(min(args.limit, len(train_dataset))))

    if args.limit is not None:
        val_dataset = Subset(val_dataset, range(min(args.limit, len(val_dataset))))

    return train_dataset, val_dataset


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_type = config["dataset_type"]
    model_config = config["model"]
    training_config = config["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipeline = build_pipeline(config)
    num_classes = len(pipeline["label_map"])
    idx_to_label = invert_label_map(pipeline["label_map"])

    scarce_species_path = config.get("scarce_species_path")
    if scarce_species_path:
        with open(scarce_species_path, encoding="utf-8") as f:
            scarce_labels = set(json.load(f))
        heavy_transform = build_train_transform(
            image_size=config["image_size"],
            augmentation_preset=config.get("scarce_augmentation_preset", "heavy"),
            random_resized_crop_scale=tuple(config["random_resized_crop_scale"]),
        )
        print(
            f"Scarcity-aware augmentation: {len(scarce_labels)} classes get "
            f"'{config.get('scarce_augmentation_preset', 'heavy')}', the rest get "
            f"'{config['augmentation_preset']}'"
        )
        pipeline["train_dataset"].transform = ScarcityAwareTransform(
            scarce_labels, pipeline["train_dataset"].transform, heavy_transform
        )

    train_dataset, val_dataset = build_datasets(pipeline, args)
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)}")

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    model = build_model(
        model_config["name"], num_classes=num_classes, pretrained=model_config["pretrained"]
    ).to(device)
    if model_config["freeze_backbone"]:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    weight = None
    if training_config["class_weighted_loss"]:
        label_counts = pipeline["frames"]["train"]["label"].value_counts().to_dict()
        weight = compute_class_weights(label_counts, pipeline["label_map"]).to(device)

    criterion = build_loss(
        training_config["loss"],
        weight=weight,
        label_smoothing=training_config.get("label_smoothing", 0.0),
    )
    optimizer = build_optimizer(
        training_config["optimizer"],
        model.parameters(),
        lr=training_config["lr"],
        weight_decay=training_config.get("weight_decay", 0.01),
    )
    lr_scheduler_patience = training_config.get("lr_scheduler_patience", 2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_scheduler_patience
    )
    early_stopping_patience = training_config.get("early_stopping_patience", 5)

    use_amp = training_config.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {args.resume}, continuing at epoch {start_epoch + 1}")

    epochs = args.epochs if args.epochs is not None else training_config["epochs"]
    # Override with `checkpoint_name` in config when running concurrent experiments that
    # would otherwise share (and race on) the same `{dataset}_{model}` checkpoint path.
    checkpoint_stem = config.get("checkpoint_name", f"{dataset_type}_{model_config['name']}")
    checkpoint_path = Path("outputs/checkpoints") / f"{checkpoint_stem}.pt"
    best_checkpoint_path = Path("outputs/checkpoints") / f"{checkpoint_stem}_best.pt"

    run_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
    if args.subset_per_class is not None:
        subset_label = f"subset{args.subset_per_class}"
    elif args.limit is not None:
        subset_label = f"limit{args.limit}"
    else:
        subset_label = "full"
    run_name = f"{dataset_type}_{model_config['name']}_{subset_label}_ep{epochs}_{run_id}"

    run_params = {
        "model": model_config["name"],
        "dataset": dataset_type,
        "batch_size": config["batch_size"],
        "lr": training_config["lr"],
        "epochs": epochs,
        "subset_per_class": args.subset_per_class,
        "limit": args.limit,
        "actual_train_size": len(train_dataset),
        "actual_val_size": len(val_dataset),
        "augmentation_preset": config["augmentation_preset"],
        "class_weighted_loss": training_config["class_weighted_loss"],
        "mixed_precision": use_amp,
        "early_stopping_patience": early_stopping_patience,
        "lr_scheduler_patience": lr_scheduler_patience,
        "weight_decay": training_config.get("weight_decay", 0.01),
        "label_smoothing": training_config.get("label_smoothing", 0.0),
    }
    tracking_context = (
        nullcontext() if args.no_wandb else wandb_run("visionlab", run_name, run_params)
    )
    run_metadata = collect_run_metadata(args.config, config.get("dataset_root"))
    mlflow_context = (
        nullcontext()
        if args.no_mlflow
        else mlflow_run("visionlab", run_name, run_params, run_metadata=run_metadata)
    )

    run_start = time.perf_counter()
    val_metrics = None
    best_val_metrics = None
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    with tracking_context, mlflow_context:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
            val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
            elapsed = time.perf_counter() - epoch_start
            val_macro_f1 = macro_f1_score(val_metrics["y_true"], val_metrics["y_pred"])
            scheduler.step(val_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_macro_f1:.4f} lr={current_lr:.2e} | "
                f"{elapsed:.1f}s"
            )

            epoch_extra_metrics = {
                "val_macro_f1": val_macro_f1,
                "epoch_duration_seconds": elapsed,
                "lr": current_lr,
            }
            if not args.no_wandb:
                log_epoch_metrics(epoch, train_metrics, val_metrics, extra_metrics=epoch_extra_metrics)
            if not args.no_mlflow:
                log_epoch_metrics_mlflow(epoch, train_metrics, val_metrics, extra_metrics=epoch_extra_metrics)

            save_checkpoint(
                model,
                checkpoint_path,
                architecture=model_config["name"],
                dataset_type=dataset_type,
                label_map_path=config["label_map_path"],
                epoch=epoch,
                optimizer=optimizer,
                scaler=scaler,
                run_metadata=run_metadata,
                label_map=pipeline["label_map"],
                image_size=config["image_size"],
            )

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_val_metrics = val_metrics
                epochs_without_improvement = 0
                save_checkpoint(
                    model,
                    best_checkpoint_path,
                    architecture=model_config["name"],
                    dataset_type=dataset_type,
                    label_map_path=config["label_map_path"],
                    epoch=epoch,
                    run_metadata=run_metadata,
                    label_map=pipeline["label_map"],
                    image_size=config["image_size"],
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"\nEarly stopping: val_macro_f1 hasn't improved for "
                        f"{early_stopping_patience} epochs (best={best_val_macro_f1:.4f})"
                    )
                    break

        run_duration = time.perf_counter() - run_start
        if best_val_metrics is not None:
            report = build_evaluation_report(
                best_val_metrics["y_true"], best_val_metrics["y_pred"], idx_to_label
            )
            report_path = Path("outputs/reports") / f"{checkpoint_stem}_eval_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"\nMacro-F1 (best epoch by val_macro_f1): {report['macro_f1']:.4f}")
            print("Worst 10 classes (by F1):")
            for row in report["worst_classes"]:
                print(
                    f"  {row['label']:<40} precision={row['precision']:.3f} "
                    f"recall={row['recall']:.3f} f1={row['f1']:.3f} support={row['support']}"
                )
            print("Top 10 confused class pairs (true -> predicted, count):")
            for row in report["top_confused_pairs"]:
                print(f"  {row['true_label']} -> {row['predicted_label']}: {row['count']}")
            print(f"Full report saved to {report_path}")

            if not args.no_wandb:
                log_evaluation_report(report)
            if not args.no_mlflow:
                log_mlflow_evaluation_report(report)
                log_mlflow_model(
                    model,
                    best_checkpoint_path,
                    registered_model_name=f"{dataset_type}_{model_config['name']}",
                    run_metadata=run_metadata,
                    label_map=pipeline["label_map"],
                    image_size=config["image_size"],
                )

        if not args.no_wandb:
            log_run_duration(run_duration)
        if not args.no_mlflow and best_val_metrics is not None:
            mlflow.log_metric("run_duration_seconds", run_duration)
        print(f"\nTotal run duration: {run_duration:.1f}s")

    print(f"Latest checkpoint saved to {checkpoint_path}")
    print(f"Best checkpoint (val_macro_f1={best_val_macro_f1:.4f}) saved to {best_checkpoint_path}")


if __name__ == "__main__":
    main()
