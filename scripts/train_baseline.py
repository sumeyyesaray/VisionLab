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

import torch
from torch.utils.data import Subset

from src.data.config import load_config
from src.data.dataloader import build_dataloader, build_pipeline
from src.data.dataset import ImageClassificationDataset
from src.data.label_map import invert_label_map
from src.data.sampling import subsample_per_class
from src.evaluation.metrics import build_evaluation_report, macro_f1_score
from src.models.checkpoint import load_checkpoint, save_checkpoint
from src.models.registry import build_model
from src.training.engine import evaluate, train_one_epoch
from src.training.losses import build_loss, compute_class_weights
from src.training.optimizers import build_optimizer
from src.training.tracking import (
    log_epoch_metrics,
    log_evaluation_report,
    log_run_duration,
    wandb_run,
)


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

    criterion = build_loss(training_config["loss"], weight=weight)
    optimizer = build_optimizer(training_config["optimizer"], model.parameters(), lr=training_config["lr"])

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
    checkpoint_path = Path("outputs/checkpoints") / f"{dataset_type}_{model_config['name']}.pt"

    run_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
    subset_label = f"subset{args.subset_per_class}" if args.subset_per_class is not None else "full"
    run_name = f"{dataset_type}_{model_config['name']}_{subset_label}_ep{epochs}_{run_id}"

    run_params = {
        "model": model_config["name"],
        "dataset": dataset_type,
        "batch_size": config["batch_size"],
        "lr": training_config["lr"],
        "epochs": epochs,
        "subset_per_class": args.subset_per_class,
        "augmentation_preset": config["augmentation_preset"],
        "class_weighted_loss": training_config["class_weighted_loss"],
        "mixed_precision": use_amp,
    }
    tracking_context = (
        nullcontext() if args.no_wandb else wandb_run("visionlab", run_name, run_params)
    )

    run_start = time.perf_counter()
    val_metrics = None
    with tracking_context:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
            val_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
            elapsed = time.perf_counter() - epoch_start
            val_macro_f1 = macro_f1_score(val_metrics["y_true"], val_metrics["y_pred"])

            print(
                f"epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_macro_f1={val_macro_f1:.4f} | "
                f"{elapsed:.1f}s"
            )

            if not args.no_wandb:
                log_epoch_metrics(
                    epoch,
                    train_metrics,
                    val_metrics,
                    extra_metrics={"val_macro_f1": val_macro_f1, "epoch_duration_seconds": elapsed},
                )

            save_checkpoint(
                model,
                checkpoint_path,
                architecture=model_config["name"],
                dataset_type=dataset_type,
                label_map_path=config["label_map_path"],
                epoch=epoch,
                optimizer=optimizer,
                scaler=scaler,
            )

        run_duration = time.perf_counter() - run_start
        if val_metrics is not None:
            report = build_evaluation_report(val_metrics["y_true"], val_metrics["y_pred"], idx_to_label)
            report_path = Path("outputs/reports") / f"{dataset_type}_{model_config['name']}_eval_report.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"\nMacro-F1 (final epoch, val): {report['macro_f1']:.4f}")
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

        if not args.no_wandb:
            log_run_duration(run_duration)
        print(f"\nTotal run duration: {run_duration:.1f}s")

    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
