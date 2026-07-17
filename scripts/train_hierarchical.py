"""Two-head (genus + species) hierarchical training for mushroom.

Separate from scripts/train_baseline.py because HierarchicalResNet50 returns
two logit tensors, not one — it needs its own loss/eval loop rather than the
shared single-task src/training/engine.py path used by the rest of the
project (flower, plain ResNet50/EfficientNet mushroom runs).

Run from the project root:

    python scripts/train_hierarchical.py --config configs/mushroom_hierarchical.yaml --epochs 30 --subset-per-class 300

Logs to Weights & Biases by default — disable with --no-wandb. Saves a
resumable "latest" checkpoint every epoch and a separate "best" checkpoint
(by val species macro-F1) — same convention as train_baseline.py, but the
checkpoint dict has a `genus_label_map_path` key on top of the usual fields
since this model needs both label maps to reload.
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Subset

from src.data.config import load_config
from src.data.dataloader import build_dataloader, build_pipeline
from src.data.dataset import ImageClassificationDataset
from src.data.label_map import invert_label_map, load_label_map
from src.data.sampling import subsample_per_class
from src.evaluation.metrics import build_evaluation_report, macro_f1_score
from src.models.hierarchical_resnet import HierarchicalResNet50
from src.training.optimizers import build_optimizer
from src.training.tracking import log_epoch_metrics, log_evaluation_report, log_run_duration, wandb_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs")
    parser.add_argument(
        "--limit", type=int, default=None, help="Truncate train/val to the first N images"
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
    species_label_map = pipeline["label_map"]
    num_species = len(species_label_map)
    idx_to_species = invert_label_map(species_label_map)

    genus_label_map_path = model_config["genus_label_map_path"]
    genus_label_map = load_label_map(genus_label_map_path)
    num_genera = len(genus_label_map)

    # species idx -> genus idx lookup, built once, indexed on the fly per batch
    species_to_genus = torch.tensor(
        [genus_label_map[idx_to_species[i].split()[0]] for i in range(num_species)],
        dtype=torch.long,
    )

    train_dataset, val_dataset = build_datasets(pipeline, args)
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)}")
    print(f"{num_species} species across {num_genera} genera")

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

    model = HierarchicalResNet50(
        num_genera=num_genera, num_species=num_species, pretrained=model_config["pretrained"]
    ).to(device)
    if model_config["freeze_backbone"]:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    genus_criterion = nn.CrossEntropyLoss()
    species_criterion = nn.CrossEntropyLoss()
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
    early_stopping_patience = training_config.get("early_stopping_patience", 8)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from {args.resume}, continuing at epoch {start_epoch + 1}")

    epochs = args.epochs if args.epochs is not None else training_config["epochs"]
    model_name = model_config["name"]
    checkpoint_path = Path("outputs/checkpoints") / f"{dataset_type}_{model_name}.pt"
    best_checkpoint_path = Path("outputs/checkpoints") / f"{dataset_type}_{model_name}_best.pt"

    def save(path: Path, epoch: int, include_optimizer: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "architecture": model_name,
            "num_species": num_species,
            "num_genera": num_genera,
            "dataset_type": dataset_type,
            "label_map_path": config["label_map_path"],
            "genus_label_map_path": genus_label_map_path,
            "epoch": epoch,
        }
        if include_optimizer:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, path)

    run_id = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
    subset_label = f"subset{args.subset_per_class}" if args.subset_per_class is not None else "full"
    run_name = f"{dataset_type}_{model_name}_{subset_label}_ep{epochs}_{run_id}"
    run_params = {
        "model": model_name,
        "dataset": dataset_type,
        "batch_size": config["batch_size"],
        "lr": training_config["lr"],
        "epochs": epochs,
        "subset_per_class": args.subset_per_class,
        "actual_train_size": len(train_dataset),
        "actual_val_size": len(val_dataset),
        "augmentation_preset": config["augmentation_preset"],
        "early_stopping_patience": early_stopping_patience,
        "lr_scheduler_patience": lr_scheduler_patience,
        "num_genera": num_genera,
        "num_species": num_species,
    }
    tracking_context = (
        nullcontext() if args.no_wandb else wandb_run("visionlab", run_name, run_params)
    )

    run_start = time.perf_counter()
    best_val_macro_f1 = -1.0
    best_species_y_true = None
    best_species_y_pred = None
    epochs_without_improvement = 0

    with tracking_context:
        for epoch in range(start_epoch, epochs):
            epoch_start = time.perf_counter()

            model.train()
            train_species_loss = train_genus_loss = 0.0
            train_correct = train_total = 0
            for images, species_labels in train_loader:
                images = images.to(device)
                species_labels = species_labels.to(device)
                genus_labels = species_to_genus[species_labels.cpu()].to(device)

                optimizer.zero_grad()
                genus_logits, species_logits = model(images)
                g_loss = genus_criterion(genus_logits, genus_labels)
                s_loss = species_criterion(species_logits, species_labels)
                loss = g_loss + s_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_genus_loss += g_loss.item() * images.size(0)
                train_species_loss += s_loss.item() * images.size(0)
                train_correct += (species_logits.argmax(dim=1) == species_labels).sum().item()
                train_total += images.size(0)

            train_metrics = {
                "species_loss": train_species_loss / train_total,
                "genus_loss": train_genus_loss / train_total,
                "accuracy": train_correct / train_total,
            }

            model.eval()
            val_species_loss = val_genus_loss = 0.0
            val_species_correct = val_genus_correct = val_total = 0
            y_true, y_pred = [], []
            with torch.no_grad():
                for images, species_labels in val_loader:
                    images = images.to(device)
                    species_labels = species_labels.to(device)
                    genus_labels = species_to_genus[species_labels.cpu()].to(device)

                    genus_logits, species_logits = model(images)
                    g_loss = genus_criterion(genus_logits, genus_labels)
                    s_loss = species_criterion(species_logits, species_labels)

                    species_preds = species_logits.argmax(dim=1)
                    genus_preds = genus_logits.argmax(dim=1)

                    val_genus_loss += g_loss.item() * images.size(0)
                    val_species_loss += s_loss.item() * images.size(0)
                    val_species_correct += (species_preds == species_labels).sum().item()
                    val_genus_correct += (genus_preds == genus_labels).sum().item()
                    val_total += images.size(0)
                    y_true.extend(species_labels.cpu().tolist())
                    y_pred.extend(species_preds.cpu().tolist())

            val_metrics = {
                "species_loss": val_species_loss / val_total,
                "genus_loss": val_genus_loss / val_total,
                "species_accuracy": val_species_correct / val_total,
                "genus_accuracy": val_genus_correct / val_total,
            }
            val_total_loss = val_metrics["species_loss"] + val_metrics["genus_loss"]
            elapsed = time.perf_counter() - epoch_start
            val_macro_f1 = macro_f1_score(y_true, y_pred)
            scheduler.step(val_total_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"epoch {epoch + 1}/{epochs} | "
                f"train_species_loss={train_metrics['species_loss']:.4f} "
                f"train_genus_loss={train_metrics['genus_loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_species_loss={val_metrics['species_loss']:.4f} "
                f"val_genus_loss={val_metrics['genus_loss']:.4f} "
                f"val_species_acc={val_metrics['species_accuracy']:.4f} "
                f"val_genus_acc={val_metrics['genus_accuracy']:.4f} "
                f"val_macro_f1={val_macro_f1:.4f} lr={current_lr:.2e} | {elapsed:.1f}s"
            )

            if not args.no_wandb:
                log_epoch_metrics(
                    epoch,
                    {"loss": train_metrics["species_loss"], "accuracy": train_metrics["accuracy"]},
                    {"loss": val_metrics["species_loss"], "accuracy": val_metrics["species_accuracy"]},
                    extra_metrics={
                        "val_macro_f1": val_macro_f1,
                        "train_genus_loss": train_metrics["genus_loss"],
                        "val_genus_loss": val_metrics["genus_loss"],
                        "val_genus_accuracy": val_metrics["genus_accuracy"],
                        "epoch_duration_seconds": elapsed,
                        "lr": current_lr,
                    },
                )

            save(checkpoint_path, epoch, include_optimizer=True)

            if val_macro_f1 > best_val_macro_f1:
                best_val_macro_f1 = val_macro_f1
                best_species_y_true, best_species_y_pred = y_true, y_pred
                epochs_without_improvement = 0
                save(best_checkpoint_path, epoch, include_optimizer=False)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"\nEarly stopping: val_macro_f1 hasn't improved for "
                        f"{early_stopping_patience} epochs (best={best_val_macro_f1:.4f})"
                    )
                    break

        run_duration = time.perf_counter() - run_start
        if best_species_y_true is not None:
            report = build_evaluation_report(best_species_y_true, best_species_y_pred, idx_to_species)
            report_path = Path("outputs/reports") / f"{dataset_type}_{model_name}_eval_report.json"
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

        if not args.no_wandb:
            log_run_duration(run_duration)
        print(f"\nTotal run duration: {run_duration:.1f}s")

    print(f"Latest checkpoint saved to {checkpoint_path}")
    print(f"Best checkpoint (val_macro_f1={best_val_macro_f1:.4f}) saved to {best_checkpoint_path}")


if __name__ == "__main__":
    main()
