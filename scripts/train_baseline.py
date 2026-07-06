"""Baseline training script — wires config, pipeline, model, and engine together.

Run from the project root:

    .venv\\Scripts\\python.exe scripts\\train_baseline.py --config configs/mushroom.yaml

Wrapped in `if __name__ == "__main__":` because the configs use
`num_workers > 0` — on Windows, multiprocessing's "spawn" start method
requires this guard (see scripts/check_pipeline.py).
"""

import argparse
from pathlib import Path

import torch

from src.data.config import load_config
from src.data.dataloader import build_dataloader, build_pipeline
from src.models.checkpoint import save_checkpoint
from src.models.registry import build_model
from src.training.engine import evaluate, train_one_epoch
from src.training.losses import build_loss, compute_class_weights
from src.training.optimizers import build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override config's training.epochs")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N training/validation images (quick smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pipeline = build_pipeline(config)
    num_classes = len(pipeline["label_map"])

    train_dataset = pipeline["train_dataset"]
    val_dataset = pipeline["val_dataset"]
    if args.limit is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.limit, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(args.limit, len(val_dataset))))

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

    model_config = config["model"]
    model = build_model(
        model_config["name"], num_classes=num_classes, pretrained=model_config["pretrained"]
    ).to(device)
    if model_config["freeze_backbone"]:
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    training_config = config["training"]
    weight = None
    if training_config["class_weighted_loss"]:
        label_counts = pipeline["frames"]["train"]["label"].value_counts().to_dict()
        weight = compute_class_weights(label_counts, pipeline["label_map"]).to(device)

    criterion = build_loss(training_config["loss"], weight=weight)
    optimizer = build_optimizer(training_config["optimizer"], model.parameters(), lr=training_config["lr"])

    epochs = args.epochs if args.epochs is not None else training_config["epochs"]
    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

    checkpoint_path = Path("outputs/checkpoints") / f"{config['dataset_type']}_{model_config['name']}.pt"
    save_checkpoint(
        model,
        checkpoint_path,
        architecture=model_config["name"],
        dataset_type=config["dataset_type"],
        label_map_path=config["label_map_path"],
    )
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
