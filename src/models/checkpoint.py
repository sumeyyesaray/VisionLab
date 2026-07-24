from datetime import datetime, timezone
from pathlib import Path

import torch

from src.models.base_model import BaseModel
from src.models.registry import build_model


def save_checkpoint(
    model: BaseModel,
    path: str | Path,
    architecture: str,
    dataset_type: str,
    label_map_path: str,
    epoch: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: "torch.amp.GradScaler | None" = None,
    run_metadata: dict | None = None,
    label_map: dict[str, int] | None = None,
    image_size: int | None = None,
) -> None:
    """Save a checkpoint. Pass `epoch`/`optimizer`/`scaler` too when the
    checkpoint needs to support resuming training (not just inference) —
    see `load_checkpoint` and scripts/train_baseline.py's `--resume` flag.

    Pass `run_metadata` (see src/utils/run_metadata.collect_run_metadata) to
    make the checkpoint self-describing: which git commit, which exact
    config, and which DVC data version produced it. Without it, a checkpoint
    found six months from now is just weights with a guessed provenance.

    Pass `label_map` and `image_size` so the checkpoint is enough on its own
    to serve predictions — no dependency on `label_map_path` resolving on
    whatever machine is running inference, and no need to go dig up the
    original training config just to know what preprocessing to apply. See
    src/inference/predictor.py.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": architecture,
        "num_classes": model.num_classes,
        "dataset_type": dataset_type,
        "label_map_path": label_map_path,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
    }
    if run_metadata is not None:
        checkpoint["run_metadata"] = run_metadata
    if label_map is not None:
        checkpoint["label_map"] = label_map
    if image_size is not None:
        checkpoint["image_size"] = image_size
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    path: str | Path, map_location: str | None = None
) -> tuple[BaseModel, dict]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    model = build_model(checkpoint["architecture"], num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
