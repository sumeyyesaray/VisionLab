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
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": architecture,
            "num_classes": model.num_classes,
            "dataset_type": dataset_type,
            "label_map_path": label_map_path,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    path: str | Path, map_location: str | None = None
) -> tuple[BaseModel, dict]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    model = build_model(checkpoint["architecture"], num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
