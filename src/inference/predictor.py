"""Runs a single prediction against a loaded ServableModel. Domain-agnostic
on purpose — it only knows about tensors, softmax, and a label lookup table,
never a dataset name, so the same code path serves mushroom, flower, or
whatever gets registered next.
"""

import torch
from PIL import Image

from src.data.transforms import build_val_transform
from src.inference.registry import ServableModel


def predict(servable: ServableModel, image: Image.Image) -> dict:
    transform = build_val_transform(servable.image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = servable.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    predicted_idx = int(torch.argmax(probs))
    return {
        "model": servable.name,
        "predicted_class": servable.idx_to_label[predicted_idx],
        "confidence": float(probs[predicted_idx]),
    }
