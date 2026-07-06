import torch
import torch.nn as nn

LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
}


def compute_class_weights(
    label_counts: dict[str, int], label_map: dict[str, int]
) -> torch.Tensor:
    """Inverse-frequency class weights, ordered by label_map index.

    Only meaningful for imbalanced datasets (flower) — see the Sprint 1
    findings on class balance in notebooks/02_flower_dataset_exploration.ipynb.
    """
    num_classes = len(label_map)
    total = sum(label_counts.values())
    weights = torch.zeros(num_classes)

    for label, idx in label_map.items():
        count = label_counts.get(label, 0)
        weights[idx] = total / (num_classes * count) if count > 0 else 0.0

    return weights


def build_loss(name: str, weight: torch.Tensor | None = None, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name!r}. Available: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](weight=weight, **kwargs)
