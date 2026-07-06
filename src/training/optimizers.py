from typing import Iterable

import torch

OPTIMIZER_REGISTRY = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


def build_optimizer(name: str, params: Iterable, lr: float, **kwargs) -> torch.optim.Optimizer:
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name!r}. Available: {sorted(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name](params, lr=lr, **kwargs)
