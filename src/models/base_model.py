from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Common interface every Model Zoo architecture must implement.

    Freeze/unfreeze are first-class methods (not an afterthought) because
    the Sprint 4 research notes settled on "default to fine-tune, freeze is
    an explicit opt-in" — see notebooks/04_model_research_part1.ipynb,
    Reading Note 1.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def freeze_backbone(self) -> None:
        """Freeze every parameter except the final classification layer."""

    @abstractmethod
    def unfreeze_backbone(self) -> None:
        """Unfreeze the whole network for full fine-tuning."""
