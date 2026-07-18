import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.models.base_model import BaseModel


class ResNet50(BaseModel):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__(num_classes)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("fc.")

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_except_last_block(self) -> None:
        """Freeze everything except `layer4` (the last residual block) and
        `fc` — a middle ground between `freeze_backbone` (classifier only)
        and `unfreeze_backbone` (everything), used for Stage 2 fine-tuning
        on a small subset where full unfreezing risks catastrophic
        forgetting of the other classes (see
        notebooks/08_mushroom_confusion_diagnosis.ipynb, two-stage section)."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("layer4.") or name.startswith("fc.")

