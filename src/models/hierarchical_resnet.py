import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class HierarchicalResNet50(nn.Module):
    """Two-head ResNet-50: one shared backbone, separate genus and species
    classification heads. Not a `BaseModel` subclass — `BaseModel.forward`
    returns a single tensor, but this returns `(genus_logits,
    species_logits)`, so it needs its own training/eval loop
    (`scripts/train_hierarchical.py`) rather than the shared single-task
    `src/training/engine.py` path.

    The idea (coarse-to-fine / hierarchical classification): training the
    backbone to also predict genus pushes it to learn genus-discriminative
    features in its shared representation, which the species head can then
    build on — see notebooks/08_mushroom_confusion_diagnosis.ipynb for why
    (species errors cluster almost entirely within genus).
    """

    def __init__(self, num_genera: int, num_species: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # drop the original 1000-way head, keep pooled features
        self.backbone = backbone
        self.genus_head = nn.Linear(in_features, num_genera)
        self.species_head = nn.Linear(in_features, num_species)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return self.genus_head(features), self.species_head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
