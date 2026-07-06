from src.models.base_model import BaseModel
from src.models.resnet import ResNet50

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "resnet50": ResNet50,
}


def build_model(name: str, num_classes: int, **kwargs) -> BaseModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
