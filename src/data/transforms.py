import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Each preset only controls appearance/geometry jitter. Crop aggressiveness is
# handled separately via `random_resized_crop_scale`, since that's the knob the
# Sprint 1 findings actually disagreed on between the two datasets (mushroom
# wants a narrow scale close to a plain resize, flower benefits from a wider
# one) — see notebooks/02_flower_dataset_exploration.ipynb, section 12.
AUGMENTATION_PRESETS = {
    "none": [],
    "light": [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ],
    "medium": [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ],
    "heavy": [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    ],
}


def build_train_transform(
    image_size: int,
    augmentation_preset: str = "light",
    random_resized_crop_scale: tuple[float, float] = (0.8, 1.0),
) -> transforms.Compose:
    if augmentation_preset not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown augmentation_preset: {augmentation_preset!r}")

    augmentations = AUGMENTATION_PRESETS[augmentation_preset]

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=random_resized_crop_scale),
            *augmentations,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_val_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean
