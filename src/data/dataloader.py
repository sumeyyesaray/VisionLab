import pandas as pd
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import ImageClassificationDataset
from src.data.label_map import load_or_build_label_map
from src.data.loaders import load_dataframes
from src.data.transforms import build_train_transform, build_val_transform


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_pipeline(config: dict) -> dict:
    """Assemble dataframes, label map, datasets and dataloaders from a config dict.

    This is the only place that wires the other `src/data` modules together —
    it makes no decisions of its own (see the "montaj hattı" note in Sprint 3
    planning).
    """
    frames = load_dataframes(config["dataset_type"], config["dataset_root"])

    extra_train_csv = config.get("extra_train_csv")
    if extra_train_csv:
        extra_df = pd.read_csv(extra_train_csv)
        print(f"Adding {len(extra_df)} extra training rows from {extra_train_csv}")
        frames["train"] = pd.concat([frames["train"], extra_df], ignore_index=True)

    label_map = load_or_build_label_map(config["label_map_path"], frames["train"]["label"])

    train_transform = build_train_transform(
        image_size=config["image_size"],
        augmentation_preset=config["augmentation_preset"],
        random_resized_crop_scale=tuple(config["random_resized_crop_scale"]),
    )
    val_transform = build_val_transform(image_size=config["image_size"])

    train_dataset = ImageClassificationDataset(frames["train"], label_map, train_transform)
    val_dataset = ImageClassificationDataset(frames["val"], label_map, val_transform)
    test_dataset = ImageClassificationDataset(frames["test"], label_map, val_transform)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
    test_loader = build_dataloader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    return {
        "frames": frames,
        "label_map": label_map,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
