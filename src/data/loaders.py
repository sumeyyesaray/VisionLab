import json
from pathlib import Path

import pandas as pd


def load_mushroom_dataframes(dataset_root: str | Path) -> dict[str, pd.DataFrame]:
    dataset_root = Path(dataset_root)
    image_root = dataset_root / "merged_dataset"

    frames = {}
    for split, filename in [("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")]:
        df = pd.read_csv(dataset_root / filename)
        df["local_path"] = df["image_path"].str.replace(
            "/kaggle/working/merged_dataset",
            str(image_root),
            regex=False,
        )
        frames[split] = df

    return frames


def _build_flower_split(split_dir: Path, cat_to_name: dict[str, str]) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        label = cat_to_name.get(class_id, class_id)
        for img_path in class_dir.glob("*.jpg"):
            rows.append(
                {
                    "image_path": str(img_path),
                    "local_path": str(img_path),
                    "class_id": class_id,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def load_flower_dataframes(dataset_root: str | Path) -> dict[str, pd.DataFrame]:
    dataset_root = Path(dataset_root)

    with open(dataset_root / "cat_to_name.json", encoding="utf-8") as f:
        cat_to_name = json.load(f)

    train_df = _build_flower_split(dataset_root / "train", cat_to_name)
    val_df = _build_flower_split(dataset_root / "valid", cat_to_name)

    # The original Udacity packaging ships the test split without labels.
    test_df = pd.DataFrame(
        {"image_path": [str(p) for p in sorted((dataset_root / "test").glob("*.jpg"))]}
    )
    test_df["local_path"] = test_df["image_path"]

    return {"train": train_df, "val": val_df, "test": test_df}


DATASET_LOADERS = {
    "mushroom": load_mushroom_dataframes,
    "flower": load_flower_dataframes,
}


def load_dataframes(dataset_type: str, dataset_root: str | Path) -> dict[str, pd.DataFrame]:
    if dataset_type not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset_type: {dataset_type!r}")
    return DATASET_LOADERS[dataset_type](dataset_root)
