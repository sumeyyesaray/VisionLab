import json
from pathlib import Path

import pandas as pd


def build_label_map(labels: pd.Series) -> dict[str, int]:
    unique_labels = sorted(labels.unique())
    return {label: idx for idx, label in enumerate(unique_labels)}


def save_label_map(label_map: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def load_label_map(path: str | Path) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def invert_label_map(label_map: dict[str, int]) -> dict[int, str]:
    return {idx: label for label, idx in label_map.items()}


def load_or_build_label_map(path: str | Path, labels: pd.Series) -> dict[str, int]:
    path = Path(path)
    if path.exists():
        return load_label_map(path)

    label_map = build_label_map(labels)
    save_label_map(label_map, path)
    return label_map
