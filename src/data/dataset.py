from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):
    """Generic image classification dataset.

    Works for any dataframe that follows the `image_path` / `local_path` /
    `label` schema shared by every VisionLab dataset (see the Sprint 1
    exploration notebooks) — it has no knowledge of which dataset it's
    reading. Dataset-specific loading lives in `loaders.py` instead.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_map: Optional[dict[str, int]] = None,
        transform: Optional[Callable] = None,
        path_col: str = "local_path",
        label_col: str = "label",
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform
        self.path_col = path_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        image = Image.open(Path(row[self.path_col])).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.label_map is not None and self.label_col in self.dataframe.columns:
            label = self.label_map[row[self.label_col]]
        else:
            # Unlabeled split (e.g. the flower dataset's test set).
            label = -1

        return image, label
