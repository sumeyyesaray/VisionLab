"""Turns verified-but-not-yet-used flywheel predictions into a training
snapshot: copies each image under `<output_root>/images/` and writes a
`train.csv` in the same `local_path`/`label` schema every other dataset
loader produces (see src/data/dataset.py), so it's a drop-in extra training
source via train_baseline.py's --extra-train-csv.

Rows whose true_label isn't already in the model's label map are dropped —
teaching the classifier a genuinely new species needs a wider output layer,
not just more training images, so that's a separate, deliberate change, not
something a retraining job should do silently.
"""

import shutil
from pathlib import Path

import pandas as pd

from src.data.label_map import load_label_map
from src.flywheel.store import get_verified_unused, mark_used_in_training


def export_flywheel_snapshot(
    dataset_type: str, label_map_path: str, output_root: str | Path
) -> Path | None:
    records = get_verified_unused(dataset_type)
    if not records:
        print(f"No verified, unused flywheel predictions for {dataset_type!r}")
        return None

    label_map = load_label_map(label_map_path)
    output_root = Path(output_root)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    used_ids = []
    skipped_unknown_label = 0
    skipped_missing_file = 0

    for record in records:
        if record["true_label"] not in label_map:
            skipped_unknown_label += 1
            continue
        src_path = Path(record["image_path"])
        if not src_path.exists():
            skipped_missing_file += 1
            continue
        dst_path = images_dir / f"{record['id']}{src_path.suffix}"
        shutil.copy2(src_path, dst_path)
        rows.append({"local_path": str(dst_path), "label": record["true_label"]})
        used_ids.append(record["id"])

    if skipped_unknown_label:
        print(f"Skipped {skipped_unknown_label} records: true_label not in {label_map_path}")
    if skipped_missing_file:
        print(f"Skipped {skipped_missing_file} records: source image file missing")

    if not rows:
        print("Nothing exportable after filtering")
        return None

    csv_path = output_root / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    mark_used_in_training(used_ids)

    print(f"Exported {len(rows)} images -> {csv_path}")
    return csv_path
