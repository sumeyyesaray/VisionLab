"""DVC preprocessing stage for the flower dataset.

Reads datasets/extracted/flower_data's train/valid/test folder structure via
the same `load_flower_dataframes` used at training time (folder-name ->
species-name mapping via cat_to_name.json) and writes a canonical,
version-controlled copy to datasets/processed/flower/{train,val,test}.csv.

Run via `dvc repro preprocess_flower`, or directly:

    python scripts/dvc_pipeline/preprocess_flower.py
"""

from pathlib import Path

from src.data.loaders import load_flower_dataframes

SOURCE_ROOT = Path("datasets/extracted/flower_data")
OUTPUT_ROOT = Path("datasets/processed/flower")


def main() -> None:
    frames = load_flower_dataframes(SOURCE_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        out_path = OUTPUT_ROOT / f"{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"{split}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
