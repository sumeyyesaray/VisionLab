"""DVC preprocessing stage for the mushroom dataset.

Reads the raw per-split CSVs under datasets/extracted/mushroom (already
extracted from Kaggle — see .gitignore's note on datasets/raw/ for why
there's no "extract" stage feeding this one) via the same
`load_mushroom_dataframes` used at training time, and writes a canonical,
version-controlled copy to datasets/processed/mushroom/{train,val,test}.csv
with `local_path` already resolved to this machine's filesystem.

Run via `dvc repro preprocess_mushroom`, or directly:

    python scripts/dvc_pipeline/preprocess_mushroom.py
"""

from pathlib import Path

from src.data.loaders import load_mushroom_dataframes

SOURCE_ROOT = Path("datasets/extracted/mushroom")
OUTPUT_ROOT = Path("datasets/processed/mushroom")


def main() -> None:
    frames = load_mushroom_dataframes(SOURCE_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split, df in frames.items():
        out_path = OUTPUT_ROOT / f"{split}.csv"
        df.to_csv(out_path, index=False)
        print(f"{split}: {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
