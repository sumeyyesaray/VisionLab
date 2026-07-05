"""Quick sanity check for the Sprint 3 data pipeline.

Run from the project root:

    .venv\\Scripts\\python.exe scripts\\check_pipeline.py

Wrapped in `if __name__ == "__main__":` because the configs use
`num_workers > 0` — on Windows, multiprocessing's "spawn" start method
re-imports this file in each worker process, which crashes with a
RuntimeError unless the entry point is guarded like this.
"""

from src.data.config import load_config
from src.data.dataloader import build_pipeline


def main() -> None:
    for config_path in ["configs/mushroom.yaml", "configs/flower.yaml"]:
        config = load_config(config_path)
        pipeline = build_pipeline(config)

        image, label = pipeline["train_dataset"][0]
        images, labels = next(iter(pipeline["train_loader"]))

        print(f"[{config['dataset_type']}]")
        print(f"  classes         : {len(pipeline['label_map'])}")
        print(f"  train/val/test  : {len(pipeline['train_dataset'])}"
              f" / {len(pipeline['val_dataset'])}"
              f" / {len(pipeline['test_dataset'])}")
        print(f"  dataset[0] shape: {tuple(image.shape)} (label index {label})")
        print(f"  batch shape     : {tuple(images.shape)} / {tuple(labels.shape)}")
        print()


if __name__ == "__main__":
    main()
