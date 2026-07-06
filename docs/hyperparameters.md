# Hyperparameters & Configurable Variables (as of Sprint 3)

Reference list of every tunable value in the codebase so far, where it lives,
and why it currently has the value it does. Anything under "Config-level"
changes per dataset without touching code; everything else is currently a
fixed constant in `src/`.

## Config-level (`configs/mushroom.yaml`, `configs/flower.yaml`)

| Key | Mushroom | Flower | Notes |
|---|---|---|---|
| `dataset_type` | `mushroom` | `flower` | selects the loader in `loaders.py` |
| `dataset_root` | `datasets/extracted/mushroom` | `datasets/extracted/flower_data` | |
| `label_map_path` | `configs/label_maps/mushroom_label_map.json` | `configs/label_maps/flower_label_map.json` | built once, then reused (`load_or_build_label_map`) |
| `image_size` | 224 | 224 | same for now; the Sprint 3 benchmark also tested 384 |
| `batch_size` | 32 | 32 | benchmark also tested 16 and 64 |
| `num_workers` | 4 | 4 | not yet tuned — needs an empirical sweep on the training machine (see notebook 03, section 9) |
| `pin_memory` | `false` | `false` | only matters on a CUDA machine; flip to `true` there |
| `augmentation_preset` | `light` | `heavy` | mushroom = balanced+plentiful → conservative; flower = scarce+imbalanced → needs more diversity (Sprint 1 hypothesis) |
| `random_resized_crop_scale` | `[0.9, 1.0]` | `[0.7, 1.0]` | how aggressively `RandomResizedCrop` can zoom in; mushroom kept narrow to avoid distorting fine-grained cues (cap texture, stem shape) |

## Transform-level (`src/data/transforms.py`, currently hardcoded)

| Variable | Value | Used by |
|---|---|---|
| `IMAGENET_MEAN` | `[0.485, 0.456, 0.406]` | `Normalize` in both train/val transforms |
| `IMAGENET_STD` | `[0.229, 0.224, 0.225]` | same |
| val `Resize` factor | `image_size * 1.14` | e.g. 224 → resize to 255 before `CenterCrop(224)` — the standard ImageNet-style ratio |

**Augmentation presets** (`AUGMENTATION_PRESETS` dict — the actual knobs behind `augmentation_preset`):

| Preset | Transforms | Parameters |
|---|---|---|
| `none` | — | no augmentation at all (used by the benchmark's "off" runs) |
| `light` | `RandomHorizontalFlip`, `ColorJitter` | flip `p=0.5`; jitter `brightness=0.1, contrast=0.1` |
| `medium` | `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter` | flip `p=0.5`; rotation `±15°`; jitter `brightness=0.2, contrast=0.2, saturation=0.2` |
| `heavy` | `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomPerspective` | flip `p=0.5`; rotation `±25°`; jitter `brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05`; perspective `distortion_scale=0.2, p=0.3` |

`medium` isn't assigned to either dataset yet — it exists as a middle ground for whenever the Benchmark sprint runs an augmentation ablation.

## Dataset-level (`src/data/dataset.py`, constructor defaults)

| Variable | Default | Notes |
|---|---|---|
| `path_col` | `"local_path"` | which dataframe column holds the file path |
| `label_col` | `"label"` | which column holds the class name; missing column (flower test set) → label `-1` |

## DataLoader-level (`src/data/dataloader.py`)

| Variable | Value | Notes |
|---|---|---|
| `shuffle` (train) | `True` | hardcoded in `build_pipeline`, not config-driven |
| `shuffle` (val/test) | `False` | same |

## Notebook-only constants (exploratory, not part of `src/`)

These live in the Sprint 1/3 notebooks themselves and would need to move into
`src/` if they ever become part of a repeatable pipeline step:

| Constant | Value | Where |
|---|---|---|
| Image sample size for resolution histograms | 1000 | `01_mushroom_dataset_exploration.ipynb`, `02_flower_dataset_exploration.ipynb` |
| Sample size for integrity checks | 3000 | same two notebooks |
| Sample visualization grid | 20 images (4×5) | same two notebooks |
| Benchmark image size sweep | `[224, 384]` | `03_data_pipeline_report.ipynb`, section 9 |
| Benchmark batch size sweep | `[16, 32, 64]` | same |
| Benchmark batches timed per combination | 5 | same |
| Benchmark `num_workers` | fixed at `0` | same — deliberately isolated from the image_size/batch_size/augmentation comparison |

## Known open questions (not yet resolved by a hyperparameter)

- Whether mushroom's 4,080 images/class are all distinct originals or include
  near-duplicates (would affect how conservative `augmentation_preset: light`
  needs to stay).
- How the flower dataset's unlabeled test set will be scored (official
  `imagelabels.mat` mapping vs. a new stratified split) — affects whether
  `test_loader` for flower stays label-less or gets a real `label_map`.
