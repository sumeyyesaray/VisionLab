# Hyperparameters & Configurable Variables (as of Sprint 5)

Reference list of every tunable value in the codebase so far, where it lives,
and why it currently has the value it does. Anything under "Config-level"
changes per dataset without touching code; everything else is currently a
fixed constant in `src/`.

*Data pipeline parameters (Sprint 3) are below in their own section;
Model Zoo / Training Engine parameters (Sprint 5) are at the end.*

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

## Model-level config (Sprint 5 — `model:` section in `configs/*.yaml`)

| Key | Mushroom | Flower | Notes |
|---|---|---|---|
| `name` | `resnet50` | `resnet50` | only registry entry so far — first model per the Sprint 4 trial order |
| `pretrained` | `true` | `true` | ImageNet-pretrained weights via `torchvision.models.resnet50` |
| `freeze_backbone` | `false` | `false` | default to fine-tune, not freeze, per Reading Note 1 (Yosinski et al.) — `BaseModel.freeze_backbone()`/`unfreeze_backbone()` exist and are verified working, just not the default |

## Training-level config (Sprint 5 — `training:` section in `configs/*.yaml`)

| Key | Mushroom | Flower | Notes |
|---|---|---|---|
| `loss` | `cross_entropy` | `cross_entropy` | only registry entry so far |
| `class_weighted_loss` | `false` | `true` | mushroom is balanced (`std=0`, Sprint 1) → no weighting; flower is imbalanced (`std≈35.4`) → inverse-frequency class weights via `compute_class_weights()` |
| `optimizer` | `adamw` | `adamw` | `OPTIMIZER_REGISTRY` also has `sgd`/`adam` available but unused so far |
| `lr` | `0.0001` | `0.0001` | a reasonable AdamW default, **not tuned** — no learning-rate search has been run |
| `epochs` | `1` | `1` | placeholder for smoke-testing the pipeline, not a real training budget — see open questions |

## Model-level fixed values (`src/models/`)

| Variable | Value | Where |
|---|---|---|
| Final layer replacement | `nn.Linear(in_features=2048, num_classes)` | `resnet.py` — `in_features` comes from the pretrained `resnet50.fc.in_features`, not hardcoded |
| Freeze criterion | parameter name starts with `"fc."` | `resnet.py` — the only unfrozen part when `freeze_backbone()` is called |
| `MODEL_REGISTRY` | `{"resnet50": ResNet50}` | `registry.py` — single entry; EfficientNet-B3 is next per the Sprint 4 wrap-up |
| Checkpoint fields | `model_state_dict`, `architecture`, `num_classes`, `dataset_type`, `label_map_path`, `saved_at` | `checkpoint.py` — one `.pt` file per checkpoint, no separate metadata sidecar |
| Checkpoint path convention | `outputs/checkpoints/{dataset_type}_{model_name}.pt` | `scripts/train_baseline.py` — not enforced inside `checkpoint.py` itself |

## Training Engine fixed values (`src/training/`)

| Variable | Value | Where |
|---|---|---|
| `LOSS_REGISTRY` | `{"cross_entropy": nn.CrossEntropyLoss}` | `losses.py` |
| `OPTIMIZER_REGISTRY` | `{"sgd": SGD, "adam": Adam, "adamw": AdamW}` | `optimizers.py` |
| Class weight formula | `total / (num_classes * count)` per class (inverse frequency) | `losses.py` — `compute_class_weights()` |
| LR schedule | none | `engine.py` — fixed learning rate for the whole run, no scheduler yet |
| Early stopping | none | `engine.py` — always runs the configured number of epochs |
| Gradient clipping | none | `engine.py` |
| `device` | `"cuda" if torch.cuda.is_available() else "cpu"` | auto-detected in `train_baseline.py`/notebook 05, not a config field |

## `scripts/train_baseline.py` CLI defaults

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | `None` (falls back to `training.epochs` in config) | override for real runs |
| `--limit` | `None` (full dataset) | set to a small number for a quick smoke test, as used in notebook 05 |

## Notebook-only constants (`05_model_training_report.ipynb`)

| Constant | Value | Purpose |
|---|---|---|
| Tiny-subset training check size | 32 train images / 16 val images | just enough to prove gradients flow end-to-end, not a real training signal |
| Tiny-subset epochs | 3 | enough to see the loss trend down on 32 images |
| Tiny-subset batch size | 8 | |

## Known open questions (not yet resolved by a hyperparameter)

- Whether mushroom's 4,080 images/class are all distinct originals or include
  near-duplicates (would affect how conservative `augmentation_preset: light`
  needs to stay).
- How the flower dataset's unlabeled test set will be scored (official
  `imagelabels.mat` mapping vs. a new stratified split) — affects whether
  `test_loader` for flower stays label-less or gets a real `label_map`.
- `epochs: 1` and `lr: 0.0001` in both configs are placeholders for pipeline
  verification, not the result of any experiment — a real training budget
  and learning-rate search are still open (Benchmark sprint).
- No learning-rate scheduler or early stopping exists yet in `engine.py` —
  both were named as still-missing in notebook 05's "Next Steps".
- `medium`/`heavy`-preset augmentation ablation and the modern training
  recipe (AdamW + RandAugment + Mixup/CutMix + stochastic depth, per the
  ConvNeXt note) haven't been implemented — ConvNeXt-T is deliberately
  deferred until they are (Sprint 4 wrap-up).
