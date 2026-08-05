# VisionLab

A modular computer vision research and experimentation platform — not a single-dataset classifier, but a reusable pipeline for comparing datasets, architectures, and training strategies under one consistent infrastructure.

## Why VisionLab

Most "I trained a model on Kaggle" projects stop at a notebook. VisionLab is built the other way around: the value isn't any single trained model, it's the platform underneath it — a config-driven data pipeline, a registry-based Model Zoo, a training engine, and an evaluation/explainability layer that all work identically regardless of which dataset or architecture is plugged in.

The guiding question behind every design decision: *would this still work if I swapped the dataset for something completely unrelated tomorrow?*

## Datasets (V1)

| Dataset | Classes | Images | Profile |
|---|---|---|---|
| Mushroom species | 169 | ~720K | Large, perfectly balanced (4,080/class) |
| Oxford Flowers | 102 | ~8.2K | Small, naturally imbalanced (27–206/class) |

These two were picked deliberately as contrasting scenarios — one data-rich and balanced, one data-scarce and imbalanced — to stress-test the pipeline under different conditions rather than just one "easy" case.

## Architecture

```
User → Inference → Experiment Orchestrator (config-driven)
                          │
                    ┌─────┴─────┐
                    │  AI Core  │
                    │           │
        Data Pipeline · Model Zoo · Training Engine · Evaluation
                    └─────┬─────┘
                          │
        Explainability → Benchmark → Experiment Tracking → Deployment
```

```
src/
├── data/            # dataset class, transforms, dataloader, label mapping, config
├── models/          # BaseModel interface, per-architecture implementations, registry/factory
├── training/         # training engine, losses, optimizers, schedulers, checkpointing
├── evaluation/        # metrics: macro-F1, per-class precision/recall, confusion analysis
├── explainability/      # Grad-CAM
├── benchmark/         # architecture / hyperparameter comparison runs
├── inference/         # checkpoint loading → prediction
└── deployment/         # FastAPI / Docker / ONNX export
```

**Design principles:**
- One generic `Dataset` class serves both datasets — it knows nothing about mushrooms or flowers, only a shared `local_path` / `label` schema.
- Everything that varies by dataset (image size, augmentation strength, class weighting) lives in a per-dataset YAML config, not in code.
- Label encoding is a persisted JSON mapping (built once, reused everywhere), not re-fit at runtime — training and inference are guaranteed to agree.
- Models are registered in a factory (`MODEL_REGISTRY`), so adding a new architecture means adding one file + one registry entry, not touching existing code.
- Checkpoints bundle architecture name, `num_classes`, and label-map path alongside the weights, so a saved model is self-describing.

## Current status

**Pipeline:** Data ingestion → Dataset/DataLoader → Model Zoo → Training Engine → Evaluation is fully working end-to-end for both datasets, on both a local GPU (RTX 3050) and a university SLURM cluster (Truba).

**Baseline model (ResNet-50, ImageNet-pretrained, fully fine-tuned):**

| Dataset | Val Accuracy | Val Macro-F1 |
|---|---|---|
| Mushroom | ~84.8% | 0.81 |
| Flower | ~99% | 0.98 |

**Key finding:** Mushroom's val accuracy plateaus with a persistent train/val gap that *did not* respond to stronger regularization (heavier augmentation, weight decay, label smoothing — see `notebooks/08_mushroom_confusion_diagnosis.ipynb`). Error analysis showed the confusion is concentrated on same-genus species pairs (e.g. *Fomitopsis pinicola* ↔ *mounceae*), not random noise. Grad-CAM confirmed the model attends to the correct structures in every misclassified example — this points to a genuine fine-grained visual discrimination limit (resolution/architecture ceiling), not overfitting or a shortcut-learning artifact. This is currently being investigated via higher input resolution and an EfficientNet-B3 comparison.

## Tech stack

- **PyTorch / torchvision** — models and training
- **Weights & Biases** — experiment tracking
- **MLflow** — model registry, "which checkpoint is production"
- **DVC** — data versioning (⚠️ the configured remote, `../.dvc-remote-storage`, sits on the
  same `/arf` Lustre filesystem as the working copy — it protects against accidental local
  deletes/overwrites, but it is **not an off-cluster backup**. If TRUBA scratch storage has a
  purge policy, or `/arf` has an outage, the working copy and the "remote" can be lost
  together. Worth confirming TRUBA's retention policy for `/arf/scratch` and, if there's any
  purge risk, adding a real off-cluster remote — S3/MinIO, or even another filesystem —
  before relying on this for anything that can't be re-downloaded/re-derived.)
- **FastAPI** — model serving (`src/deployment/api.py`)
- **SLURM (Truba HPC cluster)** — large-scale training jobs
- **Config-driven (YAML)** — per-dataset hyperparameters, no code changes needed to switch datasets

See `notebooks/09_mlops_infrastructure.ipynb` for how these fit together,
and [`RETRAINING_POLICY.md`](RETRAINING_POLICY.md) for the continuous
learning loop's retraining/promotion thresholds.

## Roadmap

- [x] Sprint 1 — Dataset exploration (mushroom + flower)
- [x] Sprint 3 — Data pipeline (Dataset, transforms, DataLoader)
- [x] Sprint 4 — Model research + Model Zoo (ResNet-50)
- [x] Sprint 5 — Training engine (checkpointing, LR scheduling, early stopping)
- [ ] Sprint 6 — Benchmark: EfficientNet-B3, ConvNeXt-T, augmentation ablation
- [ ] Sprint 7 — Explainability (Grad-CAM, expanded)
- [ ] Sprint 8 — Error analysis
- [ ] Sprint 9 — Inference pipeline
- [ ] Sprint 10 — Deployment (FastAPI, Docker, ONNX)

**Beyond V1:** disease/condition classification (PlantVillage), severity localization via segmentation, and — most importantly — proving the platform is genuinely domain-agnostic by plugging in a dataset from an unrelated field (e.g. medical imaging or industrial defect detection) without changing the core pipeline.

## Author

Built by [Sümeyye Saray](https://github.com/sumeyyesaray) as an applied deep learning / MLOps portfolio project, combining a statistics background with hands-on computer vision engineering.
