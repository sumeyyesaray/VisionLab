# Training Run Log

## Sprint 6 — First real training run (moved from notebooks/06_first_real_training_run.ipynb)

Raw results moved here to keep training-run logs out of the notebooks; the notebook keeps the
rationale (GPU discovery, MLflow/TensorBoard setup, AMP, checkpoint resume, config changes,
stratified subsetting) and the Findings/Next Steps analysis, which still refers to this data by
number.

### Results — Flower (completed run)

Run: `python scripts/train_baseline.py --config configs/flower.yaml --subset-per-class 25 --epochs 6`
(~25 images/class × 102 classes ≈ 2,550 training images, batch_size=16, AMP on)

Pulled directly from the MLflow/TensorBoard logs below — not retyped by hand.

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("visionlab")
runs = client.search_runs([experiment.experiment_id], order_by=["start_time ASC"])

# Filter out the tracking-fix verification run (params={"foo": "bar"}) from Sprint 6, section 2.
real_runs = [r for r in runs if "model" in r.data.params]

for run in real_runs:
    print(f"dataset={run.data.params['dataset']:<10} status={run.info.status:<10} "
          f"batch_size={run.data.params['batch_size']} lr={run.data.params['lr']} "
          f"augmentation={run.data.params['augmentation_preset']:<8} "
          f"class_weighted={run.data.params['class_weighted_loss']}")
    print(f"  final: {run.data.metrics}")
```

Epoch-by-epoch curve, read from the TensorBoard event file
(`runs/flower_resnet50/`):

| Epoch | train_loss | val_loss | train_acc | val_acc |
|---|---|---|---|---|
| 1 | 3.6136 | 2.3582 | 0.2298 | 0.4132 |
| 2 | 1.1184 | 0.8027 | 0.7035 | 0.7885 |
| 3 | 0.4548 | 0.4694 | 0.8765 | 0.8655 |
| 4 | 0.2383 | 0.3499 | 0.9314 | 0.9108 |
| 5 | 0.1676 | 0.2885 | 0.9596 | 0.9218 |
| 6 | 0.1097 | 0.2941 | 0.9737 | 0.9230 |

**Reading this:** both losses drop sharply in the first 2-3 epochs (expected
— the pretrained ImageNet backbone already "knows" useful features, only
the new classifier head + fine-tuning need to catch up) and validation
accuracy tracks training accuracy closely through epoch 5, which is a good
sign for a 6-epoch run on ~2,550 images. One thing worth flagging, not
fixing yet: `val_loss` ticks *up* slightly in the last epoch (0.2885 →
0.2941) while `train_loss` keeps dropping — the earliest, mildest possible
signal of the model starting to overfit the small subset. Not a concern at
6 epochs, but worth watching if this same setup runs for many more epochs
on the full flower dataset.

### Results — Mushroom (completed run)

Run: `python scripts/train_baseline.py --config configs/mushroom.yaml --subset-per-class 25 --epochs 6`
(~25 images/class × 169 classes ≈ 4,225 training images, `augmentation_preset: light`, no class weighting)

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

files = sorted(glob.glob("runs/mushroom_resnet50/events.out.tfevents.*"))
for f in files:
    ea = EventAccumulator(f)
    ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        values = [(e.step + 1, round(e.value, 4)) for e in ea.Scalars(tag)]
        print(f"{tag:<15} {values}")
```

| Epoch | train_loss | val_loss | train_acc | val_acc |
|---|---|---|---|---|
| 1 | 4.5234 | 2.7997 | 0.1112 | 0.3587 |
| 2 | 2.3008 | 1.8899 | 0.4566 | 0.5042 |
| 3 | 1.3081 | 1.5916 | 0.6764 | 0.5662 |
| 4 | 0.7884 | 1.5100 | 0.8161 | 0.5831 |
| 5 | 0.4536 | 1.4446 | 0.9015 | 0.6034 |
| 6 | 0.2815 | 1.4031 | 0.9472 | 0.6197 |

**Reading this — and comparing to flower:** the gap between train_acc
(94.7%) and val_acc (62.0%) by epoch 6 is much wider than flower's
(97.4% vs. 92.3%) — mushroom is visibly overfitting this subset, flower
isn't (yet). It's tempting to list "169 vs. 102 classes" and "`light` vs.
`heavy` augmentation" as two independent, equally-weighted causes next to
a third about data scarcity. That would be misleading — there's one
primary cause here, and the other two are downstream of it, not
independent of it.

**Primary cause: this run gives mushroom an artificially scarce slice of
itself.** 25 images/class is 0.61% of mushroom's real per-class count
(4,080) — for flower, the same 25/class is close to its *actual* average
(~64/class, 38.92% of the full train set; see notebook section 6's table).
Asking a 169-class problem to generalize from 25 examples/class is a
fundamentally harder few-shot-like task, independent of anything else in
the config.

**Why the other two factors are secondary, not independent, causes:**

- **169 vs. 102 classes at equal per-class count** compounds the scarcity
  effect (more classes to separate from the same tiny per-class sample) —
  it doesn't explain overfitting on its own; it's a multiplier on the
  scarcity problem above.
- **`augmentation_preset: light` vs. `heavy` is a config decision that was
  correct for a different scenario.** Sprint 1/3 set `light` for mushroom
  *because* the real dataset has 4,080 images/class — abundant data, so
  aggressive augmentation wasn't judged necessary, and there was also an
  open concern about possible near-duplicates making heavier augmentation
  risky. Neither justification holds in a 25-images/class subset: here
  there's no abundance to lean on, so `light` provides less regularization
  than this specific (artificial) scenario would benefit from. That's a
  mismatch between the subset and the assumption the config decision was
  based on — not evidence that the decision itself was wrong for the real,
  4,080-images/class dataset.

**The conclusion this section does *not* support:** "mushroom's config
needs to change." This overfitting is the expected result of solving a
169-class problem with an artificially scarce 25-images/class slice — not
a pipeline, model, or config defect. Changing `augmentation_preset` (or
anything else) based on this smoke test, before ever training on the real
4,080-images/class data, would be reacting to an artifact of the test
setup rather than a real signal.

## Run 1 — Mushroom baseline (job 6061165)

- Subset: 300 images/class (169 classes ≈ 50,700 train), full val (15,616)
- 6 epochs, CPU-only (`orfoz`, 56 cores)
- Metrics available: loss/accuracy only — no macro-F1, no per-class breakdown
- Result: train_acc 95.4%, best val_acc 80.2% at epoch 3, final val_acc 78.0% (epoch 6) —
  val_loss rises after epoch 3 while train_loss keeps falling, i.e. overfitting past that point
- Duration: 3h11m47s
- Checkpoint: `outputs/checkpoints/mushroom_resnet50.pt` — last-epoch weights, not the best-val
  epoch (checkpoint saving isn't best-epoch-aware yet)

## What changed before Run 2

- `src/evaluation/metrics.py` (new): macro-F1, per-class precision/recall/F1, top-10 most
  confused class pairs, worst-10 classes by F1
- `src/training/engine.py`: `evaluate()` now also returns `y_true`/`y_pred` so the above can be
  computed without a second pass over the validation set
- `src/training/tracking.py`: `log_epoch_metrics` accepts extra per-epoch metrics
  (`val_macro_f1`, `epoch_duration_seconds`); added `log_run_duration` and
  `log_evaluation_report` (MLflow)
- `scripts/train_baseline.py`: logs macro-F1 and epoch duration every epoch (MLflow +
  TensorBoard); after the last epoch, prints and saves a JSON report — worst-10 classes,
  top-10 confused pairs — to `outputs/reports/{dataset}_{model}_eval_report.json`, and logs
  total run duration to MLflow
- `pyproject.toml`: added `scikit-learn` as an explicit dependency (was only a transitive one
  via mlflow before)
- Validated with two smoke-test jobs (50 and 200 image subsets, with and without MLflow) before
  submitting real runs

## Run 2 — completed (mushroom job 6066357, flower job 6066358)

New artifacts per run: `outputs/reports/{dataset}_{model}_eval_report.json`, plus MLflow
`run_duration_seconds` and an `evaluation_report.json` artifact.

### Mushroom (300/class subset, same as Run 1, now with the new metrics)

- Duration: 13081.5s (3h38m), 56 CPU cores (`orfoz274`)
- train_acc 95.2%, val_acc 80.2%, **macro-F1 0.7596** (final epoch)
- Unlike Run 1, val_acc keeps improving through epoch 6 instead of peaking at epoch 3 — less
  overfitting this time (run-to-run variance from weight init / data shuffling, same subset size)
- Worst-10 classes are mostly lichens, not true mushrooms (Fomitopsis mounceae F1 0.385,
  Boletus reticulatus 0.395, Phaeophyscia orbicularis 0.451, …) — support 35-83 images each
- Top confused pair: Xanthoria parietina → Vulpicida pinastri (66 misclassifications) — both
  lichens, visually similar; several Fomitopsis/Fomes pairs also recur

### Flower (full dataset: 6,552 train / 818 val, first real run with macro-F1)

- Duration: 1451.0s (24m), 56 CPU cores (`orfoz286`)
- train_acc 97.8%, val_acc 97.8%, **macro-F1 0.9726** (final epoch) — strong result on the full
  dataset, consistent with Sprint 4's EfficientNet research note (up to 98.8% reported achievable)
- Worst-10 classes are almost entirely low-support ones (canterbury bells support=2, monkshood
  support=3, …) — F1 noise from small sample size, not a systematic weakness
- Top confused pairs are all single-count, isolated misclassifications — no dataset-wide
  confusion pattern like mushroom's lichen pairs
