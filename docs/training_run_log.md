# Training Run Log

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

## Run 2 — in progress

- Mushroom: same 300/class subset, 6 epochs (job 6066357) — run again so results are comparable
  to Run 1 with the new metrics attached
- Flower: full dataset (6,552 train / 818 val), 6 epochs (job 6066358) — first real run with
  macro-F1 (mandatory for this dataset per the current requirements)
- New artifacts per run: `outputs/reports/{dataset}_{model}_eval_report.json`,
  MLflow `run_duration_seconds` + `evaluation_report.json` artifact
