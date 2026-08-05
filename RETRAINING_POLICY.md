# Flywheel Retraining Policy

Three decisions that govern the continuous-learning loop (`src/flywheel/`,
`scripts/flywheel_retrain.sbatch`, `scripts/check_flywheel_threshold.py`).
Written down here so "why is this threshold 500 and not 100" has an answer
six months from now instead of a guess.

## Decision 1 — What goes on the human review queue

A prediction is not a label. Nothing a model predicts gets used for
retraining until a human confirms or corrects it via `POST /feedback`. But
not every prediction is worth a human's time to check:

> Only predictions with **confidence < 0.70** are queued for review
> (`GET /flywheel/pending`). A prediction the model made at 99% confidence
> never enters that queue.

**Why:** re-confirming something the model already got right with high
confidence wastes reviewer time and doesn't teach the model anything it
doesn't already know. The predictions worth a human's attention are
exactly the ones the model was unsure about — that's also where a
correction is most valuable as training data (hard examples).

**Mechanics:** `src/flywheel/store.py`'s `log_prediction()` computes
`needs_verification = confidence < CONFIDENCE_REVIEW_THRESHOLD` (0.70) at
insert time and stores it as a column. `GET /flywheel/pending` filters on
it. `POST /feedback` itself is *not* restricted to queued predictions — a
reviewer can still correct a high-confidence prediction they happen to
notice was wrong, they just won't be prompted to look at it.

**Where to change it:** `CONFIDENCE_REVIEW_THRESHOLD` in
`src/flywheel/store.py`.

## Decision 2 — When to trigger a retraining run

> Retrain when **verified_count >= 500 OR days_since_last_check >= 30**.

**Why:** the count condition catches a burst of new verified data (worth
retraining on right away). The day condition is a floor, not a target — it
guarantees the model is revisited at least monthly even if verified data
trickles in slowly, so staleness doesn't depend entirely on usage volume.
Triggering with little or no new data is harmless: `export.py` no-ops if
there's nothing to export, and `promote_if_better.py` won't promote a
result that isn't actually better.

**Mechanics:** `scripts/check_flywheel_threshold.py --dataset-type
<mushroom|flower>` checks both conditions (`--count-threshold 500`,
`--days-threshold 30` are the defaults, both overridable) and submits
`scripts/flywheel_retrain.sbatch` if either is true.
`src/flywheel/store.record_retrain_trigger()` timestamps every actual
submission, which is what the day condition measures against — never
having triggered before counts as infinitely overdue, not as "skip."

**Automated via crontab** (`crontab -l` on `arf-ui1`, user `hasozkan`) — runs
daily, logs to `logs/flywheel_cron.log`. Uses the `visionlab` conda env's
python by its absolute path rather than `module load`/`conda activate`,
since cron's minimal environment doesn't source the shell init those rely
on:

```
PATH=/usr/bin:/bin:/usr/local/bin

0 3 * * * cd /arf/scratch/hasozkan/sum/VisionLab && /arf/home/hasozkan/miniforge3/envs/visionlab/bin/python scripts/check_flywheel_threshold.py --dataset-type mushroom >> logs/flywheel_cron.log 2>&1
15 3 * * * cd /arf/scratch/hasozkan/sum/VisionLab && /arf/home/hasozkan/miniforge3/envs/visionlab/bin/python scripts/check_flywheel_threshold.py --dataset-type flower >> logs/flywheel_cron.log 2>&1
```

See `docs/truba_usage.md` for how to inspect/edit this.

**Where to change it:** `--count-threshold` / `--days-threshold` flags.

## Decision 3 — When a retrained model actually replaces production

> A retraining run finishing is never by itself a reason to promote.
> `scripts/flywheel_retrain.sbatch` never calls `register_model.py
> --production`. Its last step is always `scripts/promote_if_better.py`,
> which registers the candidate but only moves the `production` alias if
> the candidate's macro-F1 beats whatever is currently aliased
> `production` (fetched from that model version's MLflow run metrics).
> No current production model at all counts as "beats it."

**Why:** this is the actual safety net for the whole flywheel loop. A bad
verification batch, a labeling mistake, or a retraining run that just
landed on a worse local optimum must not be able to silently degrade the
live service. The gate is unconditional — there's no override flag, on
purpose.

**Mechanics:** see `scripts/promote_if_better.py`. It's also usable
standalone, outside the flywheel loop, for any candidate checkpoint.
