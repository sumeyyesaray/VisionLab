# MLOps hardening pass — 2026-08-05

Session notes for a TRUBA/MLOps review pass on an already-built pipeline (DVC +
MLflow registry + W&B + FastAPI serving + flywheel continuous retraining, see
`README.md` and `RETRAINING_POLICY.md`). Written down because the main finding
— the flywheel loop had never actually completed a run — wasn't visible from
reading the code; it only showed up by checking `flywheel.db` and real job
logs against what the code claims to do.

## What was already working

- DVC-versioned full datasets, MLflow Model Registry with `production` aliases
  set for both `mushroom_resnet50` (v4) and `flower_resnet50` (v3), W&B
  experiment tracking, FastAPI serving (`src/deployment/api.py`) all
  functioning as designed.
- `scripts/promote_if_better.py`'s promotion gate logic is correct (confirmed
  by the tests added below).

## What was found broken

1. **TRUBA docs/scripts had drifted from reality.** `scripts/train_truba.sbatch`
   and `docs/truba_usage.md` (written in an earlier session) assumed
   MLflow/TensorBoard and referenced `notebooks/colab_train.ipynb`, which had
   already been removed. They also presented the unused GPU path as primary
   when the project actually trains CPU-only on `orfoz`
   (`scripts/train_truba_cpu.sbatch`).

2. **The flywheel loop has never completed end-to-end.** `flywheel.db` has 19
   verified mushroom predictions, all with `used_in_training = 0`. Every real
   attempt at `scripts/flywheel_retrain.sbatch` failed before reaching the
   export step:
   - Job `6158408` (30 Jul, submitted manually): `source activate visionlab`
     failed — `activate: No such file or directory`.
   - Jobs `6182692` / `6182705` (4 Aug, mushroom/flower — the first two firings
     of the crontab added this session): `module purge` failed — `module:
     command not found`.

   No flywheel DVC snapshot, flywheel-tagged checkpoint, or flywheel eval
   report exists anywhere in the repo, confirming none of the three attempts
   ever reached the export/retrain/promote stages.

3. **Root cause of the `module: command not found` failures:**
   `check_flywheel_threshold.py` now runs daily via `crontab`, whose minimal
   environment (`PATH=/usr/bin:/bin:/usr/local/bin`, no module/conda init) is
   what actually calls `subprocess.run(["sbatch", ...])`. `sbatch`'s default
   `--export=ALL` propagates that stripped environment into the submitted
   job, so `flywheel_retrain.sbatch`'s own `module purge` / `module load`
   never had a working `module` function to begin with. The other sbatch
   scripts (`train_truba_cpu.sbatch`, `train_hierarchical.sbatch`, etc.) never
   hit this because they've only ever been submitted interactively, from a
   shell where `module` was already loaded.

4. **`tests/` was empty** (just `.gitkeep`) — no coverage on the promotion
   gate or flywheel export, the two pieces of logic a bad retrain would
   actually go through.

5. **DVC's remote (`../.dvc-remote-storage`) is on the same `/arf` Lustre
   filesystem** as the working copy — not an off-cluster backup. If TRUBA
   scratch has a purge policy, or `/arf` has an outage, both copies are at
   risk together. Not fixed (needs a TRUBA-retention-policy answer first,
   noted as a warning in `README.md`).

## What was fixed (commits `71b6ac9`, `b72b131`, `3ea8318` on `dev`, pushed)

- `scripts/train_truba.sbatch` / `docs/truba_usage.md`: rewritten to match
  reality (CPU/`orfoz` primary, GPU alternative, W&B tracking, stale
  Colab/MLflow references removed).
- `RETRAINING_POLICY.md`: decision 2's crontab example replaced with the one
  actually installed (absolute env python path, not bare `python`, which
  would never have resolved the project's dependencies from cron).
- **Crontab installed** on `arf-ui1` (`hasozkan`, `crontab -l` to view):
  daily 03:00 (mushroom) / 03:15 (flower) threshold checks, logging to
  `logs/flywheel_cron.log`.
- `tests/test_flywheel_export.py`, `tests/test_promote_if_better.py` (+
  `tests/conftest.py`): 6 smoke tests, all passing — export happy path +
  2 edge cases (unknown label / missing file, no verified records);
  `get_production_macro_f1` happy path + no-alias edge case; a `main()`-level
  test confirming a worse candidate is never promoted (decision 3's actual
  safety net). `pytest` added to `pyproject.toml`, which was missing it.
- `scripts/flywheel_retrain.sbatch`: added `#SBATCH --export=NONE` so the job
  builds its own environment instead of inheriting whatever submitted it —
  fixes the cron-triggered failure mode specifically.
- `README.md`: DVC remote / off-cluster-backup warning added.

## Verification status

- The doc/script consistency fixes and the 6 tests: verified (tests run for
  real in the `visionlab` conda env, all passed).
- The `--export=NONE` fix: **applied but not yet verified against a real run**
  — deliberately not triggered manually, since a real `flywheel_retrain.sbatch`
  run reserves a full `orfoz` node (`--exclusive`) for potentially hours and
  can change which model is aliased `production`. Natural verification point:
  the next crontab firing (03:00/03:15) — check `logs/flywheel_cron.log` and
  the new `logs/visionlab-flywheel-retrain_*.err` for whether `module purge`
  succeeds this time and the job proceeds past environment setup.

## Still open

- API (`src/deployment/api.py`) has no auth — acceptable today only because
  `scripts/serve.sbatch` is cluster-internal only.
- No CI (`.github/workflows` doesn't exist).
- DVC off-cluster backup question (see point 5 above) — unresolved.
