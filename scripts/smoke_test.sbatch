#!/bin/bash
# Fast smoke test on the `debug` partition (4h limit, quick queue) — sanity-
# checks the training script end to end (data loading, one epoch, W&B/MLflow
# logging, checkpoint save) on a tiny image subset before committing a full
# multi-day orfoz run. Not tracked ad hoc via --wrap: a previous stage-2 job
# (6114751) was submitted without a script, produced zero logs, and failed
# silently — see scripts/train_stage2.sbatch's header note.
#
# Submit with:
#   sbatch scripts/smoke_test.sbatch configs/mushroom.yaml

#SBATCH --job-name=visionlab-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch smoke_test.sbatch <config.yaml>}"

mkdir -p logs

module purge
module load apps/truba-ai/cpu-2024.0
source activate visionlab

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python scripts/train_baseline.py --config "$CONFIG" --epochs 1 --limit 32
