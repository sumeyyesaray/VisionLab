#!/bin/bash
# Closes the flywheel loop for one dataset: exports verified predictions
# accumulated via POST /feedback, versions that snapshot with DVC, retrains
# on the original data + the snapshot, and only promotes the result to
# 'production' if scripts/promote_if_better.py says it actually beats the
# current one. Triggered by scripts/check_flywheel_threshold.py once enough
# verified data has piled up — not run on a fixed schedule by anything here.
#
# Submit with:
#   sbatch scripts/flywheel_retrain.sbatch mushroom
#   sbatch scripts/flywheel_retrain.sbatch flower

#SBATCH --job-name=visionlab-flywheel-retrain
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# --export=NONE: this job is submitted by check_flywheel_threshold.py, which
# check_flywheel_threshold.py's own crontab entry runs in cron's minimal
# environment (no `module`/`conda activate` set up — see docs/truba_usage.md
# section 7). sbatch's default --export=ALL propagates that stripped
# environment straight into this script, so `module purge` below failed with
# "module: command not found" the first two times cron triggered this (jobs
# 6182692, 6182705) — --export=NONE makes this job build its own environment
# from scratch instead of inheriting whatever submitted it.
#SBATCH --export=NONE

set -euo pipefail

DATASET_TYPE="${1:?usage: sbatch flywheel_retrain.sbatch <mushroom|flower>}"

mkdir -p logs
module purge
module load apps/truba-ai/cpu-2024.0
source activate visionlab
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

case "$DATASET_TYPE" in
    mushroom)
        BASE_CONFIG=configs/mushroom.yaml
        LABEL_MAP=configs/label_maps/mushroom_label_map.json
        ;;
    flower)
        BASE_CONFIG=configs/flower.yaml
        LABEL_MAP=configs/label_maps/flower_label_map.json
        ;;
    *)
        echo "unknown dataset_type: $DATASET_TYPE" >&2
        exit 1
        ;;
esac

SNAPSHOT_DIR="datasets/extracted/${DATASET_TYPE}_flywheel"
REGISTERED_NAME="${DATASET_TYPE}_resnet50"
CHECKPOINT_NAME="${DATASET_TYPE}_resnet50_flywheel_${SLURM_JOB_ID}"

echo "=== Exporting verified flywheel predictions ==="
python -c "
from src.flywheel.export import export_flywheel_snapshot
path = export_flywheel_snapshot('${DATASET_TYPE}', '${LABEL_MAP}', '${SNAPSHOT_DIR}')
import sys
sys.exit(0 if path else 1)
"

echo "=== Versioning the snapshot with DVC ==="
dvc add "$SNAPSHOT_DIR"

echo "=== Retraining on original + flywheel data ==="
python scripts/train_baseline.py \
    --config "$BASE_CONFIG" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --extra-train-csv "$SNAPSHOT_DIR/train.csv"

echo "=== Promotion gate ==="
python scripts/promote_if_better.py \
    --name "$REGISTERED_NAME" \
    --candidate-checkpoint "outputs/checkpoints/${CHECKPOINT_NAME}_best.pt"
