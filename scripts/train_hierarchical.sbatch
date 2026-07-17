#!/bin/bash
# SLURM batch script for the hierarchical (genus+species) mushroom model.
# Mirrors scripts/train_truba_cpu.sbatch but calls scripts/train_hierarchical.py.
#
# Submit with:
#   sbatch scripts/train_hierarchical.sbatch configs/mushroom_hierarchical.yaml 30 "" 300

#SBATCH --job-name=visionlab-train-hier
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch train_hierarchical.sbatch <config.yaml> <epochs> [resume_checkpoint] [subset_per_class]}"
EPOCHS="${2:?usage: sbatch train_hierarchical.sbatch <config.yaml> <epochs> [resume_checkpoint] [subset_per_class]}"
RESUME_CKPT="${3:-}"
SUBSET_PER_CLASS="${4:-}"

mkdir -p logs

module purge
module load apps/truba-ai/cpu-2024.0
source activate visionlab

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

RESUME_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
    RESUME_ARGS=(--resume "$RESUME_CKPT")
fi

SUBSET_ARGS=()
if [[ -n "$SUBSET_PER_CLASS" ]]; then
    SUBSET_ARGS=(--subset-per-class "$SUBSET_PER_CLASS")
fi

python scripts/train_hierarchical.py --config "$CONFIG" --epochs "$EPOCHS" "${RESUME_ARGS[@]}" "${SUBSET_ARGS[@]}"
