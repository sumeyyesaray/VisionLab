#!/bin/bash
# SLURM batch script for TRUBA (ARF cluster) — CPU-only training entry point.
# No GPU partition, no --gres — everything runs on CPU cores. train_baseline.py
# already falls back to CPU automatically (torch.cuda.is_available() is False)
# and disables mixed precision in that case, so no code changes were needed.
#
# Submit with:
#   sbatch scripts/train_truba_cpu.sbatch configs/mushroom.yaml 15
#   sbatch scripts/train_truba_cpu.sbatch configs/flower.yaml 15
#
# To resume an interrupted run (walltime hit, job killed, etc.), add a third
# argument with the checkpoint path:
#   sbatch scripts/train_truba_cpu.sbatch configs/mushroom.yaml 15 outputs/checkpoints/mushroom_resnet50.pt
#
# To cap training to a stratified subset (N images per class), add a fourth
# argument (pass "" for the checkpoint slot if not resuming):
#   sbatch scripts/train_truba_cpu.sbatch configs/mushroom.yaml 10 "" 300

#SBATCH --job-name=visionlab-train-cpu
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --exclusive # avoid node-sharing contention (Run 4: another job's 5 cores on
                     # the same node ~doubled our epoch time) — reserve the whole node
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch train_truba_cpu.sbatch <config.yaml> <epochs> [resume_checkpoint] [subset_per_class]}"
EPOCHS="${2:?usage: sbatch train_truba_cpu.sbatch <config.yaml> <epochs> [resume_checkpoint] [subset_per_class]}"
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

python scripts/train_baseline.py --config "$CONFIG" --epochs "$EPOCHS" "${RESUME_ARGS[@]}" "${SUBSET_ARGS[@]}"
