#!/bin/bash
# SLURM batch script for TRUBA (ARF cluster) — GPU training entry point.
# The project's actual production runs are CPU-only on `orfoz`
# (see scripts/train_truba_cpu.sbatch and its job history under logs/
# visionlab-train-cpu_*) — this script is the GPU alternative, for when a
# GPU partition is free and worth using instead. Same env, same
# scripts/train_baseline.py, just a different module/partition.
#
# Submit with:
#   sbatch scripts/train_truba.sbatch configs/mushroom.yaml 15
#   sbatch scripts/train_truba.sbatch configs/flower.yaml 15
#
# To resume an interrupted run (walltime hit, job killed, etc.), add a third
# argument with the checkpoint path (pass "" if not resuming). A fourth
# argument caps training to a stratified subset (N images per class):
#   sbatch scripts/train_truba.sbatch configs/mushroom.yaml 15 outputs/checkpoints/mushroom_resnet50.pt
#   sbatch scripts/train_truba.sbatch configs/mushroom.yaml 10 "" 300

#SBATCH --job-name=visionlab-train-gpu
#SBATCH --partition=akya-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch train_truba.sbatch <config.yaml> <epochs> [resume_checkpoint] [subset_per_class]}"
EPOCHS="${2:?usage: sbatch train_truba.sbatch <config.yaml> <epochs> [resume_checkpoint] [subset_per_class]}"
RESUME_CKPT="${3:-}"
SUBSET_PER_CLASS="${4:-}"

mkdir -p logs

module purge
module load apps/truba-ai/gpu-2024.0
source activate visionlab

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
nvidia-smi

RESUME_ARGS=()
if [[ -n "$RESUME_CKPT" ]]; then
    RESUME_ARGS=(--resume "$RESUME_CKPT")
fi

SUBSET_ARGS=()
if [[ -n "$SUBSET_PER_CLASS" ]]; then
    SUBSET_ARGS=(--subset-per-class "$SUBSET_PER_CLASS")
fi

python scripts/train_baseline.py --config "$CONFIG" --epochs "$EPOCHS" "${RESUME_ARGS[@]}" "${SUBSET_ARGS[@]}"
