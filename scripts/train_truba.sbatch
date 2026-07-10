#!/bin/bash
# SLURM batch script for TRUBA (ARF cluster) — GPU training entry point,
# the TRUBA equivalent of notebooks/colab_train.ipynb's "6. Training" section.
#
# Submit with:
#   sbatch scripts/train_truba.sbatch configs/mushroom.yaml 15
#   sbatch scripts/train_truba.sbatch configs/flower.yaml 15
#
# To resume an interrupted run (walltime hit, job killed, etc.), add a third
# argument with the checkpoint path:
#   sbatch scripts/train_truba.sbatch configs/mushroom.yaml 15 outputs/checkpoints/mushroom_resnet50.pt

#SBATCH --job-name=visionlab-train
#SBATCH --partition=akya-cuda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch train_truba.sbatch <config.yaml> <epochs> [resume_checkpoint]}"
EPOCHS="${2:?usage: sbatch train_truba.sbatch <config.yaml> <epochs> [resume_checkpoint]}"
RESUME_CKPT="${3:-}"

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

python scripts/train_baseline.py --config "$CONFIG" --epochs "$EPOCHS" "${RESUME_ARGS[@]}"
