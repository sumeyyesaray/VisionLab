#!/bin/bash
# SLURM batch script for Stage 2 of the two-stage fine-tuning experiment
# (see notebooks/08_mushroom_confusion_diagnosis.ipynb). CPU-only, same
# module/env setup as scripts/train_truba_cpu.sbatch, but calls
# scripts/train_stage2_finetune.py instead of train_baseline.py since the
# training loop differs (scarce-only train set, full-val-set eval, frozen
# backbone except layer4+fc).
#
# Submit with:
#   sbatch scripts/train_stage2.sbatch configs/mushroom_twostage_stage2.yaml
#
# Note: job 6114751 (submitted directly, no tracked sbatch script) sat
# PENDING ~2h13m then FAILED instantly with zero elapsed time and no log
# output at all — cause unknown since nothing was written. This script
# exists so any future failure actually produces logs/%x_%j.{out,err} to
# diagnose from.

#SBATCH --job-name=visionlab-stage2
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch train_stage2.sbatch <config.yaml>}"

mkdir -p logs

module purge
module load apps/truba-ai/cpu-2024.0
source activate visionlab

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
echo "CPUs: $(nproc)"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python scripts/train_stage2_finetune.py --config "$CONFIG"
