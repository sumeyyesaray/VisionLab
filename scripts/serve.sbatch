#!/bin/bash
# Runs the VisionLab prediction API as a long-lived SLURM job. Only reachable
# from inside the TRUBA cluster network (no public exposure) — for actual
# external access you'd need a reverse proxy / SSH tunnel to whatever node
# this lands on, which `squeue` will show after submission.
#
# Submit with:
#   sbatch scripts/serve.sbatch
#
# Not auto-started by anything — deliberately a manual step. Pick a walltime
# that matches how long you actually want the service up; orfoz caps at 3
# days, so a long-lived deployment needs periodic resubmission or a
# different hosting approach.

#SBATCH --job-name=visionlab-serve
#SBATCH --partition=orfoz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

mkdir -p logs

module purge
module load apps/truba-ai/cpu-2024.0
source activate visionlab

cd "$SLURM_SUBMIT_DIR"

echo "Serving on node: $(hostname)"
uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000
