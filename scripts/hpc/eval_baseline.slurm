#!/usr/bin/env bash
#SBATCH --job-name=t2c-eval
#SBATCH --partition=c23g            # GPU partition (confirm with sinfo)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out     # logs/<jobname>-<jobid>.out
#SBATCH --error=logs/%x-%j.out
# #SBATCH --account=YOUR_ACCOUNT     # uncomment + set, or pass with: sbatch --account=...

# First real HPC job: run the FULL zero-shot Grounding DINO eval on a GPU (~minutes).
# This both (a) validates the cluster/Apptainer flow on a cheap no-training job and
# (b) produces the official zero-shot baseline number.
#
# Submit from the repo root on the cluster:
#   mkdir -p logs
#   sbatch scripts/hpc/eval_baseline.slurm

set -euo pipefail

module purge
module load Apptainer 2>/dev/null || module load apptainer

CONTAINER=${CONTAINER:-drive-vlm.sif}   # built from scripts/hpc/Apptainer.def

echo "Host: $(hostname)  GPUs:"; nvidia-smi --query-gpu=name --format=csv,noheader || true

# Run the eval inside the container, with GPU access (--nv), repo mounted at /workspace.
apptainer exec --nv \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  "$CONTAINER" \
  python scripts/eval_baseline.py --split val   # full split (no --limit)

echo "Done."
