#!/bin/bash
#SBATCH --job-name=debug_model
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/debug_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/debug_%j.err

set -euo pipefail
set -x

REPO="/mnt/fast/nobackup/users/gb0048/opro"
CONTAINER="$REPO/qwen_pipeline_v2.sif"

cd "$REPO"

export HF_HOME="/mnt/fast/nobackup/users/gb0048/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"

apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  "$CONTAINER" python3 scripts/debug_model_output.py

echo "[DONE]"
