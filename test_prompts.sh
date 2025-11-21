#!/bin/bash
#SBATCH --job-name=test_prompts
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/test_prompts_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/test_prompts_%j.err

set -euo pipefail
set -x

REPO="/mnt/fast/nobackup/users/gb0048/opro"
CONTAINER="$REPO/qwen_pipeline_v2.sif"

echo "[INFO] Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$REPO"

export HF_HOME="/mnt/fast/nobackup/users/gb0048/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"

# Test prompts on 50 samples per class (100 total)
apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  "$CONTAINER" python3 scripts/test_prompts_quick.py \
  --test_csv data/processed/experimental_variants/dev_metadata.csv \
  --n_samples 50

echo "[DONE] End: $(date)"
