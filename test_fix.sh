#!/bin/bash
#SBATCH --job-name=test_fix
#SBATCH --partition=2080ti
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/test_fix_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/test_fix_%j.err

set -euo pipefail
set -x

REPO="/mnt/fast/nobackup/users/gb0048/opro"
CONTAINER="$REPO/qwen_pipeline_v2.sif"

echo "[INFO] Start: $(date)"
nvidia-smi
cd "$REPO"

export HF_HOME="/mnt/fast/nobackup/users/gb0048/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

echo "[RUN] Re-evaluation with SPEECH/NONSPEECH tokens (fixed prompt)"
apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  "$CONTAINER" python3 scripts/evaluate_with_logits.py \
  --checkpoint checkpoints/qwen_lora_seed42/final \
  --test_csv data/processed/experimental_variants/test_metadata.csv \
  --output_csv results/test_final/test_seed42_fixed.csv \
  --batch_size 8

echo "[DONE] End: $(date)"
