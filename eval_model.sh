#!/bin/bash
#SBATCH --job-name=eval_qwen
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/eval_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/eval_%j.err

# Usage:
#   Baseline (no LoRA):   sbatch eval_model.sh --no-lora
#   Finetuned (with LoRA): sbatch eval_model.sh checkpoints/qwen_lora_seed42/final

set -euo pipefail
set -x

REPO="/mnt/fast/nobackup/users/gb0048/opro"
CONTAINER="$REPO/qwen_pipeline_v2.sif"
# Use dev set for evaluation (same as validation during training)
# Change to test_metadata.csv if you have a separate test set
TEST_CSV="data/processed/experimental_variants/dev_metadata.csv"

echo "[INFO] Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$REPO"

export HF_HOME="/mnt/fast/nobackup/users/gb0048/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Parse arguments
if [ "${1:-}" == "--no-lora" ]; then
    echo "[RUN] Evaluating BASE MODEL (no LoRA)"
    OUTPUT_CSV="results/eval_baseline.csv"
    mkdir -p results
    apptainer exec --nv \
      --env HF_HOME="$HF_HOME" \
      --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
      --env HF_HUB_CACHE="$HF_HUB_CACHE" \
      --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
      "$CONTAINER" python3 scripts/evaluate_with_logits.py \
      --no-lora \
      --test_csv "$TEST_CSV" \
      --output_csv "$OUTPUT_CSV"
else
    CHECKPOINT="${1:-checkpoints/qwen_lora_seed42/final}"
    echo "[RUN] Evaluating FINETUNED MODEL: $CHECKPOINT"
    OUTPUT_CSV="results/eval_finetuned_seed42.csv"
    mkdir -p results
    apptainer exec --nv \
      --env HF_HOME="$HF_HOME" \
      --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
      --env HF_HUB_CACHE="$HF_HUB_CACHE" \
      --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
      "$CONTAINER" python3 scripts/evaluate_with_logits.py \
      --checkpoint "$CHECKPOINT" \
      --test_csv "$TEST_CSV" \
      --output_csv "$OUTPUT_CSV"
fi

echo "[DONE] End: $(date)"
