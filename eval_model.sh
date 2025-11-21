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
#   Baseline (no LoRA):    sbatch eval_model.sh --no-lora
#   Finetuned (with LoRA): sbatch eval_model.sh checkpoints/qwen_lora_seed42/final
#
# Optional environment variables for filtering:
#   FILTER_DURATION=1000 FILTER_SNR=20 sbatch eval_model.sh --no-lora
#   TEST_CSV=data/processed/experimental_variants/test_metadata.csv sbatch eval_model.sh --no-lora

set -euo pipefail
set -x

REPO="/mnt/fast/nobackup/users/gb0048/opro"
CONTAINER="$REPO/qwen_pipeline_v2.sif"

# Configurable via environment variables
TEST_CSV="${TEST_CSV:-data/processed/experimental_variants/dev_metadata.csv}"
PROMPT_FILE="${PROMPT_FILE:-prompts/prompt_base.txt}"
FILTER_DURATION="${FILTER_DURATION:-}"
FILTER_SNR="${FILTER_SNR:-}"

echo "[INFO] Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$REPO"

export HF_HOME="/mnt/fast/nobackup/users/gb0048/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" results

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Build filter arguments
FILTER_ARGS=""
if [ -n "$FILTER_DURATION" ]; then
    FILTER_ARGS="$FILTER_ARGS --filter_duration $FILTER_DURATION"
fi
if [ -n "$FILTER_SNR" ]; then
    FILTER_ARGS="$FILTER_ARGS --filter_snr $FILTER_SNR"
fi

# Parse model argument
if [ "${1:-}" == "--no-lora" ]; then
    echo "[RUN] Evaluating BASE MODEL (no LoRA)"
    OUTPUT_CSV="results/eval_baseline.csv"
    MODEL_ARGS="--no-lora"
else
    CHECKPOINT="${1:-checkpoints/qwen_lora_seed42/final}"
    echo "[RUN] Evaluating FINETUNED MODEL: $CHECKPOINT"
    OUTPUT_CSV="results/eval_finetuned_seed42.csv"
    MODEL_ARGS="--checkpoint $CHECKPOINT"
fi

echo "[CONFIG] Test CSV: $TEST_CSV"
echo "[CONFIG] Prompt file: $PROMPT_FILE"
echo "[CONFIG] Filters: $FILTER_ARGS"

apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "$CONTAINER" python3 scripts/evaluate_with_logits.py \
  $MODEL_ARGS \
  --test_csv "$TEST_CSV" \
  --prompt_file "$PROMPT_FILE" \
  --output_csv "$OUTPUT_CSV" \
  $FILTER_ARGS

echo "[DONE] End: $(date)"
