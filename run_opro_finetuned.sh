#!/bin/bash
#SBATCH --job-name=opro_ft
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/opro_ft_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/opro_ft_%j.err

# OPRO on FINE-TUNED model

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
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CHECKPOINT="${1:-checkpoints/qwen_lora_seed42/final}"

echo "[RUN] OPRO on FINE-TUNED model: $CHECKPOINT"

apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "$CONTAINER" python3 scripts/opro_post_ft.py \
  --checkpoint "$CHECKPOINT" \
  --train_csv data/processed/experimental_variants/dev_metadata.csv \
  --output_dir results/opro_finetuned \
  --num_iterations 15 \
  --samples_per_iter 20 \
  --num_candidates 6 \
  --seed 42

echo "[DONE] End: $(date)"
echo "Results: results/opro_finetuned/"
