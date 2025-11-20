#!/bin/bash
#SBATCH --job-name=train_qwen42
#SBATCH --partition=2080ti
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/train_seed42_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/train_seed42_%j.err

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

echo "[RUN] Training Qwen2-Audio with LoRA (seed 42)"
apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  "$CONTAINER" python3 scripts/finetune_qwen_audio.py \
  --train_csv data/processed/experimental_variants/train_metadata.csv \
  --val_csv data/processed/experimental_variants/dev_metadata.csv \
  --output_dir checkpoints/qwen_lora_seed42 \
  --seed 42 \
  --num_epochs 3 \
  --learning_rate 5e-5 \
  --batch_size 1 \
  --gradient_accumulation_steps 8

echo "[DONE] End: $(date)"
