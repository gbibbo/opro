#!/bin/bash
#SBATCH --job-name=train_qwen42
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_3090|gpu_a5000|gpu_5000_ada"
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/train_seed42_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/train_seed42_%j.err

set -euo pipefail
set -x

REPO="/mnt/fast/nobackup/users/gb0048/opro"
CONTAINER="$REPO/qwen_pipeline_v2.sif"

echo "[INFO] Start: $(date)"
echo "[INFO] Allocated GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$REPO"

export HF_HOME="/mnt/fast/nobackup/users/gb0048/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

# Fix CUDA OOM: Enable expandable memory segments to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[RUN] Training Qwen2-Audio with LoRA (seed 42)"
apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "$CONTAINER" python3 scripts/finetune_qwen_audio.py \
  --train_csv data/processed/experimental_variants/train_metadata.csv \
  --val_csv data/processed/experimental_variants/dev_metadata.csv \
  --output_dir checkpoints/qwen_lora_seed42 \
  --seed 42 \
  --num_epochs 3 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4

echo "[DONE] End: $(date)"
