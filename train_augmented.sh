#!/bin/bash
#SBATCH --job-name=train_aug
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/train_aug_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/train_aug_%j.err

# Train model with augmented dataset

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

TRAIN_CSV="data/processed/augmented_dataset/train_metadata_augmented.csv"
DEV_CSV="data/processed/augmented_dataset/dev_metadata.csv"
OUTPUT_DIR="checkpoints/qwen_augmented_seed42"

echo "[RUN] Training with augmented data..."
echo "  Train: $TRAIN_CSV"
echo "  Dev: $DEV_CSV"
echo "  Output: $OUTPUT_DIR"

apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "$CONTAINER" python3 scripts/finetune_qwen_audio.py \
  --train_csv "$TRAIN_CSV" \
  --dev_csv "$DEV_CSV" \
  --output_dir "$OUTPUT_DIR" \
  --num_epochs 3 \
  --batch_size 4 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 2

echo "[DONE] End: $(date)"
echo ""
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Evaluate model:"
echo "  sbatch eval_model.sh $OUTPUT_DIR/final"
