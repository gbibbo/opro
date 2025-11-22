#!/bin/bash
#SBATCH --job-name=opro_mc
#SBATCH --partition=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/opro_mc_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/opro_mc_%j.err

# OPRO on BASE model with Multiple Choice (A/B/C/D) format

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

echo "[RUN] OPRO Multiple Choice format (A/B/C/D constrained)"

apptainer exec --nv \
  --env HF_HOME="$HF_HOME" \
  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  --env HF_HUB_CACHE="$HF_HUB_CACHE" \
  --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  "$CONTAINER" python3 scripts/opro_post_ft_v2.py \
  --no_lora \
  --train_csv data/processed/experimental_variants/dev_metadata.csv \
  --templates_file prompts/mc.json \
  --decoding mc \
  --samples_per_iter 100 \
  --num_iterations 10 \
  --num_candidates 12 \
  --output_dir results/opro_mc \
  --seed 42

echo "[DONE] End: $(date)"
echo "Results: results/opro_mc/"
