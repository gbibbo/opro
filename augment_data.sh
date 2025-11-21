#!/bin/bash
#SBATCH --job-name=augment_data
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/fast/nobackup/users/gb0048/opro/logs/augment_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/gb0048/opro/logs/augment_%j.err

# Data Augmentation Pipeline
# Generates music and noise clips, combines with existing data

set -euo pipefail
set -x

REPO="/mnt/fast/nobackup/users/gb0048/opro"
CONTAINER="$REPO/qwen_pipeline_v2.sif"

echo "[INFO] Start: $(date)"
cd "$REPO"

mkdir -p data/raw/music data/raw/silence_noise data/raw/fma_metadata

# Step 1: Generate silence/noise clips
echo "=" | tr '=' '-' | head -c 60; echo
echo "[STEP 1/5] Generating silence/noise clips..."
echo "=" | tr '=' '-' | head -c 60; echo

apptainer exec "$CONTAINER" python3 scripts/generate_silence_noise.py \
    --n_clips 500 \
    --output_dir data/raw/silence_noise \
    --duration_sec 2.0

echo "[DONE] Generated $(ls data/raw/silence_noise/*.wav 2>/dev/null | wc -l) noise clips"

# Step 2: Download FMA metadata
echo "=" | tr '=' '-' | head -c 60; echo
echo "[STEP 2/5] Downloading FMA metadata..."
echo "=" | tr '=' '-' | head -c 60; echo

apptainer exec "$CONTAINER" python3 scripts/download_process_music.py \
    --step download_metadata \
    --metadata_dir data/raw

# Step 3: Download FMA-small audio (if not exists)
echo "=" | tr '=' '-' | head -c 60; echo
echo "[STEP 3/5] Downloading FMA-small dataset..."
echo "=" | tr '=' '-' | head -c 60; echo

FMA_ZIP="data/raw/fma_small.zip"
FMA_DIR="data/raw/fma_small"

if [ ! -f "$FMA_ZIP" ] && [ ! -d "$FMA_DIR" ]; then
    echo "Downloading FMA-small (8GB)..."
    wget -O "$FMA_ZIP" https://os.unil.cloud.switch.ch/fma/fma_small.zip

    echo "Extracting..."
    unzip -q "$FMA_ZIP" -d data/raw/
    echo "[DONE] Extracted to $FMA_DIR"
else
    echo "[SKIP] FMA-small already exists"
fi

# Step 4: Process music clips
echo "=" | tr '=' '-' | head -c 60; echo
echo "[STEP 4/5] Processing music clips..."
echo "=" | tr '=' '-' | head -c 60; echo

apptainer exec "$CONTAINER" python3 scripts/download_process_music.py \
    --step process \
    --fma_dir "$FMA_DIR" \
    --output_dir data/raw/music \
    --n_clips 800

echo "[DONE] Generated $(ls data/raw/music/*.wav 2>/dev/null | wc -l) music clips"

# Step 5: Create augmented dataset
echo "=" | tr '=' '-' | head -c 60; echo
echo "[STEP 5/5] Creating augmented dataset..."
echo "=" | tr '=' '-' | head -c 60; echo

apptainer exec "$CONTAINER" python3 scripts/create_augmented_dataset.py \
    --original_train data/processed/experimental_variants/train_metadata.csv \
    --music_dir data/raw/music \
    --noise_dir data/raw/silence_noise \
    --output_dir data/processed/augmented_dataset

echo "=" | tr '=' '-' | head -c 60; echo
echo "[SUCCESS] Data augmentation complete!"
echo "=" | tr '=' '-' | head -c 60; echo
echo "Output: data/processed/augmented_dataset/train_metadata_augmented.csv"
echo ""
echo "Next step: Re-train model with augmented data:"
echo "  sbatch train_augmented.sh"
echo ""
echo "[INFO] End: $(date)"
