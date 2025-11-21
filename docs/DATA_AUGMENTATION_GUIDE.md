# Data Augmentation Guide for NONSPEECH

This guide describes the complete process to augment the NONSPEECH dataset with music and silence/noise clips to improve model balance.

## Problem

Current model shows:
- **SPEECH accuracy:** 83.85% ✓
- **NONSPEECH accuracy:** 63.72% ✗

NONSPEECH samples come only from ESC-50 (environmental sounds). Missing:
- Music
- Silence/low noise

## Solution

Augment NONSPEECH with:
- 40% ESC-50 (existing environmental sounds)
- 40% Music (various genres)
- 20% Silence/noise (different noise types and levels)

## Step-by-Step Process

### Step 1: Generate Silence/Noise Clips

```bash
python scripts/generate_silence_noise.py \
    --output_dir data/raw/silence_noise \
    --n_clips 500 \
    --duration_sec 2.0
```

This creates 500 clips with:
- White noise at various levels (-60dB to -20dB)
- Pink noise
- Brown noise
- Near-silence

**Output:** `data/raw/silence_noise/*.wav`

### Step 2: Download and Process Music

#### 2a. Download FMA metadata

```bash
python scripts/download_process_music.py --step download_metadata
```

#### 2b. Download FMA-small audio

Manual download (8GB):
```bash
wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_small.zip -d data/raw/
```

#### 2c. Extract music segments

```bash
python scripts/download_process_music.py \
    --step process \
    --fma_dir data/raw/fma_small \
    --output_dir data/raw/music \
    --n_clips 800
```

This extracts 800 × 2-second segments from instrumental music.

**Output:** `data/raw/music/*.wav`

### Step 3: Create Augmented Dataset

```bash
python scripts/create_augmented_dataset.py \
    --original_train data/processed/experimental_variants/train_metadata.csv \
    --music_dir data/raw/music \
    --noise_dir data/raw/silence_noise \
    --output_dir data/processed/augmented_dataset
```

This combines:
- All SPEECH samples (unchanged)
- Balanced NONSPEECH: 40% ESC-50, 40% music, 20% noise

**Output:**
- `data/processed/augmented_dataset/train_metadata_augmented.csv`
- `data/processed/augmented_dataset/test_metadata.csv` (copied, unchanged)
- `data/processed/augmented_dataset/dev_metadata.csv` (copied, unchanged)

### Step 4: Re-train Model

```bash
python scripts/finetune_qwen_audio.py \
    --train_csv data/processed/augmented_dataset/train_metadata_augmented.csv \
    --dev_csv data/processed/augmented_dataset/dev_metadata.csv \
    --output_dir checkpoints/qwen_augmented_seed42 \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 1e-4
```

**Output:** `checkpoints/qwen_augmented_seed42/`

### Step 5: Evaluate

```bash
# Baseline (for comparison)
sbatch eval_model.sh --no-lora

# Augmented model
sbatch eval_model.sh checkpoints/qwen_augmented_seed42/final
```

Compare results to see if NONSPEECH accuracy improved while maintaining SPEECH accuracy.

## Expected Outcomes

**Before augmentation:**
- SPEECH: 83.85%
- NONSPEECH: 63.72%
- Imbalance: +20.13% towards SPEECH

**After augmentation (target):**
- SPEECH: ~80-85% (maintain)
- NONSPEECH: ~75-80% (improve)
- Better balance

## Quick Start (All-in-One)

Run everything in sequence:

```bash
# 1. Generate noise
python scripts/generate_silence_noise.py --n_clips 500

# 2. Download music metadata
python scripts/download_process_music.py --step download_metadata

# 3. Download FMA-small (manual - 8GB)
# wget https://os.unil.cloud.switch.ch/fma/fma_small.zip && unzip fma_small.zip -d data/raw/

# 4. Process music
python scripts/download_process_music.py --step process --fma_dir data/raw/fma_small --n_clips 800

# 5. Create augmented dataset
python scripts/create_augmented_dataset.py \
    --original_train data/processed/experimental_variants/train_metadata.csv \
    --output_dir data/processed/augmented_dataset

# 6. Re-train
python scripts/finetune_qwen_audio.py \
    --train_csv data/processed/augmented_dataset/train_metadata_augmented.csv \
    --output_dir checkpoints/qwen_augmented_seed42

# 7. Evaluate
sbatch eval_model.sh checkpoints/qwen_augmented_seed42/final
```

## Troubleshooting

### Not enough music samples

If you get < 800 music clips:
- Check FMA-small was extracted correctly
- Try increasing `--n_clips` to compensate

### Memory issues during training

- Reduce `--batch_size` to 2 or 1
- Use gradient accumulation

### Model still imbalanced

- Adjust NONSPEECH distribution (e.g., 50% music, 30% ESC-50, 20% noise)
- Add more diverse music genres
- Try class weights in training

## Files Created

```
scripts/
├── generate_silence_noise.py       # Generate noise clips
├── download_process_music.py        # Download/process music
└── create_augmented_dataset.py      # Combine everything

data/
├── raw/
│   ├── silence_noise/               # Generated noise
│   ├── music/                       # Extracted music
│   └── fma_small/                   # Downloaded FMA
└── processed/
    └── augmented_dataset/           # Final augmented dataset
        ├── train_metadata_augmented.csv
        ├── test_metadata.csv
        └── dev_metadata.csv
```
