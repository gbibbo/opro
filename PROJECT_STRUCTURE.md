# OPRO Qwen - Project Structure

**Last Updated**: 2025-10-12
**Status**: Cleaned and organized

---

## Overview

This project evaluates the Qwen2-Audio model on psychoacoustic degradation conditions to measure robustness to:
- Duration reduction (20ms - 1000ms)
- SNR degradation (-10dB to +20dB)
- Band-pass filtering (telephony, low-pass, high-pass)
- Reverberation (T60: 0.0s - 1.5s)

**Current Performance**: 96.25% accuracy on 80 samples (2 clips × 2 classes × 20 variants)

---

## Directory Structure

```
OPRO Qwen/
│
├── data/
│   ├── external/                     # External datasets (RIRS_NOISES)
│   └── processed/                    # Processed audio and manifests
│       ├── conditions_final/         # Final dataset (111 MB)
│       │   ├── snr/                  # SNR variants (6 levels)
│       │   ├── duration/             # Duration variants (8 levels)
│       │   ├── band/                 # Band-pass filter variants (3 types)
│       │   ├── rir/                  # Reverb variants (3 T60 bins)
│       │   ├── conditions_manifest.parquet  # Dataset manifest
│       │   └── conditions_manifest.jsonl
│       ├── padded/                   # Original audio padded to 2000ms (44 MB)
│       │   ├── speech/
│       │   └── nonspeech/
│       ├── qsm_dev_1000ms_only.jsonl # Main manifest (32 KB)
│       └── unified_annotations.parquet
│
├── results/
│   ├── debug_2clips_v2/              # Latest evaluation results (308 KB)
│   │   ├── debug_log.txt             # Full evaluation log
│   │   ├── debug_results.parquet     # Results table
│   │   ├── debug_results.json        # Results in JSON
│   │   ├── snr_analysis.csv          # SNR measurement analysis
│   │   └── audio_samples/            # 3 problematic samples
│   ├── evaluation_config.json        # Evaluation configuration
│   ├── evaluation_metrics.json       # Metrics definitions
│   └── evaluation_results.parquet    # Previous results
│
├── scripts/
│   ├── build_conditions.py           # Generate psychoacoustic variants
│   ├── debug_evaluate.py             # Detailed evaluation with logging
│   ├── analyze_snr_samples.py        # SNR measurement verification
│   ├── cleanup_project.py            # Project cleanup script
│   └── [other scripts...]
│
├── src/
│   └── qsm/                          # QSM audio library
│       ├── audio/
│       │   ├── noise.py              # SNR mixing functions
│       │   ├── filters.py            # Band-pass filters
│       │   ├── duration.py           # Duration manipulation
│       │   └── reverb.py             # Reverb application
│       └── [other modules...]
│
└── docs/
    ├── HALLAZGOS_SNR_INVESTIGATION.md   # SNR investigation report
    └── PROJECT_STRUCTURE.md             # This file
```

---

## Key Files

### Data Files

| File | Size | Description |
|------|------|-------------|
| `data/processed/conditions_final/` | 111 MB | Final dataset with 20 variants per clip |
| `data/processed/padded/` | 44 MB | Original audio padded to 2000ms |
| `data/processed/qsm_dev_1000ms_only.jsonl` | 32 KB | Main manifest file |

### Results Files

| File | Description |
|------|-------------|
| `results/debug_2clips_v2/debug_log.txt` | Complete evaluation log with per-sample details |
| `results/debug_2clips_v2/debug_results.parquet` | Results in parquet format (80 samples) |
| `results/debug_2clips_v2/snr_analysis.csv` | SNR verification measurements |

### Documentation

| File | Description |
|------|-------------|
| `HALLAZGOS_SNR_INVESTIGATION.md` | Complete SNR investigation report |
| `PROJECT_STRUCTURE.md` | This file - project organization |

---

## Dataset Details

### conditions_final/ Structure

```
conditions_final/
├── snr/              # 6 levels: -10, -5, 0, +5, +10, +20 dB
├── duration/         # 8 levels: 20, 40, 60, 80, 100, 200, 500, 1000 ms
├── band/             # 3 filters: telephony, lp3400, hp300
└── rir/              # 3 T60 bins: 0.0-0.4, 0.4-0.8, 0.8-1.5 s
```

**Total Variants per Clip**: 20 (8 duration + 6 SNR + 3 band + 3 RIR)

**Sample Naming Convention**:
- `{clip_id}_snr{level}db.wav` - SNR variant
- `{clip_id}_dur{duration}ms.wav` - Duration variant
- `{clip_id}_band{filter}.wav` - Band-pass variant
- `{clip_id}_rir_T60_{min}-{max}.wav` - Reverb variant

---

## Evaluation Results Summary

### Latest Evaluation (debug_2clips_v2/)

**Configuration**:
- Clips: 2 per class (4 total)
- Variants: 20 per clip
- Total samples: 80
- Model: Qwen2-Audio-7B-Instruct (4-bit quantized)
- Prompt: Multiple choice (A/B/C/D)

**Results**:
```
Overall Accuracy: 96.25% (77/80 correct)

By Variant Type:
  Duration:  96.9% (31/32)
  SNR:       91.7% (22/24)
  Band:     100.0% (12/12)
  RIR:      100.0% (12/12)

By Ground Truth:
  SPEECH:    92.5% (37/40)
  NONSPEECH: 100.0% (40/40)

Confusion Matrix:
              Predicted
              NONSPEECH  SPEECH
True NONSPEECH       40       0
     SPEECH           3      37
```

**Key Findings**:
1. SNR generation is correct (measured vs expected: <1% error)
2. Only 2 SNR errors are from the same problematic clip
3. Model is excellent on NONSPEECH (100%)
4. SPEECH errors concentrated in extreme conditions (dur=20ms, SNR=-10/0dB)

---

## Scripts Usage

### Generate Dataset

```bash
python scripts/build_conditions.py \
  --input_manifest data/processed/qsm_dev_1000ms_only.jsonl \
  --output_dir data/processed/conditions_final/ \
  --snr_levels -10 -5 0 5 10 20 \
  --band_filters telephony lp3400 hp300 \
  --rir_root data/external/RIRS_NOISES/RIRS_NOISES \
  --rir_metadata data/external/RIRS_NOISES/rir_metadata.json \
  --rir_t60_bins 0.0-0.4 0.4-0.8 0.8-1.5 \
  --n_workers 4
```

### Run Evaluation

```bash
# Small test (2 clips)
python scripts/debug_evaluate.py \
  --n_clips 2 \
  --output_dir results/debug_2clips_v2

# Larger evaluation (50 clips)
python scripts/debug_evaluate.py \
  --n_clips 50 \
  --output_dir results/eval_50clips \
  --seed 42
```

### Verify SNR

```bash
python scripts/analyze_snr_samples.py
```

### Cleanup Project

```bash
python scripts/cleanup_project.py
```

---

## Cleanup History

**Date**: 2025-10-12

**Removed**:
- Old test directories: `custom_prompt_1`, `debug_10clips`, `debug_20clips`, etc.
- Old dataset versions: `conditions/`, `conditions_refactored/`, etc.
- Temporary files and old manifests

**Space Freed**: 1.27 GB (1299 MB)

**Kept**:
- `conditions_final/` - Final correct dataset
- `debug_2clips_v2/` - Latest evaluation with best results
- `padded/` - Original audio (needed for regeneration)
- Documentation and configuration files

---

## Next Steps

1. **Run larger evaluation** (50-100 clips) for statistical significance
2. **Analyze problematic clips** - Identify why certain clips fail
3. **Fine-tune model** - If needed, fine-tune on noisy samples
4. **Document edge cases** - Keep list of difficult samples

---

## Contact

For questions about this project structure or evaluation results, refer to:
- `HALLAZGOS_SNR_INVESTIGATION.md` - Complete technical investigation
- `results/debug_2clips_v2/debug_log.txt` - Detailed evaluation logs
