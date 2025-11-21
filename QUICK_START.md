# Quick Start Guide - Speech Detection Pipeline

**Last Updated:** 2025-11-09

---

## Prerequisites

- **Hardware:** 16GB RAM, 12GB+ VRAM GPU (CUDA-enabled)
- **Software:** Python 3.8+, PyTorch with CUDA, transformers, peft
- **Disk Space:** 100GB+ free

---

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd OPRO_Qwen

# Install dependencies
pip install torch transformers peft bitsandbytes accelerate
pip install scikit-learn pandas numpy matplotlib seaborn tqdm
pip install soundfile librosa scipy pyyaml

# Download datasets (manual)
# 1. VoxConverse dev: Place in data/raw/voxconverse/dev/
# 2. ESC-50: Place in data/raw/esc50/audio/
```

---

## Quick Test (Smoke Test)

**Validates entire pipeline in ~10-15 minutes with minimal data:**

```bash
python run_complete_pipeline.py --config config/pipeline_config.yaml --smoke_test
```

**What it does:**
- Uses 2 samples per split
- Trains 1 model for 1 epoch
- Evaluates all stages
- Generates full reports

**Check logs:** `logs/pipeline_YYYYMMDD_HHMMSS.log`

---

## Full Execution

```bash
# Dry run first (see commands without executing)
python run_complete_pipeline.py --config config/pipeline_config.yaml --dry_run

# Full pipeline (~80 hours)
python run_complete_pipeline.py --config config/pipeline_config.yaml
```

---

## Pipeline Stages

| Stage | Description | Time | Output |
|-------|-------------|------|--------|
| 0 | Environment validation | 1min | GPU/RAM check |
| 1 | Dataset check | 1min | Verify data exists |
| 2 | Prepare base clips (1000ms) | 15min | 160 base clips |
| 3 | Generate variants (8 dur × 6 SNR) | 30min | 7,680 variants |
| 4 | Train multi-seed (3 seeds) | 72h | 3 checkpoints |
| 5 | Evaluate DEV | 3h | DEV predictions |
| 6-8 | Optimize (temp/prompt/threshold) | 3h | Optimized hyperparams |
| 9 | Evaluate TEST (once) | 3h | Final results |
| 10 | Psychometric curves | 30min | DT75, SNR-75 |
| 11 | Baselines (Silero VAD) | 15min | Baseline comparison |
| 12-13 | Aggregate & report | 10min | Final report |

---

## Key Files

### Input
- `config/pipeline_config.yaml` - All configuration
- `data/raw/voxconverse/dev/` - Speech samples
- `data/raw/esc50/audio/` - Non-speech samples

### Output
- `logs/pipeline_YYYYMMDD_HHMMSS.log` - Main log (share this for debugging)
- `checkpoints/qwen_lora_seed*/` - Trained models
- `results/` - All evaluation results
- `PIPELINE_EXECUTION_REPORT.md` - Final report

---

## Expected Results

### Test Set (1000ms, 20dB):
- **Silero VAD:** 66.7%
- **Qwen2 + LoRA:** 83.3% (ROC-AUC=1.0)
- **+ Optimized Prompt:** 83.3%
- **+ Optimal Threshold:** **100.0%** ✨

### Psychometric Metrics:
- **DT75:** ~35ms
- **SNR-75:** ~-2.9dB

---

## Troubleshooting

### OOM Errors
```yaml
# Edit config/pipeline_config.yaml
training:
  hyperparameters:
    per_device_train_batch_size: 1  # Reduce if needed
    gradient_accumulation_steps: 8   # Increase to compensate
```

### Missing Datasets
Download manually and place in correct directories (see Installation)

### Check Logs
All stages log to individual files: `logs/HHMMSS_<stage_name>.log`

---

## Documentation

- **Full Pipeline Details:** [PIPELINE_IMPLEMENTATION_SUMMARY.md](PIPELINE_IMPLEMENTATION_SUMMARY.md)
- **Project Summary:** [COMPLETE_PROJECT_SUMMARY.md](COMPLETE_PROJECT_SUMMARY.md)
- **Navigation Guide:** [INDEX.md](INDEX.md)
- **Main README:** [README.md](README.md)

---

## Support

**Logs Location:** All execution logs saved to `logs/` with timestamps

**To share for debugging:**
1. Run pipeline (or smoke test)
2. Share the main log: `logs/pipeline_YYYYMMDD_HHMMSS.log`
3. Share any failed stage logs: `logs/HHMMSS_<stage_name>.log`

---

**Note:** This pipeline implements best practices including:
- Zero-leakage GroupShuffleSplit
- Hyperparameter optimization on DEV only
- TEST evaluated exactly once
- Multi-seed reproducibility (3 seeds)
- Bootstrap confidence intervals
