# OPRO Qwen - Speech Detection with Psychoacoustic Conditions

![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen)
![Accuracy](https://img.shields.io/badge/accuracy-96.25%25-blue)

Speech detection system using Qwen2-Audio-7B-Instruct evaluated across psychoacoustic conditions (duration, SNR, band-limiting, reverb).

## Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd OPRO-Qwen

# Install dependencies
pip install -r requirements.txt
```

### Pipeline Validation (10 seconds)
```bash
python scripts/validate_evaluation_pipeline.py
```

### Full Evaluation with Robust Metrics
```bash
# Evaluate on dev split (15-20 min with GPU)
python scripts/evaluate_with_robust_metrics.py --split dev

# Evaluate on test split (after hyperparameters frozen)
python scripts/evaluate_with_robust_metrics.py --split test
```

### Recompute Metrics from Saved Predictions
```bash
# Fast re-analysis without model loading
python scripts/recompute_metrics.py
```

### Fit Psychometric Curves
```bash
# Fit logistic curves and extract thresholds
python scripts/fit_psychometric_curves.py --n_bootstrap 1000

# Output: results/psychometric_curves/
#   - psychometric_results.json
#   - duration_curve.png
#   - snr_curve.png
```

## Current Results

**Overall Performance: 96.25% (77/80 correct)**

| Condition | Accuracy | Notes |
|-----------|----------|-------|
| Duration  | 96.9%    | Expected failure at 20ms |
| SNR       | 91.7%    | 2 errors at high noise levels |
| Band-limiting | 100.0% | Perfect performance |
| Reverb (RIR) | 100.0% | Perfect performance |

### Psychometric Thresholds (MLE Binomial Fitting)

**Duration Thresholds (Sprint 7 Revised):**
- **DT75**: 34.8 ms (CI95: [20.0, 64.3]) - **PRIMARY METRIC**
- **DT50**: 26.8 ms (CI95: [20.0, 45.9])
- **McFadden R² = 0.063** (moderate fit, monotonically increasing)
- **Tjur R² = 0.056** (discrimination index)
- Model requires ~35ms to reach 75% accuracy

**SNR Thresholds (Sprint 7 - Under Investigation):**
- **SNR-75**: -4.9 dB (CI95: [-10.0, 2.5]) - **PRIMARY METRIC**
- **McFadden R² = 0.018** (poor fit, non-monotonic pattern)
- Status: Duration mixing effect - needs GLM/GLMM with SNR×Duration interaction
- See [SPRINT7_REVISED_SUMMARY.md](SPRINT7_REVISED_SUMMARY.md) for methodology

### Dataset Statistics
- **Total clips**: 87 (40 SPEECH + 47 NONSPEECH)
- **Total samples**: 1,740 (87 clips × 20 variants)
- **Variants per clip**: 20
  - 8 duration variants (20ms to 1000ms)
  - 6 SNR levels (-10dB to +20dB)
  - 3 band-limiting conditions
  - 3 room impulse responses

## Project Structure

```
OPRO-Qwen/
├── data/
│   └── processed/
│       ├── conditions_final/    # Final dataset (111 MB)
│       │   ├── duration/        # Duration variants
│       │   ├── snr/             # SNR variants
│       │   ├── band/            # Band-limiting variants
│       │   └── rir/             # Reverb variants
│       └── padded/              # Original 1000ms clips (44 MB)
│
├── results/
│   ├── test_final/              # Latest evaluation results
│   ├── debug_2clips_v2/         # SNR investigation results
│   └── psychometric_curves/     # Psychometric curves and thresholds
│
├── scripts/
│   ├── validate_evaluation_pipeline.py  # Pipeline validation (no model)
│   ├── evaluate_with_robust_metrics.py  # Full evaluation with robust metrics
│   ├── recompute_metrics.py             # Re-analyze saved predictions
│   ├── create_train_test_split.py       # Create stratified dev/test split
│   └── fit_psychometric_curves.py       # Psychometric curve fitting
│
├── src/qsm/
│   ├── audio/
│   │   ├── duration.py          # Duration truncation
│   │   ├── noise.py             # SNR mixing (verified correct)
│   │   ├── bandlimit.py         # Band-limiting filters
│   │   └── reverb.py            # Reverb application
│   └── models/
│       └── qwen_audio.py        # Qwen2-Audio wrapper
│
└── docs/
    ├── HALLAZGOS_SNR_INVESTIGATION.md  # SNR investigation report
    ├── PROJECT_STRUCTURE.md            # Detailed project layout
    └── TESTING_GUIDE.md                # Complete testing guide
```

## Documentation

- **[Sprint 6 Summary](SPRINT6_SUMMARY.md)** - Robust evaluation pipeline with stratified split
- **[Sprint 7 Revised Summary](SPRINT7_REVISED_SUMMARY.md)** - MLE psychometric curves with pseudo-R²
- **[SNR Investigation Report](HALLAZGOS_SNR_INVESTIGATION.md)** - Complete technical analysis of SNR generation and validation
- **[Evaluation Guide](EVALUATION_GUIDE.md)** - Complete workflow for running evaluations
- **[Project Structure](PROJECT_STRUCTURE.md)** - Detailed organization after cleanup
- **[Testing Guide](TESTING_GUIDE.md)** - How to run different test levels
- **[Cleanup Summary](CLEANUP_SUMMARY.txt)** - What was cleaned and why (1.27 GB freed)

## Known Issues & Solutions

### Issue 1: SNR=0dB Lower Accuracy (91.7%)
**Status**: Not a bug - verified correct generation  
**Cause**: Specific clips are harder to classify at certain SNR levels  
**Evidence**: Measured SNR ratio 3.320 vs expected 3.317 (<1% error)  
**See**: [HALLAZGOS_SNR_INVESTIGATION.md](HALLAZGOS_SNR_INVESTIGATION.md)

### Issue 2: Path Format Issues (Windows/Linux)
**Status**: Fixed  
**Solution**: All scripts now convert Windows backslashes to forward slashes

### Issue 3: Column Naming Inconsistency
**Status**: Fixed  
**Solution**: Standardized on `ground_truth` column across all scripts

## Dataset Generation

To regenerate the psychoacoustic conditions:

```bash
# Generate all conditions (duration, SNR, band, RIR)
python scripts/generate_conditions_final.py

# Generate specific condition types
python scripts/generate_conditions_final.py --condition_types snr band
```

## Model Configuration

**Default Model**: Qwen2-Audio-7B-Instruct  
**Quantization**: 4-bit (requires ~5 GB VRAM)  
**Sample Rate**: 16 kHz  
**Auto-padding**: 1000ms clips → 2000ms containers with low-amplitude noise

**Optimal Prompt** (hardcoded default):
```
What best describes this audio?
A) Human speech or voice
B) Music
C) Noise or silence
D) Animal sounds

Answer with ONLY the letter (A, B, C, or D).
```

## Performance Tuning

- **Multiple choice prompting**: 96% accuracy (current)
- **Custom yes/no prompting**: 55% accuracy (not recommended)
- **Optimal padding**: 2000ms containers (verified)
- **Optimal duration**: 1000ms effective segments

## Evaluation Workflow

| Script | Duration | Model | Purpose |
|--------|----------|-------|---------|
| `validate_evaluation_pipeline.py` | 10s | No | Validate pipeline logic |
| `smoke_test.py` | 30s | No | CI smoke test |
| `evaluate_with_robust_metrics.py` | 15-20min | Yes | Full evaluation with metrics |
| `recompute_metrics.py` | 10s | No | Re-analyze saved predictions |

## Recent Changes

### Sprint 7 (Revised with MLE Fitting)
- ✅ MLE binomial fitting (Wichmann & Hill 2001)
- ✅ Fixed gamma=0.5, free lapse parameter [0, 0.1]
- ✅ Pseudo-R² metrics (McFadden & Tjur)
- ✅ Duration curves: DT75=35ms [20, 64], McFadden R²=0.063 (COMPLETE)
- ⚠️ SNR curves: SNR-75=-5dB, McFadden R²=0.018 (non-monotonic, needs GLM)
- ✅ Paper-ready figures with log-scale x-axis (PNG, 300 DPI)

### Sprint 6 Completion
- ✅ Stratified dev/test split (80/20) with reproducibility
- ✅ Clip-level aggregation (majority vote across 20 variants)
- ✅ Robust metrics (Balanced Accuracy, Macro-F1)
- ✅ Condition-specific analysis corrected
- ✅ All scripts renamed to functional names
- ✅ Repository cleanup (12 debug/test scripts removed)

### Sprint 5 Completion
- ✅ Psychoacoustic condition generators (SNR, band-limiting, reverb)
- ✅ Qwen2-Audio validation complete (96.25% accuracy)
- ✅ SNR generation verified correct (detailed investigation)
- ✅ Project cleanup (1.27 GB freed)
- ✅ Testing framework (3-tier system)
- ✅ All bugs fixed (paths, column names, validation)

### Bugs Fixed
1. Label column naming (`label_normalized` → `ground_truth`)
2. Cross-platform path handling (Windows backslashes)
3. Sample size validation (auto-adjust to available clips)
4. Empty DataFrame handling in analysis
5. Unicode encoding issues on Windows

## Contributing

When adding new features:
1. Run `python scripts/simple_test.py` (smoke test)
2. Run `python scripts/quick_validation.py` (validation)
3. Document changes in appropriate files
4. Update test expectations if needed

## License

[Add license information]

## Contact

[Add contact information]
