# Testing Guide - OPRO Qwen

Quick guide to test the pipeline with minimal examples.

---

## 🚀 Quick Start (Choose One)

### Option 1: Ultra-Fast Smoke Test (~30 seconds)

**What it tests**: Dataset integrity, SNR accuracy, audio files
**What it skips**: Model loading/inference

```bash
python scripts/smoke_test.py
```

**Expected output**:
```
============================================================
SMOKE TEST - Quick Pipeline Validation
============================================================

1. Testing dataset...
   [OK] Manifest loaded: XXXX samples
   [OK] Sample files exist

2. Testing SNR accuracy...
   [OK] SNR measurements correct

3. Testing audio files...
   [OK] All variant types readable

============================================================
SUMMARY
============================================================
[OK] Dataset
[OK] SNR
[OK] Audio

[OK] SMOKE TEST PASSED

Next: Run full validation
  python scripts/quick_validation.py
```

---

### Option 2: Quick Validation with Model (~2-3 minutes)

**What it tests**: Everything + model inference on 1 clip per class (40 samples)

```bash
python scripts/quick_validation.py
```

**Expected output**:
```
======================================================================
TEST 1: Dataset Integrity
======================================================================

✓ Manifest loaded: XXXX samples
✓ All required columns present
✓ All variant types present
✓ File exists: ...

✅ PASS: Dataset integrity validated

======================================================================
TEST 2: SNR Measurement
======================================================================

✓ SNR=-10dB: ratio=3.320 (expected=3.317, error=0.1%)
✓ SNR= 0dB: ratio=1.414 (expected=1.414, error=0.0%)
✓ SNR=+10dB: ratio=1.005 (expected=1.005, error=0.0%)

✅ PASS: SNR measurements within tolerance

======================================================================
TEST 3: Quick Evaluation (1 clip per class)
======================================================================

This will take ~2-3 minutes...
Loading model and running evaluation on 40 samples

Results:
  Total samples: 40
  Correct: 38
  Accuracy: 95.0%

By variant type:
  duration    :  93.8% (n=16)
  snr         :  91.7% (n=12)
  band        : 100.0% (n=6)
  rir         : 100.0% (n=6)

✅ PASS: Evaluation completed successfully

======================================================================
VALIDATION SUMMARY
======================================================================

Test Results:
  1. Dataset Integrity:  ✅ PASS
  2. SNR Measurement:    ✅ PASS
  3. Quick Evaluation:   ✅ PASS

✅ ALL TESTS PASSED - Pipeline is working correctly!
```

---

### Option 3: Full Evaluation (Optional, ~10-15 minutes)

**What it tests**: Complete evaluation on 50 clips (1000 samples)

```bash
python scripts/debug_evaluate.py --n_clips 50 --output_dir results/eval_50clips --seed 42
```

⚠️ **Note**: Only run this if quick validation passed and you need detailed statistics.

---

## 📊 What Each Test Does

### Smoke Test (30 sec)
- ✅ Checks dataset manifest exists
- ✅ Verifies 3 random audio files exist
- ✅ Measures SNR on 3 samples (-10dB, 0dB, +10dB)
- ✅ Tests reading one file per variant type
- ❌ Does NOT load model (too slow)

### Quick Validation (2-3 min)
- ✅ All smoke test checks
- ✅ Loads Qwen2-Audio model (4-bit quantized)
- ✅ Evaluates 1 clip per class = 40 samples
- ✅ Calculates accuracy metrics
- ✅ Verifies results are reasonable (>70%)

### Full Evaluation (10-15 min)
- ✅ Complete evaluation on N clips
- ✅ Detailed per-variant statistics
- ✅ Saves results to parquet/JSON
- ✅ Copies problematic audio samples

---

## 🔧 Troubleshooting

### Smoke test fails on dataset
```bash
# Check if dataset exists
ls data/processed/conditions_final/

# If missing, regenerate:
python scripts/build_conditions.py \
  --input_manifest data/processed/qsm_dev_1000ms_only.jsonl \
  --output_dir data/processed/conditions_final/ \
  --snr_levels -10 -5 0 5 10 20 \
  --band_filters telephony lp3400 hp300 \
  --rir_root data/external/RIRS_NOISES/RIRS_NOISES \
  --rir_metadata data/external/RIRS_NOISES/rir_metadata.json \
  --rir_t60_bins 0.0-0.4 0.4-0.8 0.8-1.5
```

### Smoke test fails on SNR
```bash
# Check SNR implementation
python scripts/analyze_snr_samples.py

# Should show SNR error <10% for all levels
```

### Quick validation fails on model loading
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, check CUDA installation
nvidia-smi
```

### Evaluation accuracy too low (<70%)
```bash
# Check which variants fail
python -c "
import pandas as pd
df = pd.read_parquet('results/quick_validation/debug_results.parquet')
print(df[~df['correct']][['clip_id', 'variant_type', 'ground_truth', 'predicted']])
"

# Check if it's specific clips or systematic
```

---

## 📈 Expected Performance

Based on previous evaluations:

| Test | Expected Accuracy | Sample Size |
|------|------------------|-------------|
| Smoke Test | N/A (no inference) | 3 SNR samples |
| Quick Validation | 85-100% | 40 samples (1 clip × 2 classes × 20 variants) |
| Full Evaluation (2 clips) | 95-98% | 80 samples |
| Full Evaluation (50 clips) | 92-96% | 2000 samples |

**Variant-specific expectations**:
- Duration: 95-100% (except dur=20ms which may fail)
- SNR: 90-95% (some clips harder with noise)
- Band: 95-100%
- RIR: 95-100%

---

## 🎯 When to Run What

**Just cleaned up project?**
→ Run smoke test (30 sec)

**After code changes?**
→ Run quick validation (2-3 min)

**Need statistics for paper/report?**
→ Run full evaluation with 50-100 clips (10-20 min)

**Debugging specific issue?**
→ Run debug_evaluate.py with --n_clips 2 and check logs

---

## 📝 Output Files

### Smoke Test
- No output files (prints to console only)

### Quick Validation
```
results/quick_validation/
├── debug_results.parquet     # Results table
├── debug_results.json         # Results in JSON
└── debug_log.txt              # Full evaluation log
```

### Full Evaluation
```
results/eval_50clips/
├── debug_results.parquet      # Results table
├── debug_results.json         # Results in JSON
├── debug_log.txt              # Full evaluation log
└── audio_samples/             # Problematic samples (if any)
    └── incorrect_*.wav
```

---

## ✅ Success Criteria

**Smoke Test**: All 3 tests pass
- Dataset: Files exist and readable
- SNR: Measurements within 20% of expected
- Audio: All variant types readable

**Quick Validation**: All 3 tests pass
- Dataset integrity ✓
- SNR measurements ✓
- Evaluation accuracy >70% ✓

**Full Evaluation**:
- Overall accuracy >90%
- No systematic failures in any variant type
- Errors distributed across different clips

---

## 🚨 Red Flags

❌ **Smoke test fails**: Dataset is corrupted or missing
❌ **SNR >20% error**: Audio generation has bugs
❌ **Accuracy <70%**: Model loading or inference issues
❌ **All samples of one variant fail**: Systematic bug in that variant type
❌ **One clip fails on all variants**: Problematic clip (expected, not a bug)

---

## 📞 Getting Help

1. Check [HALLAZGOS_SNR_INVESTIGATION.md](HALLAZGOS_SNR_INVESTIGATION.md) for known issues
2. Look at debug logs in `results/*/debug_log.txt`
3. Run `analyze_snr_samples.py` to verify SNR generation
4. Check project structure in [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
