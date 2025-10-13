# Evaluation Pipeline Guide

Complete guide for evaluating Qwen2-Audio on psychoacoustic conditions.

## Quick Reference

| Script | Purpose | Duration | Requires Model |
|--------|---------|----------|----------------|
| `create_train_test_split.py` | Create stratified dev/test split | <1s | No |
| `validate_evaluation_pipeline.py` | Validate pipeline logic | ~10s | No |
| `evaluate_with_robust_metrics.py` | Full evaluation with robust metrics | 15-20min | Yes (GPU) |
| `recompute_metrics.py` | Re-analyze saved predictions | ~10s | No |

---

## Step 1: Create Train/Test Split

**Purpose:** Create reproducible stratified 80/20 split (dev/test).

```bash
python scripts/create_train_test_split.py --seed 42 --test_size 0.2
```

**Output:**
```
data/processed/conditions_final/
├── conditions_manifest_split.parquet  (manifest with 'split' column)
└── conditions_manifest_split.metadata.json  (split metadata)
```

**Features:**
- Stratification by dataset (esc50/voxconverse) and label (SPEECH/NONSPEECH)
- All 20 variants of same clip stay in same split (no leakage)
- Fixed seed for reproducibility

**Result:**
- Dev: 70 clips (1,400 variants)
- Test: 17 clips (340 variants)

---

## Step 2: Validate Pipeline (Optional but Recommended)

**Purpose:** Fast validation without loading the expensive model.

```bash
python scripts/validate_evaluation_pipeline.py
```

**Tests:**
1. Split reproducibility (same seed → same split)
2. Clip grouping logic (majority vote)
3. Robust metrics computation (Balanced Acc, Macro-F1)
4. Output structure

**Expected:** All 4 tests pass

---

## Step 3: Full Evaluation

**Purpose:** Evaluate model on dev split with robust metrics.

```bash
# Evaluate on dev split (15-20 min with GPU)
python scripts/evaluate_with_robust_metrics.py --split dev

# Options:
python scripts/evaluate_with_robust_metrics.py \
    --split dev \
    --output_dir results/robust_eval \
    --model_name "Qwen/Qwen2-Audio-7B-Instruct" \
    --device cuda \
    --seed 42
```

**Output:**
```
results/robust_eval/
├── dev_predictions.parquet  (variant-level predictions with metadata)
├── dev_clips.parquet        (clip-level aggregation via majority vote)
└── dev_metrics.json         (all metrics hierarchically organized)
```

**Metrics Computed:**

1. **Variant-level** (reference, includes inflation):
   - Accuracy, Balanced Accuracy, Macro-F1

2. **Clip-level** (PRIMARY - anti-inflation):
   - Majority vote across 20 variants per clip
   - Balanced Accuracy, Macro-F1

3. **Condition-specific** (variant-level):
   - Per duration: 20ms, 40ms, ..., 1000ms
   - Per SNR: -10dB, -5dB, 0dB, +5dB, +10dB, +20dB
   - Per band filter: telephony, lp3400, hp300
   - Per T60: 0.0-0.4s, 0.4-0.8s, 0.8-1.5s

4. **Macro across conditions** (OBJECTIVE METRIC):
   - Average Balanced Accuracy across all conditions
   - Average Macro-F1 across all conditions

---

## Step 4: Re-analyze Results (Fast Iteration)

**Purpose:** Re-compute metrics from saved predictions without re-running the model.

```bash
# Use default paths
python scripts/recompute_metrics.py

# Or specify custom paths
python scripts/recompute_metrics.py \
    --predictions results/robust_eval/dev_predictions.parquet \
    --output_dir results/reanalyzed
```

**Use case:** Useful when:
- Iterating on metric computation logic
- Adding new metrics
- Debugging metric issues
- Quick exploration without GPU

---

## Understanding the Metrics

### Variant-level vs Clip-level

**Variant-level:** Each of the 1,400 predictions counts equally.
- 70 clips × 20 variants per clip = 1,400 predictions
- Problem: Same clip counted 20 times (inflation)

**Clip-level:** Majority vote across 20 variants → 1 prediction per clip.
- 70 clips → 70 predictions
- Solution: Correct degrees of freedom (n=70, not n=1,400)

**Expected:** Clip-level accuracy ≥ Variant-level accuracy
- Majority vote tolerates some variant errors
- Example: 15/20 variants correct → clip is correct (100%)

### Why Balanced Accuracy?

**Problem:** If dataset is imbalanced (70% SPEECH, 30% NONSPEECH), a naive model that always predicts SPEECH gets 70% accuracy.

**Solution:** Balanced Accuracy = (SPEECH_recall + NONSPEECH_recall) / 2
- Always-SPEECH model: Bal.Acc = (100% + 0%) / 2 = 50%
- Perfect model: Bal.Acc = (100% + 100%) / 2 = 100%

### Why Macro-F1?

Similar to Balanced Accuracy but considers precision as well.

**Macro-F1** = (F1_SPEECH + F1_NONSPEECH) / 2

Equal weight to both classes regardless of prevalence.

### Condition-Specific Metrics

For condition analysis (e.g., "How does the model perform on 20ms audio?"):
- Filter to specific condition (e.g., duration=20ms)
- Each clip has exactly 1 variant at that condition
- Compute metrics at variant-level (which equals per-clip for that condition)

This is DIFFERENT from overall clip-level aggregation (which averages across all 20 variants).

---

## Expected Results (Current Baseline)

### Overall Metrics:
- **Variant-level**: 85.7% accuracy
- **Clip-level**: 94.3% accuracy (majority vote)
- **Macro Balanced Accuracy**: 86.2%

### Duration (monotonic increase expected):
```
20ms:   64.7% ← HARDEST
40ms:   78.5%
60ms:   84.3%
80ms:   87.7%
100ms:  87.7%
200ms:  90.8%
500ms:  92.9%
1000ms: 95.8% ← EASIEST
```

### SNR (general upward trend):
```
-10dB:  70.5% ← HARD
-5dB:   78.3%
0dB:    75.3%
+5dB:   80.0%
+10dB:  75.5%
+20dB:  92.5% ← EASY
```

### Band and RIR (all high):
```
Band filters: 93-96%
RIR (reverb): 94-96%
```

---

## Troubleshooting

### Model fails to load
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Try CPU if no GPU
python scripts/evaluate_with_robust_metrics.py --device cpu
```

### Out of memory
```bash
# Use 4-bit quantization (already default)
# Model requires ~5GB VRAM with 4-bit quantization
```

### Predictions don't match
- Ensure same seed (--seed 42)
- Ensure same model checkpoint
- Temperature is hardcoded to 0 for determinism

### Metrics seem wrong
```bash
# Run validation tests
python scripts/validate_evaluation_pipeline.py

# Re-analyze saved predictions
python scripts/recompute_metrics.py
```

---

## Next Steps

After establishing baseline with robust evaluation:

1. **Sprint 7:** Psychometric curves
   - Fit logistic curves (P(SPEECH) vs duration, P(SPEECH) vs SNR)
   - Extract thresholds (DT50/DT75, SNR-50)
   - Bootstrap confidence intervals

2. **Sprint 8:** Technical validation
   - Verify SNR accuracy (measure vs target)
   - Verify filter responses (FFT analysis)
   - Verify RIR application

3. **Sprint 9:** Calibration (optional)
   - Temperature scaling for confidence
   - Expected Calibration Error (ECE)

4. **Fine-tuning:** (after baseline complete)
   - LoRA on hard conditions
   - Probe/head on encoder features

---

## File Organization

```
results/
└── robust_eval/
    ├── dev_predictions.parquet   (1,400 variant predictions)
    ├── dev_clips.parquet          (70 clip aggregations)
    └── dev_metrics.json           (all metrics)

data/processed/conditions_final/
├── conditions_manifest_split.parquet    (manifest with split)
└── conditions_manifest_split.metadata.json
```

---

## References

- **Balanced Accuracy**: Brodersen et al., 2010
- **Psychometric Curves**: Wichmann & Hill, 2001
- **Qwen2-Audio**: https://arxiv.org/pdf/2407.10759
- **Sprint Documentation**: SPRINT6_SUMMARY.md, SPRINT6_CORRECTION.md
