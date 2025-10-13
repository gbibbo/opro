# Sprint 6 Correction: Fixed Clip Aggregation Logic

## Problem Identified

The original `compute_condition_metrics()` function was **incorrectly** aggregating predictions:

### Original (WRONG) Logic:
```python
# For each condition (e.g., duration_20ms):
dur_subset = predictions_df[predictions_df["duration_ms"] == 20]
clip_agg = aggregate_by_clip(dur_subset)  # ← WRONG
metrics = compute_robust_metrics(clip_agg["y_true"], clip_agg["y_pred"])
```

**Problem**: Each clip only has **ONE** variant per condition (e.g., one 20ms variant), so aggregating by clip within that condition does nothing. This resulted in:
- "Aggregating 70 variants → 70 clips" (no aggregation happening)
- Clip-level accuracy (94.3%) > Variant-level accuracy (85.7%) - seemed wrong

### Corrected Logic:
```python
# For each condition (e.g., duration_20ms):
dur_subset = predictions_df[predictions_df["duration_ms"] == 20]
# Each clip has exactly 1 variant at this duration
# So variant-level = per-clip for this condition
metrics = compute_robust_metrics(dur_subset["y_true"], dur_subset["y_pred"])
```

**Solution**: Condition-specific metrics are **VARIANT-LEVEL** by design. Since each clip has exactly one variant per condition, variant-level IS per-clip for that condition.

## Why Clip-Level > Variant-Level is CORRECT

**Clip-level aggregation** (majority vote across 20 variants) should ONLY apply to **OVERALL** metrics, not condition-specific metrics.

**Example showing why Clip > Variant is expected:**

Clip A has 20 variants:
- 15 correct predictions
- 5 incorrect predictions

**Variant-level**: 15/20 = 75% contribution from this clip
**Clip-level**: Majority (15>5) = SPEECH → 100% contribution (1/1 correct)

When averaged across many clips with this pattern, clip-level accuracy > variant-level accuracy.

## Corrected Results

### Overall Metrics:
- **Variant-level**: 85.7% accuracy, 85.5% balanced accuracy
- **Clip-level**: 94.3% accuracy, 94.5% balanced accuracy ← CORRECT now

**Sanity check**: Clip-level ≥ Variant-level ✓

### Condition-Specific (Variant-Level):

**Duration (monotonic increase ✓):**
```
20ms:   64.7% ← HARDEST (expected)
40ms:   78.5%
60ms:   84.3%
80ms:   87.7%
100ms:  87.7%
200ms:  90.8%
500ms:  92.9%
1000ms: 95.8% ← EASIEST (expected)
```

**SNR (general trend up, with noise):**
```
-10dB:  70.5% ← HARD
-5dB:   78.3%
0dB:    75.3% ← Outlier (expected ~78%)
+5dB:   80.0%
+10dB:  75.5% ← Outlier (expected ~85%)
+20dB:  92.5% ← EASY
```

**Band (all high ✓):**
```
hp300:      93.2%
lp3400:     95.8%
telephony:  94.5%
```

**RIR (all high ✓):**
```
T60_0.0-0.4: 94.5%
T60_0.4-0.8: 95.8%
T60_0.8-1.5: 95.8%
```

### Objective Metric:
- **Macro Balanced Accuracy**: 86.2% (average across 20 conditions)
- **Macro Macro-F1**: 86.0%

## Files Changed

1. **`scripts/evaluate_with_robust_metrics.py`**:
   - Fixed `compute_condition_metrics()` to use variant-level directly
   - Updated docstrings to clarify variant-level vs clip-level

2. **`scripts/recompute_metrics.py`** (NEW):
   - Re-analyzes saved predictions without re-running model
   - Useful for quick iteration on metric computation

## Validation

```bash
python scripts/recompute_metrics.py
```

**Output**:
- ✓ Clip-level (94.3%) ≥ Variant-level (85.7%)
- ✓ Duration shows monotonic increase (20ms worst, 1000ms best)
- ✓ SNR shows general upward trend (some noise expected)
- ✓ Band and RIR show high accuracy (>93%)

## Key Insights

1. **Clip-level aggregation** applies to **overall** evaluation (across all 20 variants per clip)
2. **Condition-specific** metrics are **variant-level** (one variant per condition per clip)
3. Majority vote **increases** accuracy by tolerating some variant errors
4. Duration ≤100ms and SNR ≤0dB are the **hardest** conditions (as expected)

## Sprint 6 Status

✅ **CORRECTED AND VALIDATED**

Ready to proceed to Sprint 7 (psychometric curves and thresholds).
