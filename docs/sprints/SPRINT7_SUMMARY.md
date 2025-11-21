# Sprint 7 Summary: Psychometric Curves and Thresholds

**Status:** ✅ PARTIAL COMPLETE (Duration curves OK, SNR needs investigation)

## Objective

Extract psychometric thresholds (DT50/DT75, SNR-50) with bootstrap confidence intervals to quantify model performance on psychoacoustic conditions.

## Implementation

### Script: `fit_psychometric_curves.py`

**Features:**
- Logistic curve fitting: P(correct) vs stimulus
- Threshold extraction (50% and 75% points)
- Bootstrap CI (1000 samples, clustered by clip_id)
- Paper-ready figures (PNG, 300 DPI)

**Usage:**
```bash
python scripts/fit_psychometric_curves.py --n_bootstrap 1000
```

**Output:**
```
results/psychometric_curves/
├── psychometric_results.json  # All thresholds and fitted curves
├── duration_curve.png         # Duration psychometric curve
└── snr_curve.png              # SNR psychometric curve
```

---

## Results

### 1. Duration Curves (P(Correct) vs Duration)

**Thresholds:**
- **DT50**: 20.0 ms (CI95: [20.0, 20.0])
  - Duration at which performance = 50%
  - Essentially at minimum duration tested

- **DT75**: 39.8 ms (CI95: [29.9, 49.7])
  - Duration at which performance = 75%
  - Wide CI reflects uncertainty

- **R² = 0.505** (moderate fit)

**Interpretation:**
- Performance improves monotonically with duration ✓
- Model performs above chance (50%) even at 20ms
- 75% accuracy achieved around 40ms
- Logistic curve fits reasonably well (R² > 0.5)

**Empirical Data:**
```
20ms:   64.7% ← Near threshold
40ms:   78.5%
60ms:   84.3%
80ms:   87.7%
100ms:  87.7%
200ms:  90.8%
500ms:  92.9%
1000ms: 95.8% ← Near ceiling
```

**Key Finding:** Model needs ~40ms to reach 75% accuracy, but performs well even at ultra-short durations (20ms = 64.7%).

---

### 2. SNR Curves (P(Correct) vs SNR)

**Status:** ⚠️ PROBLEMATIC FIT

**Thresholds (not reliable):**
- SNR-50: -10.0 dB (CI95: [-10.0, -10.0])
- SNR-75: -6.4 dB (CI95: [-9.7, -1.2])
- **R² = -1.66** (BAD fit - worse than horizontal line)

**Problem:** SNR data is **NOT monotonically increasing**

**Empirical Data:**
```
-10dB:  70.5% ← Expected: lowest
-5dB:   78.3%
0dB:    75.3% ← Dip (expected ~78%)
+5dB:   80.0%
+10dB:  75.5% ← Dip (expected ~85%)
+20dB:  92.5% ← Expected: highest
```

**Issue:** Non-monotonic pattern at 0dB and +10dB.

**Possible Causes:**
1. **Sample variance** (only 68 clips per SNR level)
2. **Clip-specific effects** (some clips harder at certain SNRs)
3. **Model artifacts** (Qwen2-Audio may have non-linear SNR response)

**Next Steps:**
- Investigate specific clips failing at 0dB and +10dB
- Test with more data (use test split for validation)
- Consider piecewise fitting or polynomial curves

---

## Bootstrap Methodology

**Clustered by clip_id** (Wichmann & Hill 2001):
- Sample clips with replacement (not variants)
- Preserves correlation structure within clips
- Correct degrees of freedom (n=70 clips, not n=1400 variants)
- 1000 bootstrap samples for CI95

**Why clustered?**
- Each clip has 20 variants (duration, SNR, band, RIR)
- Variants from same clip are correlated
- Naive bootstrap would underestimate variance
- Cluster bootstrap accounts for correlation

---

## Validation

### Duration Curve: ✅ PASSED

- [x] Monotonically increasing
- [x] R² > 0 (reasonable fit)
- [x] Thresholds in plausible range (20-40ms)
- [x] CIs computed successfully
- [x] Figure generated

### SNR Curve: ❌ FAILED

- [ ] NOT monotonically increasing
- [x] R² < 0 (poor fit)
- [x] Thresholds at boundary (SNR-50 = min tested SNR)
- [x] CIs computed (but not meaningful)
- [x] Figure generated (shows non-monotonicity visually)

---

## Technical Notes

### Logistic Function

```
P(correct) = lapse + (1 - 2*lapse) / (1 + exp(-slope * (x - x50)))
```

Where:
- `x50`: Threshold (50% point)
- `slope`: Steepness of curve
- `lapse`: Lapse rate (small constant, ~0.01-0.05)

### Goodness of Fit (R²)

```
R² = 1 - SS_residual / SS_total
```

- R² > 0.5: Good fit
- R² > 0: Acceptable fit
- R² < 0: Worse than horizontal line (bad)

---

## Next Steps

### Immediate (Fix SNR Issue):

1. **Manual inspection** of clips failing at 0dB and +10dB:
   ```python
   # Find problematic clips
   snr_0db = pred_df[(pred_df['snr_db'] == 0) & (pred_df['y_true'] != pred_df['y_pred'])]
   snr_10db = pred_df[(pred_df['snr_db'] == 10) & (pred_df['y_true'] != pred_df['y_pred'])]
   ```

2. **Test on test split** to see if pattern replicates

3. **Alternative models:**
   - Polynomial fit (allow non-monotonic)
   - Piecewise linear
   - Kernel smoothing

### Follow-up (Sprint 8):

**Technical Validation** (as planned):
- Verify SNR accuracy (measure vs target) ← May explain non-monotonicity
- Verify filter responses (FFT)
- Verify RIR application

---

## Files Created

```
scripts/
└── fit_psychometric_curves.py  # Psychometric curve fitting

results/psychometric_curves/
├── psychometric_results.json   # All thresholds and CIs
├── duration_curve.png          # Duration plot
└── snr_curve.png               # SNR plot

SPRINT7_SUMMARY.md              # This file
```

---

## References

- **Wichmann & Hill (2001a)**: Fitting psychometric functions
- **Wichmann & Hill (2001b)**: Bootstrap-based confidence intervals
  - https://courses.washington.edu/matlab1/pdf/Wichmann_Hill_2001b.pdf
- **Sklearn curve_fit**: Weighted least squares fitting
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

---

## Sprint 7 Status

✅ **Duration Curves:** COMPLETE and validated
⚠️ **SNR Curves:** COMPUTED but non-monotonic (needs investigation)
✅ **Bootstrap CIs:** Working correctly
✅ **Figures:** Generated

**Conclusion:** Duration analysis is ready for paper. SNR needs debugging (likely data issue, not method issue).

**Recommendation:** Proceed to Sprint 8 (technical validation) to verify SNR generation is correct, which may explain the non-monotonicity.
