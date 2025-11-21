# Sprint 8: SNR×Duration Factorial Design and Stratified Analysis

## Motivation

Sprint 7 successfully implemented MLE binomial fitting with pseudo-R² for psychometric curves. **Duration curves are complete and paper-ready** (DT75 = 34.8 ms, McFadden R² = 0.063). However, **SNR curves show non-monotonic pattern** (McFadden R² = 0.018) due to a fundamental design limitation:

**Problem:** Current SNR variants all have `duration_ms = 1000` (long duration). When fitting SNR curves "overall", we're averaging performance across clips that may have different effective durations, creating non-monotonic artifacts.

**Solution:** Generate factorial SNR×Duration dataset to enable:
1. Stratified SNR curves by duration
2. GLMM with SNR×Duration interaction
3. Extended SNR range (-20 to +20 dB)

## Objectives

1. **Generate factorial dataset**: 4 durations × 8 SNR levels per clip
2. **Fit stratified SNR curves**: Separate curves for each duration level
3. **Verify monotonicity**: Check that each stratified curve is monotonically increasing
4. **Improve pseudo-R²**: Target McFadden R² > 0.05 for stratified curves
5. **Extract robust thresholds**: SNR-75 with narrow CIs that don't touch boundaries

## Design Specification

### Factorial Conditions

**Durations:** 4 levels
- 20 ms (short - near DT50)
- 80 ms (medium-short - above DT75)
- 200 ms (medium-long - plateau region)
- 1000 ms (long - ceiling performance)

**SNR levels:** 8 levels
- -20 dB (very hard, extended range)
- -15 dB (hard, extended range)
- -10 dB (current minimum)
- -5 dB
- 0 dB
- +5 dB
- +10 dB
- +20 dB (ceiling)

**Total conditions per clip:** 4 × 8 = 32 variants

### Dataset Size

**Option A: Mini-dataset (validation)**
- 20 clips (10 SPEECH + 10 NONSPEECH)
- 640 audio files (20 × 32)
- ~40 MB storage (64kB per 1000ms/16kHz WAV)
- Evaluation time: ~30 min with GPU

**Option B: Full dev set**
- 70 clips (balanced by label/dataset)
- 2,240 audio files (70 × 32)
- ~140 MB storage
- Evaluation time: ~2 hours with GPU

**Recommendation:** Start with Option A for validation, then extend to Option B if successful.

## Implementation Steps

### 1. Generate Factorial Dataset

Script: `scripts/generate_snr_duration_crossed.py`

```bash
# Generate for 20-clip subset
python scripts/generate_snr_duration_crossed.py \
    --input_file data/processed/subset_20clips_for_crossed.csv \
    --output_dir data/processed/snr_duration_crossed \
    --durations 20 80 200 1000 \
    --snr_levels -20 -15 -10 -5 0 5 10 20
```

**Output:**
- `data/processed/snr_duration_crossed/*.wav` (640 files)
- `data/processed/snr_duration_crossed/metadata.csv` (with `duration_ms` and `snr_db`)

### 2. Evaluate with Qwen2-Audio

Modify `evaluate_with_robust_metrics.py` or create new script:

```bash
python scripts/evaluate_snr_duration_crossed.py \
    --manifest data/processed/snr_duration_crossed/metadata.csv \
    --output_dir results/snr_duration_evaluation
```

**Output:**
- `results/snr_duration_evaluation/predictions.parquet` (640 predictions)

### 3. Fit Stratified Psychometric Curves

New script: `scripts/fit_stratified_snr_curves.py`

```python
# For each duration level:
for duration_ms in [20, 80, 200, 1000]:
    # Filter to this duration
    subset = predictions_df[predictions_df['duration_ms'] == duration_ms]

    # Fit MLE curve: P(correct) vs SNR
    params, fitted = fit_psychometric_mle(
        subset['snr_db'].values,
        subset['correct'].values,
        gamma=0.5
    )

    # Bootstrap CI for SNR-75
    snr75, ci_low, ci_high = bootstrap_threshold(
        subset['snr_db'].values,
        subset['correct'].values,
        subset['clip_id'].values,
        n_bootstrap=1000,
        threshold_type='dt75',
        gamma=0.5
    )

    # Store results per duration
    results[f'duration_{duration_ms}ms'] = {
        'snr75': snr75,
        'snr75_ci_lower': ci_low,
        'snr75_ci_upper': ci_high,
        'mcfadden_r2': params['mcfadden_r2'],
        'tjur_r2': params['tjur_r2'],
    }
```

**Output:**
- `results/snr_stratified/snr_curves_by_duration.json`
- `results/snr_stratified/snr_curve_20ms.png`
- `results/snr_stratified/snr_curve_80ms.png`
- `results/snr_stratified/snr_curve_200ms.png`
- `results/snr_stratified/snr_curve_1000ms.png`
- `results/snr_stratified/snr_curves_combined.png` (4 curves on same plot)

### 4. Optional: GLMM Analysis

For full analysis, fit mixed-effects logistic regression:

```r
# R code (or Python statsmodels)
library(lme4)

model <- glmer(
    correct ~ snr_db * log(duration_ms) + (1 | clip_id),
    data = predictions_df,
    family = binomial(link = "logit")
)

# Extract marginal effects
# SNR at duration=200ms:
new_data <- expand.grid(
    snr_db = seq(-20, 20, by=0.5),
    duration_ms = 200,
    clip_id = unique(predictions_df$clip_id)[1]  # Reference level
)
new_data$pred <- predict(model, newdata=new_data, type="response", re.form=NA)

# Plot marginal SNR curve at 200ms
plot(new_data$snr_db, new_data$pred)
```

**Output:**
- `results/snr_glmm/glmm_summary.txt` (model coefficients)
- `results/snr_glmm/marginal_snr_curves.png` (SNR curves at fixed durations)
- `results/snr_glmm/marginal_duration_curves.png` (Duration curves at fixed SNR)

## Acceptance Criteria

### ✅ Pass Criteria

1. **All stratified curves monotonic**
   - Visual inspection: no dips or reversals
   - Slope coefficient > 0 for all duration levels

2. **Improved pseudo-R²**
   - McFadden R² > 0.05 for **at least 2 duration levels**
   - Tjur R² > 0.05 for **at least 2 duration levels**
   - Better fit than "overall" curve (current: 0.018)

3. **SNR-75 thresholds robust**
   - CIs narrower than current [-10, 2.5]
   - CIs don't touch boundaries (-20 or +20 dB)
   - SNR-75 increases monotonically with shorter duration

4. **Expected pattern**
   - **Short duration (20ms):** Higher SNR-75 (harder to detect in noise)
   - **Long duration (1000ms):** Lower SNR-75 (easier to detect in noise)
   - **Interaction:** Duration helps more at low SNR

### ⚠️ Warning Signs

- Any stratified curve with R² < 0.02 (still poor fit)
- Wide CIs that span >15 dB range
- Non-monotonic pattern persists even after stratification

If these occur, investigate:
1. SNR generation accuracy (Sprint 5 validation)
2. Clip-specific effects (some clips may be inherently harder)
3. Model artifacts (Qwen2-Audio may have non-linear SNR response)

## Timeline Estimate

- **Dataset generation:** 1 hour (20 clips) to 3 hours (70 clips)
- **Evaluation:** 30 min (20 clips) to 2 hours (70 clips) with GPU
- **Curve fitting:** 10 min (stratified) + 20 min (GLMM if used)
- **Figures & documentation:** 30 min

**Total:** ~2-6 hours depending on dataset size

## Expected Outcomes

### Scenario A: Success (Most Likely)

**Stratified curves are monotonic** with McFadden R² = 0.10-0.20:
- SNR-75 at 20ms: -5 dB [CI: -8, -2]
- SNR-75 at 80ms: -8 dB [CI: -11, -5]
- SNR-75 at 200ms: -12 dB [CI: -15, -9]
- SNR-75 at 1000ms: -15 dB [CI: -18, -12]

**Interpretation:** Duration mixing was the issue. Model performs better with longer stimuli at low SNR, as expected.

**Next steps:** Extend to full dev set (70 clips), report in paper.

### Scenario B: Partial Success

**Most curves monotonic, but one duration level problematic:**
- 20ms curve: Non-monotonic (R² = 0.01)
- 80ms, 200ms, 1000ms: Monotonic (R² = 0.08-0.15)

**Interpretation:** Very short durations may have ceiling/floor effects or insufficient dynamic range.

**Next steps:** Report 3 successful curves, note limitation for 20ms.

### Scenario C: Failure (Unlikely)

**Non-monotonic pattern persists** across multiple duration levels:
- McFadden R² < 0.03 for all stratified curves
- Dips at 0dB and +10dB remain

**Interpretation:** Fundamental issue with SNR generation or model artifacts.

**Next steps:** Sprint 9 technical validation:
1. Measure actual SNR ratio in generated audio (FFT analysis)
2. Verify noise mixing algorithm
3. Test with simpler model (logistic regression on acoustic features)

## References

1. **Wichmann & Hill (2001):** Bootstrap CIs for psychometric functions
2. **McFadden (1974):** Pseudo-R² for maximum likelihood models
3. **Bates et al. (2015):** Fitting linear mixed-effects models using lme4

## Files to Create/Modify

**New files:**
- `scripts/generate_snr_duration_crossed.py` ← Already created
- `scripts/evaluate_snr_duration_crossed.py`
- `scripts/fit_stratified_snr_curves.py`
- `data/processed/subset_20clips_for_crossed.csv` ← Already created
- `SPRINT8_SPECIFICATION.md` ← This document

**Modified files:**
- `scripts/fit_psychometric_curves.py` (optional: add stratification support)
- `README.md` (update with Sprint 8 status)

**Generated outputs (gitignored):**
- `data/processed/snr_duration_crossed/*.wav`
- `results/snr_duration_evaluation/predictions.parquet`
- `results/snr_stratified/*.json`
- `results/snr_stratified/*.png`
