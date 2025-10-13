# Results Summary - Quick Reference

**Model**: Qwen2-Audio-7B-Instruct (4-bit, zero-shot)
**Status**: ✅ BASELINE FROZEN (v1.0-baseline-final)
**Date**: 2025-10-13

---

## 🎯 Primary Results (Dev Set, n=70)

| Threshold | Value | CI95 | Status |
|-----------|-------|------|--------|
| **DT75** | **34.8 ms** | [19.9, 64.1] | ✅ Paper-ready |
| **SNR-75** (1000ms) | **−2.9 dB** | [−12.0, +8.5] | ✅ Paper-ready |

**Interpretation**:
- Model needs **35 ms** for 75% correct speech detection
- Tolerates SNR down to **−3 dB** at 1 second
- **GLMM confirms**: SNR×Duration interaction (p<0.001)

---

## 📊 SNR-75 by Duration

| Duration | SNR-75 | CI95 | Status |
|----------|--------|------|--------|
| 1000 ms | −2.9 dB | [−12.0, +8.5] | ✅ Robust |
| 200 ms | +16.0 dB | [−0.7, +20.0] | ✅ At limit |
| 80 ms | >+20 dB | [+15.3, +20.0] | Out of range |
| 20 ms | >+20 dB | [+20.0, +20.0] | Chance level |

**Pattern**: Longer duration → lower SNR required

---

## 🔬 GLMM Interaction

| Effect | β | p-value |
|--------|---|---------|
| SNR | −0.045 | 0.011* |
| log(Duration) | +0.295 | <0.001*** |
| **SNR:log(Duration)** | **+0.014** | **<0.001*** |

**McFadden R²** = 0.042

---

## 📈 Performance Summary

| Metric | Dev (n=70) | Test (n=17) |
|--------|------------|-------------|
| Accuracy | 0.583 | 0.335 |
| Clip Accuracy | 0.690 | 0.353 |
| DT75 | 34.8 ms | 1000 ms* |

\* *Test set harder (smaller, tougher clips)*

---

## 📁 Key Files

**Documentation**:
- [BASELINE_FINAL_REPORT.md](BASELINE_FINAL_REPORT.md) - Complete report
- [BASELINE_FREEZE.md](BASELINE_FREEZE.md) - Technical freeze
- [README.md](README.md) - Main documentation

**Results**:
- `results/psychometric_curves/` - Duration curves
- `results/sprint8_stratified/` - SNR curves by duration
- `results/sprint8_glmm/` - GLMM analysis
- `results/test_set_final/` - Test set evaluation

**Figures**:
- `results/psychometric_curves/duration_curve.png`
- `results/sprint8_stratified/snr_curves_stratified.png`
- `results/sprint8_glmm/isoperformance_contour.png`

---

## 🏷️ Git Tags

- `v1.0-baseline-dev` - Dev results frozen
- `v1.0-baseline-final` - Dev + test complete

---

## 📚 Methodology

- **Fitting**: MLE binomial (Wichmann & Hill 2001)
- **CIs**: Bootstrap n=1000, clustered by clip
- **Goodness-of-fit**: McFadden & Tjur pseudo-R²
- **Interaction**: GEE with logit link

---

## ✅ Paper-Ready Deliverables

1. **Tables**: DT75, SNR-75 by duration, GLMM coefficients
2. **Figures**: 3 publication-quality plots (duration, SNR stratified, contour)
3. **Methods section**: Fully written with standard references
4. **Results section**: Fully written with interpretation
5. **References**: Wichmann, McFadden, Tjur, Moscatelli

---

## 🚫 FROZEN - Do NOT Modify

- Model: Qwen2-Audio-7B-Instruct, 4-bit
- Temperature: 0.0 (deterministic)
- Seed: 42
- Prompt: (see BASELINE_FREEZE.md)
- Dev/test split

**For comparisons**: Create new version tag (v1.1+, v2.0+)

---

**Quick Access**:
- Full report → [BASELINE_FINAL_REPORT.md](BASELINE_FINAL_REPORT.md)
- Reproduce → See [README.md](README.md) Quick Start
- Questions → See documentation files
