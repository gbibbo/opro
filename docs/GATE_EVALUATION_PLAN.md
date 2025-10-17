# Gate Evaluation - Fine-Tuning Readiness Check

## Overview

This document describes Step 1 of the fine-tuning preparation: the gate evaluation to verify if we're ready to proceed with fine-tuning.

## Objective

Perform a comprehensive evaluation of the best prompt from optimization to determine if:
1. The prompt improvement is substantial and stable
2. Constrained decoding works properly
3. Performance is robust across conditions
4. We've reached the ceiling of prompt engineering (ready for fine-tuning)

## Gate Criteria

The evaluation checks 4 key criteria:

### 1. BA_clip improvement >= +2%
- **Threshold**: Optimized prompt must improve baseline by at least 2 percentage points
- **Rationale**: Smaller improvements may not justify proceeding to fine-tuning

### 2. Constrained decoding delta >= -1%
- **Threshold**: Constrained decoding should not hurt performance by more than 1%
- **Rationale**: Ensures forcing SPEECH/NONSPEECH tokens doesn't degrade quality

### 3. Error rate < 5%
- **Threshold**: Less than 5% of predictions return UNKNOWN or fail
- **Rationale**: Model should reliably output valid labels

### 4. Monotonic duration relationship
- **Threshold**: Accuracy should generally increase with duration (>=70% of pairs)
- **Rationale**: Confirms model has learned the fundamental psychoacoustic relationship

## Evaluation Protocol

### Dataset
- **Source**: `data/processed/snr_duration_crossed/metadata.csv`
- **Size**: 640 samples (20 clips × 32 conditions)
- **Conditions**: Factorial design (5 durations × 6 SNRs + baseline)
- **Split**: Stratified by variant_type and label

### Prompts Tested

1. **Baseline**: `"Is this audio clip SPEECH or NON-SPEECH?"`
   - Simple, direct question format
   - From initial optimization iteration

2. **Optimized**: `"Based on the audio file, is it SPEECH or NON-SPEECH?"`
   - Best prompt from local optimizer
   - Achieved +11.54% improvement on subset (BA_clip: 0.8462 → 0.9615)

### Decoding Modes

1. **Unconstrained**: Standard greedy decoding
2. **Constrained**: LogitsProcessor forcing only SPEECH/NONSPEECH tokens

### Metrics Computed

#### Primary Metrics
- **BA_clip**: Balanced accuracy at clip level (majority voting across segments)
- **Confusion Matrix**: True positive/negative rates
- **Error Rate**: Proportion of UNKNOWN predictions

#### Psychometric Analysis
- **DT75**: Duration threshold for 75% accuracy (ms)
- **SNR-75**: SNR threshold for 75% accuracy (dB) by duration bin
- **Monotonicity**: Correlation between duration and accuracy

#### Condition-Level Breakdown
- BA_clip by variant_type (duration, SNR, band, RIR)
- Performance across duration bins
- Performance across SNR bins

## Expected Outcomes

### If Gate PASSES

**Interpretation**: Prompt optimization has reached its ceiling. Further improvements require parameter adaptation.

**Next Steps**:
1. Prepare fine-tuning dataset with train/val/test splits
2. Set up LoRA/QLoRA configuration:
   - `r=8-16, alpha=16-32`
   - Target modules: `q_proj,k_proj,v_proj,o_proj`
   - Freeze audio encoder
3. Train with constrained decoding
4. Re-run psychometric evaluation on test set
5. Compare DT75 and SNR-75 thresholds

**Expected FT Improvements**:
- DT75: 35ms → ~25-30ms (better short duration detection)
- SNR-75: -15dB → ~-18dB (better noise robustness)
- BA_clip: +3-5% absolute gain

### If Gate FAILS

**Interpretation**: Prompt optimization has not exhausted all gains. More prompt engineering needed.

**Next Steps**:
1. Analyze failure modes by condition
2. Run more optimization iterations (increase search budget)
3. Explore canonical templates:
   - Multiple choice (A/B/C/D)
   - Chain-of-thought
   - Few-shot with examples
   - Contextual calibration
4. Test systematic prompt variations
5. Re-run gate evaluation

## Implementation

### Scripts

1. **`scripts/gate_evaluation.py`**: Full evaluation on 640 samples
   - ~20-30 minutes runtime
   - Generates comprehensive report with psychometric curves

2. **`scripts/gate_evaluation_quick.py`**: Quick test on 50 samples
   - ~2-3 minutes runtime
   - Useful for debugging and verification

### Output Files

All results saved to `results/gate_evaluation/`:

- `baseline_predictions.csv`: Per-segment predictions for baseline prompt
- `optimized_predictions.csv`: Per-segment predictions for optimized prompt (unconstrained)
- `optimized_constrained_predictions.csv`: Per-segment predictions with constrained decoding
- `gate_report.txt`: Summary report with gate decision

### Report Format

```
================================================================================
GATE EVALUATION REPORT - FINE-TUNING READINESS
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Baseline BA_clip:               0.XXXX
Optimized BA_clip:              0.XXXX
Optimized + Constrained BA_clip: 0.XXXX

Improvement over baseline:      +X.XX%
Constrained decoding impact:    +/-X.XX%

GATE CRITERIA
--------------------------------------------------------------------------------
[PASS/FAIL] 1. BA_clip improvement >= +2%: +X.XX%
[PASS/FAIL] 2. Constrained decoding delta >= -1%: +/-X.XX%
[PASS/FAIL] 3. Error rate < 5%: X.XX%
[PASS/FAIL] 4. Monotonic duration relationship: XX%

================================================================================
GATE DECISION: PASS - PROCEED TO FINE-TUNING
================================================================================

PSYCHOMETRIC THRESHOLDS
--------------------------------------------------------------------------------
Baseline DT75:   XX.X ms
Optimized DT75:  XX.X ms

PERFORMANCE BY VARIANT TYPE
--------------------------------------------------------------------------------
Variant         Baseline    Optimized   Delta
--------------------------------------------------------------------------------
duration        0.XXXX      0.XXXX      +/-X.XX%
snr             0.XXXX      0.XXXX      +/-X.XX%
...
```

## Technical Details

### Constrained Decoding Implementation

Uses `ConstrainedVocabLogitsProcessor` to force model output to only SPEECH/NONSPEECH tokens:

```python
class ConstrainedVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.ones_like(scores, dtype=torch.bool)
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = False
        return scores.masked_fill(mask, float("-inf"))
```

### Clip-Level BA Computation

Majority voting across segments from the same clip:

```python
def compute_clip_level_ba(results_df):
    for clip_id in results_df['clip_id'].unique():
        clip_rows = results_df[results_df['clip_id'] == clip_id]

        # Majority vote
        speech_votes = (clip_rows['y_pred'] == 1).sum()
        nonspeech_votes = (clip_rows['y_pred'] == 0).sum()

        y_pred_clip = 1 if speech_votes > nonspeech_votes else 0
        ...

    return balanced_accuracy_score(y_true, y_pred)
```

### Psychometric Curve Fitting

Linear interpolation to estimate 75% accuracy threshold:

```python
threshold_75 = np.interp(0.75, accuracies, condition_values)
```

Bootstrap confidence intervals (1000 samples) for error bars.

## References

### Related Sprints

- **Sprint 6**: Baseline robust evaluation
- **Sprint 7**: Duration psychometric curves
- **Sprint 8**: SNR × duration factorial design
- **Sprint 9**: Local prompt optimization

### Key Results So Far

| Metric | Baseline | After Optimization |
|--------|----------|-------------------|
| BA_clip (subset) | 0.8462 | 0.9615 (+11.54%) |
| Best Prompt | "Is this audio clip SPEECH or NON-SPEECH?" | "Based on the audio file, is it SPEECH or NON-SPEECH?" |

### Literature Support

- **LoRA/QLoRA**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Qwen2-Audio**: [arXiv:2407.10759](https://arxiv.org/abs/2407.10759)
- **Fine-tuning Audio LLMs**: LLaMA-Factory supports Qwen2-Audio with PEFT

## Timeline

- **Gate evaluation**: ~30 minutes (640 samples)
- **Report generation**: ~1 minute
- **Decision point**: Immediate after gate report
- **FT preparation** (if pass): 1-2 hours
- **FT training** (if pass): 2-4 hours (1-3 epochs)
- **FT evaluation** (if pass): ~30 minutes (full psychometric suite)

Total time from gate start to FT results: **~5-8 hours**

## Success Metrics for Fine-Tuning

If gate passes and we proceed to FT, success criteria:

1. **DT75 improvement**: At least 5ms reduction (35ms → 30ms)
2. **SNR-75 improvement**: At least 2dB reduction at long durations
3. **BA_clip gain**: +3-5% absolute over optimized prompt
4. **Generalization**: Test set BA within 2% of dev set BA
5. **Monotonicity preserved**: Duration and SNR relationships maintained

## Risk Mitigation

### Potential Issues

1. **Overfitting**: FT may overfit to dev set conditions
   - Mitigation: Hold-out test set, early stopping, dropout=0.1

2. **Catastrophic forgetting**: Model may lose general audio understanding
   - Mitigation: Freeze audio encoder, use LoRA (not full FT)

3. **Data imbalance**: 50/50 SPEECH/NONSPEECH may not match deployment
   - Mitigation: Test on realistic class distributions

4. **Computational cost**: Full FT may exceed 8GB VRAM
   - Mitigation: QLoRA 4-bit, gradient accumulation, batch=1

## Conclusion

The gate evaluation provides a rigorous, data-driven decision point for proceeding to fine-tuning. By requiring:
- Substantial improvement (+2%)
- Robust constrained decoding
- Low error rates
- Psychophysical validity (monotonic duration effect)

We ensure that fine-tuning investment is justified and likely to yield meaningful gains beyond prompt engineering alone.
