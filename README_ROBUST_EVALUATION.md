# Robust Statistical Evaluation Methodology

This document explains the statistical methodology for robustly evaluating and comparing speech detection models.

## Table of Contents

1. [Why Robust Evaluation Matters](#why-robust-evaluation-matters)
2. [The Problem with Small Test Sets](#the-problem-with-small-test-sets)
3. [Methodology](#methodology)
4. [Tools and Usage](#tools-and-usage)
5. [Interpreting Results](#interpreting-results)
6. [References](#references)

---

## Why Robust Evaluation Matters

### The Original Problem

With the initial test set of **32 samples**, we observed:
- Multi-seed training (5 seeds) gave **identical results**: 96.9% accuracy
- Standard deviation: **0.0%**
- Impossible to distinguish between models statistically

### What Went Wrong?

With only 32 samples:
- **1 error = 3.1% accuracy change** (32 samples)
- **1 error = 6.25% class accuracy change** (16 samples per class)
- Models scoring 93.8% vs 96.9% might be statistically **equivalent**
- Cannot detect real improvements < 3-5%

### The Solution

1. **Larger test set** (150-200 samples) for statistical power
2. **Bootstrap confidence intervals** for robust uncertainty estimation
3. **McNemar's test** for paired model comparison
4. **Logit-based evaluation** (faster, more stable than generation)
5. **Temperature scaling** for calibrated confidence scores

---

## The Problem with Small Test Sets

### Statistical Power Analysis

For binary classification with accuracy ~95%:

| Test Size | 1 Error Impact | Min Detectable Δ (p<0.05) |
|-----------|----------------|---------------------------|
| 32        | 3.1%           | ±8-10%                    |
| 64        | 1.6%           | ±5-7%                     |
| 100       | 1.0%           | ±3-5%                     |
| 200       | 0.5%           | ±2-3%                     |

**Conclusion**: To detect **2-3% accuracy differences** with p<0.05, you need **≥150-200 samples**.

### Why Bootstrap CI?

Standard confidence intervals assume:
- Normal distribution (not true for accuracy on small samples)
- Large sample size (n>30 is rough guideline)

Bootstrap CI:
- **Non-parametric** (no distribution assumptions)
- **Robust** to skewed distributions
- **Accurate** even for small samples (n=32)
- **Reports uncertainty** properly

---

## Methodology

### 1. Extended Test Set Generation

**Script**: `scripts/generate_extended_test_set.py`

Generates 150-200 balanced test samples with **factorial design**:

- **Durations**: 200ms, 400ms, 600ms, 800ms, 1000ms
- **SNRs**: 0dB, 5dB, 10dB, 15dB, 20dB
- **Classes**: SPEECH (50%), NONSPEECH (50%)
- **Sources**: VoxConverse (speech), MUSAN/ESC-50 (nonspeech)

**Why factorial?**
- Ensures **balanced coverage** of all conditions
- Prevents **confounding** (e.g., all difficult samples being low SNR AND short)
- Allows **stratified analysis** (performance by duration, SNR, etc.)

### 2. Logit-Based Evaluation

**Script**: `scripts/test_with_logit_scoring.py`

**Why logits instead of generate()?**

| Method | Speed | Stability | Calibration |
|--------|-------|-----------|-------------|
| `generate()` | Slow | Varies | Poor |
| Logit scoring | **5-10× faster** | **Deterministic** | **Better** |

**How it works:**
1. Forward pass with audio + prompt
2. Extract logits for tokens 'A' and 'B' at output position
3. Apply softmax: P(A), P(B) = softmax([logit_A, logit_B])
4. Predict: argmax(P(A), P(B))

**Benefits:**
- **Deterministic**: Same input → same output
- **Fast**: No iterative decoding
- **Calibrated**: Direct probability estimation
- **Temperature scaling ready**: Can apply T>1 for better calibration

### 3. Bootstrap Confidence Intervals

**Implementation**: `run_robust_evaluation.py`

**Algorithm:**
1. For B=10,000 bootstrap iterations:
   - Resample test set **with replacement** (n samples → n samples)
   - Compute accuracy on bootstrap sample
2. CI = [percentile(2.5%), percentile(97.5%)]

**Example output:**
```
Model: attention_only
Accuracy: 96.9% [95% CI: 94.2%, 98.7%]
```

This means:
- Best estimate: 96.9%
- **95% confident** true accuracy is between 94.2% and 98.7%
- CI width = 4.5% (reflects test set size)

### 4. McNemar's Test for Paired Comparison

**Script**: `scripts/compare_models_mcnemar.py`

**Why McNemar, not independent t-test?**

Because:
- Models are evaluated on the **same test set** (paired data)
- McNemar uses **disagree** ments only → more powerful
- Accounts for correlation between model predictions

**Contingency table:**

|                | Model B Correct | Model B Wrong |
|----------------|----------------|---------------|
| **Model A Correct** | both_correct   | a_right_b_wrong |
| **Model A Wrong**   | a_wrong_b_right | both_wrong    |

**Test statistic:**
```
χ² = (|b - c| - 1)² / (b + c)
```
Where:
- b = a_right_b_wrong
- c = a_wrong_b_right
- Continuity correction: -1 (recommended for small b+c)

**Interpretation:**
- **p < 0.05**: Statistically significant difference
- **p ≥ 0.05**: Cannot conclude models differ

**Example:**
```
Model A: 96.9% (31/32)
Model B: 93.8% (30/32)

Contingency:
  Both correct: 30
  A right, B wrong: 1
  A wrong, B right: 0
  Both wrong: 1

χ² = (|1 - 0| - 1)² / (1 + 0) = 0.0
p-value = 1.0

→ No significant difference (only 1 disagreement)
```

### 5. Temperature Scaling (Optional)

**Purpose**: Calibrate confidence scores

Neural networks often produce **overconfident** predictions:
- Says 99% confidence but only 90% accurate
- Temperature scaling fixes this **post-hoc**

**Algorithm:**
1. Forward pass → logits
2. Scale logits: `logits_scaled = logits / T`
3. Softmax → calibrated probabilities

**Finding optimal T:**
- Use **dev set** (or cross-validation on train)
- Minimize **negative log-likelihood** or **ECE** (Expected Calibration Error)
- Typical values: T ∈ [1.0, 3.0]

**References:**
- Guo et al. (2017). "On Calibration of Modern Neural Networks" [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

---

## Tools and Usage

### Quick Start: Compare Two Models

```bash
# 1. Evaluate Model A (attention-only, seed 42)
python scripts/test_with_logit_scoring.py \
    --checkpoint checkpoints/qwen2_audio_speech_detection_multiseed/seed_42/final \
    --test_csv data/processed/normalized_clips/test_metadata.csv \
    --save_predictions results/attention_only_seed42.json

# 2. Evaluate Model B (MLP, seed 42)
python scripts/test_with_logit_scoring.py \
    --checkpoint checkpoints/with_mlp_seed_42/final \
    --test_csv data/processed/normalized_clips/test_metadata.csv \
    --save_predictions results/mlp_seed42.json

# 3. Compare with McNemar's test
python scripts/compare_models_mcnemar.py \
    --model_a results/attention_only_seed42.json \
    --model_b results/mlp_seed42.json \
    --name_a "Attention-only (seed 42)" \
    --name_b "Attention+MLP (seed 42)"
```

### Full Pipeline: Robust Evaluation

```bash
# Compare multiple models with bootstrap CI + McNemar
python scripts/run_robust_evaluation.py \
    --models attention_only mlp baseline \
    --checkpoints \
        checkpoints/qwen2_audio_speech_detection_multiseed/seed_42/final \
        checkpoints/with_mlp_seed_42/final \
        Qwen/Qwen2-Audio-7B-Instruct \
    --test_csv data/processed/normalized_clips/test_metadata.csv \
    --results_dir results/robust_evaluation
```

**Output:**
- `results/robust_evaluation/attention_only_predictions.json`
- `results/robust_evaluation/mlp_predictions.json`
- `results/robust_evaluation/baseline_predictions.json`
- `results/robust_evaluation/evaluation_report.txt` ← **Main report**
- `results/robust_evaluation/consolidated_results.json`

### Generate Extended Test Set (150-200 samples)

```bash
# 1. Generate metadata (factorial design)
python scripts/generate_extended_test_set.py \
    --voxconverse_csv data/raw/voxconverse_metadata.csv \
    --musan_csv data/raw/musan_metadata.csv \
    --output_dir data/processed/extended_test_clips \
    --n_samples_per_class 100  # Total: 200 samples

# 2. Process audio clips (extract, add noise, normalize)
python scripts/process_extended_test_clips.py

# 3. Re-run evaluation with larger test set
python scripts/run_robust_evaluation.py \
    --models attention_only mlp \
    --test_csv data/processed/extended_test_clips/extended_test_metadata.csv \
    --results_dir results/extended_evaluation
```

---

## Interpreting Results

### Example Report

```
================================================================================
ROBUST STATISTICAL EVALUATION REPORT
================================================================================

Model Performance (with 95% Bootstrap CI):
--------------------------------------------------------------------------------
Model                          Accuracy        95% CI
--------------------------------------------------------------------------------
attention_only                 0.969 (96.9%)  [0.942, 0.987]
mlp                            0.969 (96.9%)  [0.942, 0.987]
baseline                       0.625 (62.5%)  [0.573, 0.677]
--------------------------------------------------------------------------------

Pairwise Statistical Comparisons (McNemar's Test):
--------------------------------------------------------------------------------
Model A vs Model B                      χ²         p-value    Significant?
--------------------------------------------------------------------------------
attention_only vs mlp                   0.0000     1.0000     No (p≥0.05)
attention_only vs baseline              28.1234    <0.0001    Yes (p<0.05)
mlp vs baseline                         28.1234    <0.0001    Yes (p<0.05)
--------------------------------------------------------------------------------

Recommendations:
--------------------------------------------------------------------------------
✓ Tie between: attention_only, mlp
  No statistically significant difference between top models

Confidence Interval Overlaps:
  ≈ attention_only and mlp: CIs overlap (models may be equivalent)
================================================================================
```

### Key Insights

1. **CIs overlap** → Models are statistically **equivalent**
   - attention_only: [94.2%, 98.7%]
   - mlp: [94.2%, 98.7%]
   - Conclusion: Use **simpler model** (attention-only, 2× smaller)

2. **p-value < 0.05** → Model A **significantly better** than Model B
   - attention_only vs baseline: p < 0.0001
   - Conclusion: Fine-tuning **works**!

3. **p-value ≥ 0.05** → **Cannot conclude** models differ
   - attention_only vs mlp: p = 1.0
   - Conclusion: MLP targets **don't help** (for this dataset/config)

---

## Workflow: From Training to Publication

### Phase 1: Model Development (You are here!)

✅ **Done:**
- Baseline evaluation (50% → 62.5% → 90.6% → 96.9%)
- Multi-seed training (variance assessment)
- LoRA configuration (attention-only, MLP targets)

### Phase 2: Robust Evaluation (Next step)

**TODO:**
1. Generate extended test set (200 samples)
2. Re-evaluate all models with logit scoring + bootstrap CI
3. McNemar tests for pairwise comparison
4. Select best model based on:
   - Accuracy + CI width
   - Statistical significance
   - Model size / inference speed

### Phase 3: Advanced Optimization (If needed)

If best model < 98% accuracy:
- **OPRO on fine-tuned model** (prompt optimization)
- **Hyperparameter grid search** (LoRA rank, alpha, learning rate)
- **Dataset scaling** (1-3k samples)

### Phase 4: Final Evaluation & Reporting

For paper/publication:
- Freeze best model
- Evaluate on **held-out test set** (never seen before)
- Report:
  - Accuracy ± 95% CI
  - Breakdown by duration, SNR (stratified analysis)
  - McNemar p-values vs baselines
  - Inference time, model size

**Table template for paper:**

| Model | Params | Accuracy | 95% CI | p-value vs Baseline |
|-------|--------|----------|--------|---------------------|
| Baseline (zero-shot) | 7.6B | 62.5% | [59.3%, 65.7%] | - |
| OPRO-optimized prompt | 7.6B | 68.4% | [65.2%, 71.5%] | 0.012 |
| LoRA FT (attention) | +20.7M | **96.9%** | [94.2%, 98.7%] | <0.001 |
| LoRA FT + OPRO | +20.7M | **98.1%** | [96.1%, 99.2%] | 0.034 |

---

## Best Practices

### ✅ DO

1. **Use paired tests** (McNemar) when comparing models on same test set
2. **Report confidence intervals** (bootstrap or exact binomial)
3. **Pre-register test set** (freeze before evaluation, never use for HP tuning)
4. **Use stratified sampling** (balance duration, SNR, classes)
5. **Use logit scoring** for deterministic evaluation
6. **Report all comparisons** (avoid cherry-picking)

### ❌ DON'T

1. **Don't use accuracy alone** without CI (misleading for small n)
2. **Don't use independent t-tests** for paired data (loses power)
3. **Don't tune on test set** (causes overfitting, inflated results)
4. **Don't compare models on different test sets** (confounded)
5. **Don't report p<0.05 without multiple comparison correction** (if testing many models)

---

## References

### Statistics

1. **McNemar's Test**:
   - McNemar, Q. (1947). "Note on the sampling error of the difference between correlated proportions or percentages". *Psychometrika*.
   - [Wikipedia](https://en.wikipedia.org/wiki/McNemar%27s_test)

2. **Bootstrap Confidence Intervals**:
   - Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife". *The Annals of Statistics*.
   - [Wikipedia](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

3. **Temperature Scaling**:
   - Guo, C., et al. (2017). "On Calibration of Modern Neural Networks". [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

### Machine Learning

4. **OPRO (Prompt Optimization)**:
   - Yang, C., et al. (2023). "Large Language Models as Optimizers". [arXiv:2309.03409](https://arxiv.org/abs/2309.03409)

5. **LoRA (Low-Rank Adaptation)**:
   - Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models". [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

6. **QLoRA (Quantized LoRA)**:
   - Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs". [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

### Audio Processing

7. **Qwen2-Audio**:
   - [HuggingFace Docs](https://huggingface.co/docs/transformers/model_doc/qwen2_audio)

8. **Speech Activity Detection**:
   - Lavechin, M., et al. (2020). "An open-source voice type classifier for child-centered daylong recordings". *Interspeech*.

---

**Last Updated**: 2025-10-20
**Status**: Evaluation infrastructure ready, awaiting extended test set generation
