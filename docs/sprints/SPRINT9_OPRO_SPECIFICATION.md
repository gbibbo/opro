# Sprint 9: OPRO Prompt Optimization

**Status**: 🚧 IMPLEMENTATION COMPLETE - Ready for execution
**Date**: 2025-10-13

---

## Objective

Optimize prompts using **OPRO (Optimization by PROmpting)** to improve:
1. **Primary**: Clip-level Balanced Accuracy (BA_clip > 0.69)
2. **Secondary**: Condition-averaged Balanced Accuracy (harder conditions)
3. **Tertiary**: Psychometric thresholds (DT75 lower, SNR-75 lower)

**Constraint**: No model weight modifications (zero-shot prompt optimization only)

---

## Baseline Performance (v1.0-baseline-final)

| Metric | Dev Set | Test Set | Status |
|--------|---------|----------|--------|
| **BA_clip** | 0.690 | 0.353 | Baseline |
| **DT75** | 34.8 ms [19.9, 64.1] | 1000 ms* | Baseline |
| **SNR-75 (1000ms)** | -2.9 dB [-12.0, +8.5] | 20 dB* | Baseline |

\* *Test set substantially harder (smaller N, tougher clips)*

**Baseline Prompt**:
```
<|audio_bos|><|AUDIO|><|audio_eos|>Does this audio contain human speech?
Reply with ONLY one word: SPEECH or NON-SPEECH.
```

---

## OPRO Algorithm

### Core Loop

```
For iteration = 1 to N_max:
    1. Build meta-prompt with top-k best (prompt, reward) pairs
    2. LLM generates M new candidate prompts
    3. Evaluate each candidate on dev set → compute reward
    4. Update top-k memory
    5. Check convergence (early stopping if no improvement)
```

### Reward Function

```
R = BA_clip + α × BA_conditions - β × len(prompt)/100
```

**Weights**:
- `α = 0.25`: Weight for condition-averaged BA
- `β = 0.05`: Penalty for prompt length (per 100 chars)

**Rationale**:
- Primary objective: BA_clip (most direct measure of performance)
- Secondary: BA_conditions (encourages robustness across hard conditions)
- Tertiary: Length penalty (favors conciseness, reduces inference overhead)

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer LLM** | Claude 3.5 Sonnet | Generates prompt candidates |
| **Evaluator Model** | Qwen2-Audio-7B (4-bit) | Frozen baseline model |
| **Top-k Memory** | 10 | Keep best 10 prompts |
| **Candidates/Iteration** | 3-5 | Balance exploration vs compute |
| **Max Iterations** | 30-50 | Typical OPRO convergence range |
| **Early Stopping** | 5 iterations | Stop if no improvement |
| **Temperature** | 0.7 | LLM generation (exploration) |
| **Eval Temperature** | 0.0 | Deterministic evaluation |
| **Seed** | 42 | Reproducibility |

---

## Meta-Prompt Template

```
TASK: Optimize prompts for audio classification (Qwen2-Audio-7B-Instruct).
The model receives audio and must classify it as SPEECH or NON-SPEECH.

OBJECTIVE: Maximize balanced accuracy, especially on hard conditions:
- Short durations (20-200ms)
- Low SNR (-10 to 0 dB)
- Band-pass filtered audio
- Reverberant audio

REWARD FUNCTION:
R = BA_clip + 0.25 × BA_conditions - 0.05 × len(prompt)/100

BASELINE PROMPT:
Prompt: {baseline_prompt}
Reward: {baseline_reward}

TOP-{k} PROMPTS:
1. Reward={r1} | BA_clip={ba1} | BA_cond={bc1}
   Prompt: {prompt1}
2. ...

INSTRUCTIONS:
Generate {N} NEW prompt candidates that:
1. Are clear and concise (target <150 chars)
2. Encourage robust detection on SHORT and NOISY clips
3. Use simple, direct language
4. Build on insights from top prompts
5. Explore semantic variations (question, command, description styles)

OUTPUT FORMAT:
PROMPT_1: <your prompt here>
PROMPT_2: <your prompt here>
...
```

---

## Implementation Structure

### Scripts

1. **`opro_optimizer.py`**: Core OPRO engine
   - `OPROOptimizer` class
   - Meta-prompt generation
   - Top-k memory management
   - LLM API integration (Claude/OpenAI)

2. **`evaluate_prompt.py`**: Single-prompt evaluator
   - Evaluates prompt on dev/test split
   - Returns (BA_clip, BA_conditions, full_metrics)
   - Reuses Sprint 6 evaluation infrastructure

3. **`run_opro.py`**: Main runner
   - Integrates optimizer + evaluator
   - Loads model once (reuses for all evals)
   - Runs full optimization loop

4. **`refit_psychometric_opro.py`**: Refit curves with best prompt
   - Refits duration curves (DT75)
   - Refits SNR curves stratified by duration (SNR-75)
   - Compares to baseline

5. **`evaluate_opro_test.py`**: ONE-TIME test evaluation
   - Evaluates best prompt on hold-out test set
   - Prevents test set leakage
   - Confirms improvement holds on unseen data

### File Structure

```
results/
└── sprint9_opro/
    ├── opro_prompts.jsonl          # Full iteration history
    ├── opro_memory.json             # Top-k best prompts
    ├── opro_history.json            # Reward curve
    ├── best_prompt.txt              # Final best prompt
    ├── best_metrics.json            # Best prompt metrics
    ├── dev_predictions.parquet      # Dev set predictions
    ├── dev_clips.parquet            # Clip-level aggregation
    ├── dev_metrics.json             # Dev set metrics
    ├── psychometric_opro/           # Refitted curves
    │   ├── duration_curve.png
    │   ├── psychometric_results.json
    │   └── baseline_vs_opro.json
    ├── test_set_opro/               # ONE-TIME test eval
    │   ├── test_predictions.parquet
    │   ├── test_results.json
    │   └── baseline_vs_opro_test.json
    └── comparison_report.md         # Final comparison report
```

---

## Usage

### 1. Initial Test (5 iterations)

```bash
# Validate pipeline with short run
python scripts/run_opro.py \
    --n_iterations 5 \
    --candidates_per_iter 3 \
    --output_dir results/sprint9_opro_test \
    --api_key $ANTHROPIC_API_KEY
```

**Expected time**: ~2-3 hours (5 iter × 3 candidates × ~30 min eval)

### 2. Full Optimization (30-50 iterations)

```bash
# Production run
python scripts/run_opro.py \
    --optimizer_llm claude-3-5-sonnet-20241022 \
    --n_iterations 50 \
    --candidates_per_iter 3 \
    --top_k 10 \
    --early_stopping 5 \
    --output_dir results/sprint9_opro \
    --api_key $ANTHROPIC_API_KEY
```

**Expected time**: ~20-30 hours (50 iter × 3 candidates × ~30 min eval)
**Early stopping**: May converge earlier (~20-30 iterations typical)

### 3. Refit Psychometric Curves

```bash
# Refit duration + SNR curves with best prompt
python scripts/refit_psychometric_opro.py \
    --opro_dir results/sprint9_opro \
    --baseline_results results/psychometric_curves/psychometric_results.json \
    --n_bootstrap 1000
```

### 4. Test Set Evaluation (ONE TIME ONLY)

```bash
# Evaluate on hold-out test set
python scripts/evaluate_opro_test.py \
    --opro_dir results/sprint9_opro \
    --baseline_test results/test_set_final \
    --n_bootstrap 1000
```

⚠️ **WARNING**: Run this ONLY ONCE per optimization. Test set is hold-out data.

---

## Success Criteria

### Primary (Must Achieve)

- ✅ **BA_clip improvement**: Δ BA_clip > +0.03 (e.g., 0.69 → 0.72)
- ✅ **No test set leakage**: Test eval performed ONCE only
- ✅ **Reproducibility**: Full logs + seeds enable exact replication

### Secondary (Desired)

- ✅ **DT75 improvement**: Lower threshold (e.g., 35ms → 25-30ms)
- ✅ **SNR-75 improvement**: More noise tolerance (e.g., -3dB → -5dB at 1000ms)
- ✅ **Condition robustness**: Improved BA on hard conditions (20ms, -10dB)

### Tertiary (Bonus)

- ✅ **Prompt interpretability**: Best prompt is human-understandable
- ✅ **Generalization**: Test set performance close to dev set

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **OPRO doesn't converge** | Start with 5-iter test, adjust reward weights |
| **Test set overfitting** | Strict ONE-TIME eval policy, no prompt tuning on test |
| **LLM API costs** | Estimate: 50 iter × 3 cand × $0.15/eval = ~$22.50 |
| **Long runtime** | Use early stopping, parallel evals if possible |
| **Model loading overhead** | Load model once, reuse for all evals in run_opro.py |

---

## References

1. **Yang et al. (2023)**. "Large Language Models as Optimizers." *arXiv:2309.03409*
   - Original OPRO paper
   - Meta-prompting approach
   - Prompt optimization results (GSM8K, Big-Bench Hard)

2. **Wichmann & Hill (2001a, 2001b)**. "The psychometric function I & II." *Perception & Psychophysics*
   - MLE fitting methodology
   - Bootstrap confidence intervals

3. **McFadden (1974)** & **Tjur (2009)**. Pseudo-R² for logistic models
   - Goodness-of-fit metrics

4. **Moscatelli et al. (2012)**. "Modeling psychophysical data at the population-level." *Journal of Vision*
   - GLMM for repeated measures

---

## Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Implementation** | ✅ Complete | All scripts + infrastructure |
| **Initial Test** | ~3 hours | 5-iteration validation |
| **Full Optimization** | ~20-30 hours | 30-50 iterations with early stopping |
| **Psychometric Refit** | ~2 hours | Refit curves + comparison |
| **Test Evaluation** | ~1 hour | ONE-TIME test set eval |
| **Documentation** | ~2 hours | Final report + git tag |

**Total**: ~2-3 days (mostly compute time)

---

## Next Steps

1. ✅ Implementation complete (all scripts created)
2. ⏭️ Run initial test (5 iterations)
3. ⏭️ Launch full optimization (30-50 iterations)
4. ⏭️ Refit psychometric curves
5. ⏭️ ONE-TIME test evaluation
6. ⏭️ Create final comparison report
7. ⏭️ Git tag: `v2.0-opro-baseline`

---

**Status**: Ready for execution
**Implementation**: Complete
**Estimated Total Time**: 2-3 days (compute-bound)
