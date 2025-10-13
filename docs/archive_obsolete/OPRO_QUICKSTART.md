# OPRO Quick Start Guide

**Sprint 9**: Prompt optimization using OPRO (Optimization by PROmpting)

---

## Prerequisites

1. **API Key**: You need either:
   - Anthropic API key (for Claude 3.5 Sonnet) - Recommended
   - OpenAI API key (for GPT-4)

2. **Environment Setup**:
```bash
export ANTHROPIC_API_KEY="your-key-here"
# OR
export OPENAI_API_KEY="your-key-here"
```

3. **Dependencies**:
```bash
pip install anthropic openai  # Add to existing requirements
```

---

## Quick Start: 5-Iteration Test

Validate the pipeline with a short run (2-3 hours):

```bash
python scripts/run_opro.py \
    --n_iterations 5 \
    --candidates_per_iter 3 \
    --output_dir results/sprint9_opro_test \
    --seed 42
```

**What happens**:
1. Loads Qwen2-Audio model (4-bit, once)
2. Evaluates baseline prompt on dev set
3. Runs 5 OPRO iterations:
   - Claude generates 3 new prompts per iteration
   - Each prompt evaluated on full dev set (70 clips × 20 variants = 1400 samples)
   - Top-10 prompts kept in memory
4. Saves results to `results/sprint9_opro_test/`

**Expected output**:
```
Iteration 1: Generated 3 candidates
  Evaluating candidate 1/3...
  Results: BA_clip=0.715, BA_cond=0.680, Reward=0.882
  ...
NEW BEST REWARD: 0.892 (+0.010)
```

---

## Full Optimization: 30-50 Iterations

Production run (20-30 hours, may converge earlier with early stopping):

```bash
python scripts/run_opro.py \
    --optimizer_llm claude-3-5-sonnet-20241022 \
    --n_iterations 50 \
    --candidates_per_iter 3 \
    --top_k 10 \
    --early_stopping 5 \
    --output_dir results/sprint9_opro \
    --seed 42
```

**Parameters**:
- `--optimizer_llm`: LLM for generating prompts (claude-* or gpt-4*)
- `--n_iterations`: Max iterations (50 typical, may stop earlier)
- `--candidates_per_iter`: Prompts to generate per iteration (3-5)
- `--top_k`: Keep best k prompts in memory (10)
- `--early_stopping`: Stop if no improvement for N iterations (5)

**Monitoring**:
- Watch `results/sprint9_opro/opro_history.json` for convergence
- Best prompt updated in `results/sprint9_opro/best_prompt.txt`

---

## Refit Psychometric Curves

After optimization completes, refit duration and SNR curves with the best prompt:

```bash
python scripts/refit_psychometric_opro.py \
    --opro_dir results/sprint9_opro \
    --baseline_results results/psychometric_curves/psychometric_results.json \
    --n_bootstrap 1000 \
    --seed 42
```

**Outputs**:
- `psychometric_opro/duration_curve.png`: Refitted duration curve
- `psychometric_opro/psychometric_results.json`: New DT75, SNR-75 thresholds
- `psychometric_opro/baseline_vs_opro.json`: Comparison to baseline
- `comparison_report.md`: Human-readable comparison report

**Example comparison**:
```
Duration DT75:
  Baseline: 34.8 ms
  OPRO: 28.3 ms
  Delta: -6.5 ms (-18.7%)
  Status: ✅ IMPROVED

SNR-75 (1000ms):
  Baseline: -2.9 dB
  OPRO: -4.2 dB
  Delta: -1.3 dB
  Status: ✅ IMPROVED
```

---

## Test Set Evaluation (ONE TIME ONLY)

⚠️ **CRITICAL**: Run this ONLY ONCE per optimization. Test set is hold-out data.

```bash
python scripts/evaluate_opro_test.py \
    --opro_dir results/sprint9_opro \
    --baseline_test results/test_set_final \
    --n_bootstrap 1000 \
    --seed 42
```

**Safety check**: Script will abort if test results already exist (use `--force` to override)

**Confirmation prompt**:
```
⚠️  WARNING: Test set evaluation is ONE-TIME ONLY!

Proceed with ONE-TIME test set evaluation? (yes/no): yes
```

**Outputs**:
- `test_set_opro/test_predictions.parquet`: Test predictions
- `test_set_opro/test_results.json`: Test metrics + curves
- `test_set_opro/baseline_vs_opro_test.json`: Test comparison

---

## Troubleshooting

### Issue: "API key not found"

**Solution**:
```bash
# Set environment variable
export ANTHROPIC_API_KEY="your-key"

# OR pass directly
python scripts/run_opro.py --api_key "your-key" ...
```

### Issue: "Out of memory during evaluation"

**Possible causes**:
- Model loaded multiple times
- Large batch size

**Solution**:
- `run_opro.py` loads model once and reuses it
- Use 4-bit quantization (default)
- Monitor GPU memory: `nvidia-smi -l 1`

### Issue: "OPRO not improving after many iterations"

**Diagnosis**:
```bash
# Check reward history
cat results/sprint9_opro/opro_history.json | jq '.best_reward_per_iteration'
```

**Solutions**:
- Adjust reward weights (modify `reward_weights` in `OPROOptimizer.__init__`)
- Increase `candidates_per_iter` for more exploration
- Lower `early_stopping` patience (e.g., 3 instead of 5)

### Issue: "Evaluation taking too long"

**Expected time per evaluation**:
- Dev set: ~25-30 minutes (1400 samples)
- Test set: ~7-10 minutes (340 samples)

**If slower**:
- Check GPU utilization: Should be ~90%+
- Ensure 4-bit quantization enabled
- Consider using fewer bootstrap samples (1000 → 500)

---

## File Structure After Optimization

```
results/sprint9_opro/
├── opro_prompts.jsonl              # Full history (all iterations)
├── opro_memory.json                 # Top-10 best prompts
├── opro_history.json                # Reward curve over time
├── best_prompt.txt                  # BEST: Use this for downstream
├── best_metrics.json                # Best prompt's full metrics
├── dev_predictions.parquet          # Dev set predictions (best prompt)
├── dev_clips.parquet                # Clip-level aggregation
├── dev_metrics.json                 # Dev set metrics
├── psychometric_opro/               # Refitted curves
│   ├── duration_curve.png
│   ├── psychometric_results.json
│   └── baseline_vs_opro.json
├── test_set_opro/                   # ONE-TIME test eval
│   ├── test_predictions.parquet
│   ├── test_results.json
│   └── baseline_vs_opro_test.json
└── comparison_report.md             # Final comparison report
```

**Key files to review**:
1. `best_prompt.txt` - The optimized prompt
2. `comparison_report.md` - Human-readable results
3. `psychometric_opro/baseline_vs_opro.json` - Quantitative comparison

---

## Example: Complete Workflow

```bash
# 1. Initial test (validate pipeline)
python scripts/run_opro.py \
    --n_iterations 5 \
    --output_dir results/sprint9_opro_test

# 2. Review test results
cat results/sprint9_opro_test/best_prompt.txt
cat results/sprint9_opro_test/best_metrics.json

# 3. If test looks good, launch full optimization
python scripts/run_opro.py \
    --n_iterations 50 \
    --early_stopping 5 \
    --output_dir results/sprint9_opro

# 4. Refit psychometric curves
python scripts/refit_psychometric_opro.py \
    --opro_dir results/sprint9_opro

# 5. Review comparison report
cat results/sprint9_opro/comparison_report.md

# 6. If dev results look good, evaluate on test (ONE TIME)
python scripts/evaluate_opro_test.py \
    --opro_dir results/sprint9_opro

# 7. Create git tag
git tag -a v2.0-opro-baseline -m "OPRO optimization complete"
git push origin v2.0-opro-baseline
```

---

## Cost Estimation

**Anthropic API pricing** (Claude 3.5 Sonnet):
- Input: $3/M tokens
- Output: $15/M tokens

**Typical OPRO run** (50 iterations, 3 candidates/iter, early stopping):
- Meta-prompts: ~1k tokens each × 50 = 50k tokens input
- LLM responses: ~500 tokens each × 50 = 25k tokens output
- **Estimated cost**: ~$0.15 + $0.38 = **$0.53 total**

**Qwen2-Audio evaluation** (runs locally on GPU):
- Free (uses your GPU)

**Total estimated cost**: <$1 for full optimization 🎉

---

## Next Steps After Optimization

1. ✅ Review `comparison_report.md`
2. ✅ Check if improvements meet success criteria (Δ BA_clip > +0.03)
3. ✅ Evaluate on test set (ONE TIME)
4. ✅ Document findings in final report
5. ✅ Tag release: `v2.0-opro-baseline`
6. 🔄 If needed: Adjust reward weights and re-run

---

## Support

- **Documentation**: See `docs/sprints/SPRINT9_OPRO_SPECIFICATION.md`
- **Implementation**: All scripts in `scripts/` with `--help` flag
- **Issues**: Check `results/sprint9_opro/opro_prompts.jsonl` for iteration logs
