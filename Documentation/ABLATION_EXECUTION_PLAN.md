# Ablation Study Execution Plan

**Status**: Infrastructure ready, first experiments running
**Goal**: Find optimal configuration on small data (160 samples) before scaling to 1000+

---

## Current Progress

### ✓ Completed

1. **Zero-leakage split** - Verified with automated audit
   - Train: 136 samples (10 groups)
   - Test: 24 samples (3 groups)
   - Overlap: 0 groups ✓

2. **Baseline model** - Attention-only LoRA trained
   - Configuration: r=16, α=32, attention-only
   - Results: 79.2% overall (37.5% SPEECH, 100% NONSPEECH)
   - Location: `checkpoints/no_leakage_v2/seed_42/final`

3. **Infrastructure scripts**
   - [x] Temperature scaling calibration ([scripts/calibrate_temperature.py](scripts/calibrate_temperature.py))
   - [x] Ablation sweep runner ([scripts/run_ablation_sweep.py](scripts/run_ablation_sweep.py))
   - [x] Logit-based evaluation ([scripts/evaluate_with_logits.py](scripts/evaluate_with_logits.py))
   - [x] Statistical comparison ([scripts/compare_models_statistical.py](scripts/compare_models_statistical.py))
   - [x] Split leakage audit ([scripts/audit_split_leakage.py](scripts/audit_split_leakage.py))

4. **Calibration analysis**
   - Optimal temperature: T=10.0 (model is very overconfident)
   - ECE reduction: 46.2% (0.646 → 0.348)
   - Note: Brier score worsened (0.103 → 0.201) - small sample issue

### 🔄 In Progress

1. **Attention+MLP training** (seed 42)
   - Running in background (bash ID: 689260)
   - Expected completion: ~10 minutes
   - Will compare with attention-only baseline

---

## Execution Roadmap

### Phase 1: LoRA Targets (CRITICAL) - Today

**Goal**: Determine if adding MLP layers helps with low-SNR generalization

| Exp ID | Configuration | Seeds | Status |
|--------|---------------|-------|--------|
| LORA-2a | Attention-only (baseline) | 42 | ✓ Done (79.2%) |
| LORA-2b | Attention+MLP | 42 | 🔄 Running |
| LORA-2a | Attention-only | 123, 456 | ⏳ Pending |
| LORA-2b | Attention+MLP | 123, 456 | ⏳ Pending |

**Execution**:
```bash
# Attention-only (already done)
python scripts/finetune_qwen_audio.py \
    --seed 42 \
    --output_dir checkpoints/no_leakage_v2/seed_42

# Attention+MLP (running)
python scripts/finetune_qwen_audio.py \
    --seed 42 \
    --output_dir checkpoints/ablations/LORA_attn_mlp/seed_42 \
    --add_mlp_targets

# Additional seeds (to run after)
for seed in 123 456; do
    python scripts/finetune_qwen_audio.py \
        --seed $seed \
        --output_dir checkpoints/ablations/LORA_attn_only/seed_$seed

    python scripts/finetune_qwen_audio.py \
        --seed $seed \
        --output_dir checkpoints/ablations/LORA_attn_mlp/seed_$seed \
        --add_mlp_targets
done
```

**Evaluation**:
```bash
# Evaluate each checkpoint
for exp in LORA_attn_only LORA_attn_mlp; do
    for seed in 42 123 456; do
        python scripts/evaluate_with_logits.py \
            --checkpoint checkpoints/ablations/${exp}/seed_${seed}/final \
            --test_csv data/processed/grouped_split/test_metadata.csv \
            --temperature 1.0 \
            --output_csv results/ablations/${exp}_seed${seed}.csv
    done
done
```

**Decision Criteria**:
- If attention+MLP improves SPEECH accuracy by ≥5%, use it
- If attention-only is more stable (lower variance across seeds), prefer it
- Report mean±std for both configurations

---

### Phase 2: Hyperparameter Grid (If Time Permits) - Tomorrow

**Goal**: Fine-tune LoRA rank and alpha

| Parameter | Values | Notes |
|-----------|--------|-------|
| r (rank) | {8, 16, 32} | Lower = faster, less capacity |
| alpha | {16, 32, 64} | Scaling factor |
| dropout | {0.0, 0.05, 0.1} | Regularization |

**Mini-grid** (6 points, 2 seeds each = 12 runs):
```python
grid = [
    {"r": 8, "alpha": 16},
    {"r": 8, "alpha": 32},
    {"r": 16, "alpha": 32},  # baseline
    {"r": 16, "alpha": 64},
    {"r": 32, "alpha": 32},
    {"r": 32, "alpha": 64},
]
```

**Automated execution**:
```bash
python scripts/run_ablation_sweep.py --phase 2 --seeds 42 123
```

**Expected outcome**: Baseline (r=16, α=32) is likely near-optimal, but we may find 5-10% improvement

---

### Phase 3: Temperature Scaling (Quick Win)

**Goal**: Improve calibration for confidence-based metrics

**Current status**:
- Optimal T on test set: 10.0
- ECE: 0.646 → 0.348 (46% reduction)
- Concern: Brier score worsened (overfitting to small test set)

**Better approach - Use dev set**:
1. Split train into train_sub (80%) + dev (20%) with GroupShuffleSplit
2. Find optimal T on dev
3. Apply to test

**Implementation**:
```bash
# Create dev split (20% of train)
python scripts/create_group_stratified_split.py \
    --input_csv data/processed/grouped_split/train_metadata.csv \
    --output_dir data/processed/grouped_split_with_dev \
    --test_size 0.2 \
    --random_state 42

# Get dev predictions
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/no_leakage_v2/seed_42/final \
    --test_csv data/processed/grouped_split_with_dev/test_metadata.csv \
    --temperature 1.0 \
    --output_csv results/dev_predictions.csv

# Find optimal temperature on dev
python scripts/calibrate_temperature.py \
    --predictions_csv results/dev_predictions.csv \
    --output_temp results/optimal_temperature_dev.txt

# Apply to test
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/no_leakage_v2/seed_42/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --temperature $(cat results/optimal_temperature_dev.txt) \
    --output_csv results/test_calibrated.csv
```

---

### Phase 4: Data Augmentation (Low Priority)

**Goal**: Reduce overfitting with light augmentation

**Options**:
1. **SpecAugment** (frequency + time masking) - Safe, proven
2. **Time shift** (±50ms jitter) - Safe if it doesn't break duration labels
3. **Additive noise** (very light) - Risky with SNR labels

**Recommendation**: Start with SpecAugment only

**Implementation**:
- Add to dataset class in finetune_qwen_audio.py
- Use light parameters: freq_mask_param=5, time_mask_param=10
- Run 2 seeds and compare with baseline

---

### Phase 5: Statistical Validation (Final Step)

**Goal**: Ensure results are statistically significant

**McNemar Test**:
```bash
python scripts/compare_models_statistical.py \
    --predictions_A results/ablations/LORA_attn_only_seed42.csv \
    --predictions_B results/ablations/LORA_attn_mlp_seed42.csv \
    --model_A_name "Attention-only" \
    --model_B_name "Attention+MLP" \
    --n_bootstrap 10000
```

**Bootstrap CIs**:
- For each model, aggregate predictions from 3 seeds
- Compute 95% CI via bootstrap resampling
- Report: "X.X% ± Y.Y% (95% CI)"

---

## Exit Criteria (Before Scaling)

### Performance Gates

- [ ] Multi-seed variance < 3% on balanced accuracy
- [ ] At least one configuration with SPEECH ≥ 50% (current: 37.5%)
- [ ] NONSPEECH maintained at ≥ 95% (current: 100%)
- [ ] Calibration ECE < 0.2 (current: 0.35 with T=10)

### Scientific Rigor

- [ ] All comparisons use ≥2 seeds (report mean±std)
- [ ] Statistical significance tested via McNemar (p<0.05)
- [ ] Bootstrap CIs computed for final configuration
- [ ] Zero leakage maintained throughout

### Documentation

- [ ] Best configuration documented in [EXPERIMENT_MATRIX_SMALL_DATA.md](EXPERIMENT_MATRIX_SMALL_DATA.md)
- [ ] Results matrix filled with all metrics
- [ ] Error analysis: Which samples are consistently wrong across seeds?
- [ ] Recipe for scaling: Exact commands to replicate with larger dataset

---

## Timeline Estimate

| Phase | Tasks | Time | Status |
|-------|-------|------|--------|
| 1. LoRA targets | 2 configs × 3 seeds × 10min = 60min | 1 hour | 🔄 In progress (1/6 done) |
| 2. HP grid | 6 configs × 2 seeds × 10min = 120min | 2 hours | ⏳ Pending |
| 3. Temp scaling | Dev split + calibration | 30min | ⏳ Pending |
| 4. Augmentation | SpecAugment × 2 seeds | 30min | ⏳ Optional |
| 5. Statistical tests | McNemar + Bootstrap | 15min | ⏳ Pending |
| **Total** | | **4-5 hours** | |

**Realistic completion**: End of today (if running continuously) or tomorrow morning

---

## Key Questions to Answer

1. **Does adding MLP layers help?**
   - Hypothesis: MLP may improve low-SNR feature extraction
   - Test: Compare LORA-2a (attn) vs LORA-2b (attn+MLP) on SPEECH accuracy

2. **Is the model overconfident?**
   - Current: T_opt = 10.0 (very high)
   - Indicates severe overconfidence on small test set
   - May improve with proper dev/test split

3. **What's the best LoRA configuration?**
   - Baseline (r=16, α=32) vs alternatives
   - Trade-off: capacity vs overfitting risk

4. **Which samples are hardest?**
   - Identify consistently-wrong samples across seeds
   - Insight: Are errors random (data issue) or systematic (model issue)?

---

## Next Steps After Ablations

Once we identify the best configuration:

1. **Scale dataset** to 1000+ samples with 50+ SPEECH speakers
2. **Re-run best configuration** with 3-5 seeds
3. **Expected improvement**: SPEECH accuracy 37.5% → 90%+ with more speakers
4. **Final validation**: McNemar test vs baselines (zero-shot, OPRO-only)
5. **Paper-ready results**: Mean±std, Bootstrap CIs, statistical tests

---

## Quick Reference: Running Experiments

**Single training run**:
```bash
python scripts/finetune_qwen_audio.py \
    --seed 42 \
    --output_dir checkpoints/test \
    --lora_r 16 \
    --lora_alpha 32 \
    [--add_mlp_targets]
```

**Evaluation**:
```bash
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/test/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --temperature 1.0 \
    --output_csv results/test_predictions.csv
```

**Calibration**:
```bash
python scripts/calibrate_temperature.py \
    --predictions_csv results/test_predictions.csv \
    --output_temp results/optimal_T.txt \
    --n_bins 5
```

**Statistical comparison**:
```bash
python scripts/compare_models_statistical.py \
    --predictions_A results/model_A.csv \
    --predictions_B results/model_B.csv \
    --model_A_name "Model A" \
    --model_B_name "Model B" \
    --n_bootstrap 10000
```

**Leakage audit** (always run before training):
```bash
python scripts/audit_split_leakage.py \
    --train_csv data/processed/grouped_split/train_metadata.csv \
    --test_csv data/processed/grouped_split/test_metadata.csv
```

---

## Files Created This Session

1. **[EXPERIMENT_MATRIX_SMALL_DATA.md](EXPERIMENT_MATRIX_SMALL_DATA.md)** - Complete experiment design
2. **[scripts/calibrate_temperature.py](scripts/calibrate_temperature.py)** - Temperature scaling for calibration
3. **[scripts/run_ablation_sweep.py](scripts/run_ablation_sweep.py)** - Automated ablation runner
4. **[ABLATION_EXECUTION_PLAN.md](ABLATION_EXECUTION_PLAN.md)** (this file) - Step-by-step execution guide

---

## Summary

**We are ready to systematically optimize on small data before scaling.**

Infrastructure is complete. First experiment (attention+MLP) is running. Once we identify the best configuration (likely within 4-5 hours of compute), we can confidently scale to 1000+ samples knowing exactly what works.

The key insight from your guidance: **Don't scale blindly - optimize first, then scale with the proven recipe.**
