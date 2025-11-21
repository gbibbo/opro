# Experiment Matrix: Small Data Optimization

**Goal**: Optimize all configurations on small dataset (160 samples) BEFORE scaling to 1000+ samples

**Current Status**: Zero-leakage split validated, baseline attention-only LoRA trained (79.2% accuracy)

**Strategy**: Systematic ablation to find best configurations, then scale with proven recipe

---

## Baseline Configuration

```yaml
Model: Qwen2-Audio-7B-Instruct
Quantization: 4-bit NF4
LoRA:
  rank: 16
  alpha: 32
  targets: [q_proj, v_proj, k_proj, o_proj]  # attention-only
  dropout: 0.0
Training:
  epochs: 3
  batch_size: 2
  learning_rate: 2e-4
  warmup_ratio: 0.05
  optimizer: AdamW
Dataset:
  train: 136 samples (10 groups)
  test: 24 samples (3 groups)
  zero_leakage: VERIFIED
```

**Baseline Results**:
- Overall: 79.2% (19/24)
- SPEECH: 37.5% (3/8) - limited by single test speaker
- NONSPEECH: 100% (16/16)

---

## Experiment Matrix

### 1. Training Objective (Ablation)

**Hypothesis**: Binary classification head may generalize better than text generation with low data

| Exp ID | Objective | Architecture | Expected Impact |
|--------|-----------|--------------|-----------------|
| OBJ-1a | CE on A/B tokens (current) | Text generation | Baseline |
| OBJ-1b | Binary classification head | Linear(hidden_dim, 2) + CE | May improve SPEECH generalization |

**Implementation**:
- Add classification head on top of pooled audio-text representation
- Compare cross-entropy loss on binary labels
- Evaluate: Does direct classification bypass speaker-specific text generation?

**Validation**: 3 seeds each, report mean ± std

---

### 2. LoRA Target Layers (Ablation)

**Hypothesis**: Adding MLP layers may help with feature extraction for low-SNR audio

| Exp ID | Targets | Trainable Params | Expected Trade-off |
|--------|---------|------------------|---------------------|
| LORA-2a | attention-only (q,k,v,o) | 20.7M (0.25%) | Baseline - fast, stable |
| LORA-2b | attention+MLP (q,k,v,o,gate,up,down) | ~35M (0.42%) | May improve NONSPEECH at low SNR |

**Grid**:
- r ∈ {8, 16, 32}
- alpha ∈ {16, 32, 64}
- lr ∈ {1e-4, 2e-4}
- dropout ∈ {0.0, 0.05, 0.1}

**Validation**: 2 seeds per promising point

**Reference**: LoRA paper ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685)), Qwen2-Audio ([arXiv:2407.10759](https://arxiv.org/abs/2407.10759))

---

### 3. Inference Method + Calibration

**Hypothesis**: Temperature scaling improves confidence calibration without changing accuracy

| Exp ID | Method | Temperature | Metrics |
|--------|--------|-------------|---------|
| INF-3a | Logits (1-step) | T=1.0 | Accuracy, ECE (uncalibrated) |
| INF-3b | Logits + temp scaling | T=optimal (via dev) | Accuracy, ECE↓, Brier Score↓ |
| INF-3c | Generate (baseline) | N/A | Slower, for comparison only |

**Implementation**:
- Grid search T ∈ [0.5, 0.7, 1.0, 1.5, 2.0] on dev set
- Optimize for Expected Calibration Error (ECE)
- Report calibration curves (confidence vs accuracy)

**Reference**: Temperature scaling ([arXiv:1706.04599](https://arxiv.org/abs/1706.04599))

---

### 4. Curriculum Learning

**Hypothesis**: Start with easy (1000ms) then introduce hard (200ms) improves convergence

| Exp ID | Curriculum Strategy | Epoch Schedule |
|--------|---------------------|----------------|
| CURR-4a | No curriculum (mixed) | All durations from start |
| CURR-4b | Duration curriculum | Epoch 1: 1000ms only, Epoch 2-3: mixed |
| CURR-4c | SNR curriculum | Epoch 1: SNR≥10dB, Epoch 2-3: all SNR |

**Implementation**:
- Modify dataset loader to filter by duration/SNR per epoch
- Track loss curves to see if curriculum accelerates convergence

**Validation**: 2 seeds each

---

### 5. Hard Negative Mining

**Hypothesis**: Re-weighting confused samples improves robustness on edge cases

| Exp ID | Strategy | Implementation |
|--------|----------|----------------|
| HN-5a | No re-weighting (uniform) | Standard CE loss |
| HN-5b | Focal loss (γ=2) | Focus on hard examples automatically |
| HN-5c | Manual hard-negative sampling | 2× sample confused clips (maintaining group split) |

**Target**: Improve NONSPEECH @ 200ms (currently where errors concentrate)

**Implementation**:
- Run baseline model, identify top-10 confused samples
- Oversample those clips (without breaking group split)
- Retrain and measure improvement

**Reference**: Focal Loss ([arXiv:1708.02002](https://arxiv.org/abs/1708.02002))

---

### 6. Data Augmentation

**Hypothesis**: Light augmentation reduces overfitting without destroying SNR/duration cues

| Exp ID | Augmentation | Parameters |
|--------|--------------|------------|
| AUG-6a | None (baseline) | - |
| AUG-6b | SpecAugment | freq_mask=5, time_mask=10 (light) |
| AUG-6c | Time shift | ±50ms jitter |
| AUG-6d | Additive noise | Gaussian σ=0.01 (very light) |

**Constraints**:
- Do NOT break SNR labels (additive noise must be SNR-aware)
- Do NOT break duration labels (time stretch forbidden)

**Validation**: 2 seeds each

**Reference**: SpecAugment ([arXiv:1904.08779](https://arxiv.org/abs/1904.08779))

---

### 7. Post-FT Prompt Optimization (OPRO)

**Hypothesis**: Optimizing prompt on fine-tuned model gives additional 1-2% boost

| Exp ID | Model | Prompt Strategy | Budget |
|--------|-------|-----------------|--------|
| OPRO-7a | FT model + baseline prompt | "Does this audio contain speech?" | - |
| OPRO-7b | FT model + OPRO-optimized | 20-50 candidate prompts | Small |

**Implementation**:
- Use best FT model from previous ablations
- Run OPRO with 20 iterations (5-10 prompts per iteration)
- Evaluate on dev set, final test on hold-out

**Reference**: OPRO ([arXiv:2309.03409](https://arxiv.org/abs/2309.03409))

---

### 8. Baseline Comparison (External Model)

**Goal**: Establish ceiling with current SOTA

| Model | Configuration | Purpose |
|-------|---------------|---------|
| Qwen2-Audio + baseline prompt | Zero-shot | Lower bound |
| Qwen2-Audio + OPRO prompt | Zero-shot | Prompt-only ceiling |
| Qwen3-Omni + OPRO prompt | Zero-shot | SOTA ceiling |
| Qwen2-Audio + LoRA + OPRO | Fine-tuned (ours) | Our best |

**Validation**: McNemar test for statistical significance on paired predictions

**Reference**: McNemar ([Wikipedia](https://en.wikipedia.org/wiki/McNemar%27s_test))

---

## Experiment Execution Plan

### Phase 1: Core Ablations (Days 1-2)

**Priority 1**: Training objective (OBJ-1)
- [x] Baseline: A/B text generation (already done)
- [ ] Binary classification head (implement + train 3 seeds)

**Priority 2**: LoRA targets (LORA-2)
- [x] Attention-only (already done)
- [ ] Attention+MLP (r=16, α=32, lr=2e-4, 3 seeds)

**Priority 3**: Inference + calibration (INF-3)
- [x] Logits T=1.0 (already done)
- [ ] Temperature scaling (grid search + evaluation)

### Phase 2: Data Strategies (Days 3-4)

**Priority 4**: Curriculum (CURR-4)
- [ ] Duration curriculum (2 seeds)
- [ ] SNR curriculum (2 seeds)

**Priority 5**: Hard negatives (HN-5)
- [ ] Identify confused samples
- [ ] Focal loss (2 seeds)
- [ ] Manual oversampling (2 seeds)

**Priority 6**: Augmentation (AUG-6)
- [ ] SpecAugment light (2 seeds)
- [ ] Time shift (2 seeds)

### Phase 3: Final Integration (Day 5)

**Priority 7**: OPRO post-FT (OPRO-7)
- [ ] Take best model from phases 1-2
- [ ] Run OPRO with 20 iterations
- [ ] Evaluate on hold-out test

**Priority 8**: Statistical validation (STAT-8)
- [ ] Collect all predictions on test set
- [ ] McNemar tests for pairwise comparisons
- [ ] Bootstrap CIs for all metrics

---

## Exit Criteria (Before Scaling)

### Performance Gates

- [ ] NONSPEECH @ 200ms: ≥85% (currently 100%, maintain)
- [ ] Multi-seed variance: <3% on balanced accuracy
- [ ] Calibration: ECE <0.1 after temperature scaling
- [ ] FT+OPRO vs FT: ≥+1% improvement (statistically significant via McNemar)

### Scientific Rigor

- [ ] All experiments: 2-3 seeds with mean±std reported
- [ ] Group-based split maintained (zero leakage verified)
- [ ] Statistical tests for all comparisons (McNemar, Bootstrap)
- [ ] Calibration curves plotted (confidence vs accuracy)

### Documentation

- [ ] Results matrix filled with all metrics
- [ ] Best configuration documented (recipe for scaling)
- [ ] Error analysis per configuration
- [ ] Lessons learned summary

---

## Metrics to Track

### Primary Metrics

1. **Balanced Accuracy**: Mean of per-class accuracies
2. **Per-class Accuracy**: SPEECH, NONSPEECH separately
3. **Per-condition Accuracy**: Duration×SNR breakdown (8 conditions)

### Secondary Metrics

4. **Expected Calibration Error (ECE)**: Confidence calibration quality
5. **Brier Score**: Probabilistic prediction quality
6. **Confidence Gap**: Conf(correct) - Conf(wrong)
7. **Training Time**: Wall-clock minutes per epoch

### Statistical Metrics

8. **Multi-seed Variance**: Std dev across 3 seeds
9. **McNemar p-value**: Significance of pairwise differences
10. **Bootstrap 95% CI**: Uncertainty quantification

---

## Results Matrix Template

| Exp ID | Config | Overall Acc | SPEECH | NONSPEECH | ECE | Brier | Seeds | Time |
|--------|--------|-------------|--------|-----------|-----|-------|-------|------|
| BASELINE | attn-only, r=16, α=32 | 79.2±? | 37.5±? | 100±? | ? | ? | 1 | 10min |
| OBJ-1b | binary head | - | - | - | - | - | 3 | - |
| LORA-2b | attn+MLP | - | - | - | - | - | 3 | - |
| INF-3b | temp scaled | - | - | - | - | - | - | - |
| CURR-4b | dur curriculum | - | - | - | - | - | 2 | - |
| HN-5b | focal loss | - | - | - | - | - | 2 | - |
| AUG-6b | SpecAugment | - | - | - | - | - | 2 | - |
| OPRO-7b | post-FT OPRO | - | - | - | - | - | 1 | - |

---

## Implementation Priorities

### Immediate (This Session)

1. **Binary classification head** ([scripts/finetune_binary_head.py](scripts/finetune_binary_head.py))
2. **Temperature scaling** (add to [scripts/evaluate_with_logits.py](scripts/evaluate_with_logits.py))
3. **Curriculum scheduler** ([scripts/curriculum_dataloader.py](scripts/curriculum_dataloader.py))

### Next Session

4. **LoRA targets sweep** (modify [scripts/finetune_qwen_audio.py](scripts/finetune_qwen_audio.py))
5. **Hard negative mining** ([scripts/identify_hard_negatives.py](scripts/identify_hard_negatives.py))
6. **SpecAugment** (add to dataset class)

### Final Session

7. **OPRO integration** ([scripts/opro_post_ft.py](scripts/opro_post_ft.py))
8. **Statistical comparison** (use [scripts/compare_models_statistical.py](scripts/compare_models_statistical.py))
9. **Results aggregation** ([scripts/aggregate_results.py](scripts/aggregate_results.py))

---

## References

1. **LoRA**: [LORA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **Qwen2-Audio**: [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)
3. **Temperature Scaling**: [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
4. **SpecAugment**: [SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)
5. **OPRO**: [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
6. **Focal Loss**: [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
7. **McNemar Test**: [Statistical Methods in AI](https://en.wikipedia.org/wiki/McNemar%27s_test)

---

## Notes

- All experiments respect **GroupShuffleSplit** (zero leakage)
- Multi-seed training uses seeds: {42, 123, 456}
- Hold-out test set (24 samples) never used for hyperparameter selection
- Dev set (if needed) carved from train with same group-split strategy

**Once we find the best configuration, we scale to 1000+ samples and repeat with the proven recipe.**
