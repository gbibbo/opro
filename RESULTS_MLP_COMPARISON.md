# MLP Targets vs Attention-Only Comparison

## Training Configuration

**Base Configuration:**
- Model: Qwen2-Audio-7B-Instruct
- LoRA rank: 16, alpha: 32
- Batch size: 2, gradient accumulation: 8 (effective batch: 16)
- Learning rate: 2e-4
- Epochs: 3
- Quantization: 4-bit (QLoRA)

**Attention-Only (default):**
- Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- Trainable params: 20,738,048 (0.2457%)
- Model size: ~84MB

**Attention + MLP:**
- Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Trainable params: 43,909,120 (0.5202%)
- Model size: ~168MB

---

## Results (Seed 42)

| Configuration | Overall Accuracy | SPEECH Acc | NONSPEECH Acc | Trainable % | Model Size |
|---------------|-----------------|------------|---------------|-------------|------------|
| Attention-Only | 93.8% (30/32) | 87.5% | 100.0% | 0.25% | 84MB |
| **Attention + MLP** | **96.9% (31/32)** | **100.0%** | **93.8%** | 0.52% | 168MB |

**Improvement**: +3.1% overall accuracy (+12.5% SPEECH, -6.2% NONSPEECH)

---

## Training Loss Comparison

**Attention-Only (seed 42):**
- Epoch 1.25: loss=0.3968
- Epoch 2.5: loss=0.1673
- Final: train_loss=0.2481

**Attention + MLP (seed 42):**
- Epoch 1.25: loss=0.3967
- Epoch 2.5: loss=0.1663
- Final: train_loss=0.2483

**Observation**: Training losses are nearly identical, suggesting similar learning dynamics.

---

## Multi-Seed Results (Attention-Only)

| Seed | Overall | SPEECH | NONSPEECH |
|------|---------|--------|-----------|
| 42   | 93.8%   | 87.5%  | 100.0%    |
| 123  | 93.8%   | 87.5%  | 100.0%    |
| 456  | 96.9%   | 93.8%  | 100.0%    |
| 2024 | 93.8%   | 87.5%  | 100.0%    |

**Statistics:**
- Mean: 94.6% ± 1.3%
- 95% CI: [92.7%, 96.4%]
- Range: [93.8%, 96.9%]

---

## Analysis

### What Worked

1. **MLP targets improved SPEECH accuracy** from 87.5% → 100.0% (seed 42)
   - This is significant: 0 misclassifications on SPEECH samples
   - Suggests MLP layers help capture acoustic features of human voice

2. **Model size doubled** (84MB → 168MB) but **still practical**
   - 2.1× more trainable parameters
   - Inference time likely unaffected (LoRA merging)
   - Disk space: freed 1.8GB, model fits comfortably

3. **Training completed successfully** after disk cleanup
   - `save_only_model=True` prevented optimizer state bloat
   - ~10 minutes training time (acceptable)

### Trade-offs

1. **NONSPEECH accuracy decreased slightly** (100% → 93.8%)
   - 1 more misclassification on NONSPEECH samples
   - Could be random variance (need McNemar test)

2. **Same best result as attention-only seed 456** (both 96.9%)
   - MLP advantage not statistically clear from single seed
   - Need multi-seed MLP training for robust comparison

---

## Next Steps

### Immediate (Statistical Validation)

1. **Multi-seed MLP training** (seeds 123, 456, 789, 2024)
   ```bash
   for seed in 123 456 789 2024; do
       python scripts/finetune_qwen_audio.py --add_mlp_targets --seed $seed \
           --output_dir checkpoints/mlp_multiseed/seed_$seed
   done
   ```

2. **McNemar test**: Compare attention-only vs MLP (seed 42)
   ```bash
   # Generate predictions with logit scoring
   python scripts/test_with_logit_scoring.py \
       --checkpoint checkpoints/qwen2_audio_speech_detection_multiseed/seed_42/final \
       --save_predictions results/attention_only_seed42.json

   python scripts/test_with_logit_scoring.py \
       --checkpoint checkpoints/with_mlp_seed_42/final \
       --save_predictions results/mlp_seed42.json

   # Statistical comparison
   python scripts/compare_models_mcnemar.py \
       --model_a results/attention_only_seed42.json \
       --model_b results/mlp_seed42.json \
       --name_a "Attention-only (seed 42)" \
       --name_b "Attention+MLP (seed 42)"
   ```

3. **Meta-analysis**: Compare mean±SD across all seeds
   - Attention-only: 94.6% ± 1.3% (4 seeds)
   - MLP: TBD (need 5 seeds for fair comparison)

### Later (Optimization)

4. **Hyperparameter grid search** (if MLP shows consistent advantage)
   - LoRA rank: {8, 16, 32}
   - LoRA alpha: {16, 32, 64}
   - Identify optimal HP for MLP configuration

5. **OPRO on fine-tuned model** (prompt optimization after FT)
   - Freeze best MLP model
   - Optimize prompt with OPRO on dev set
   - Test hypothesis: FT + OPRO > FT or OPRO alone

---

## Disk Space Management

**Issue**: Training with MLP targets initially failed due to disk full (C: drive 100% full)

**Solution Applied**:
1. Removed failed MLP checkpoint (~312MB)
2. Removed intermediate checkpoints from multi-seed runs (~500MB)
3. Removed intermediate checkpoints from normalized model (~200MB)
4. **Total freed**: ~1.8GB (now sufficient for MLP model saves)

**Current Disk Usage**:
```
C:              930G  928G  1.8G 100% /c
```

**Recommendation**: Clean HuggingFace cache (22GB) if more space needed
```bash
huggingface-cli delete-cache
```

---

## Conclusion

**MLP targets show promise** but need more validation:
- ✅ **Best SPEECH accuracy achieved**: 100% (16/16) on seed 42
- ✅ **Training successful** after disk cleanup
- ⚠️ **Statistical significance unclear**: Same 96.9% as best attention-only seed
- ⚠️ **Trade-off observed**: SPEECH ↑, NONSPEECH ↓

**Recommendation**: Run multi-seed MLP training + McNemar test before concluding.

---

**Last Updated**: 2025-10-20
**Status**: MLP training verified, awaiting multi-seed validation
