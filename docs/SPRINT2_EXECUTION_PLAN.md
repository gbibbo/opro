# SPRINT 2: Model Comparisons & Baselines - Execution Plan

**Objective**: Compare fine-tuned Qwen2-Audio against baselines and prompt-optimization methods

**Duration**: ~6-8 hours (mostly computational)

**Prerequisites**:
- ✅ Fine-tuned models available (seeds 42, 123, 456)
- ✅ Test set ready (24 samples)
- ✅ Evaluation infrastructure (evaluate_with_logits.py)

---

## Overview

SPRINT 2 focuses on establishing a complete comparison table:

1. **Baseline Methods** (no fine-tuning)
   - Qwen2-Audio zero-shot (prompt-only)
   - Qwen2-Audio + OPRO prompt optimization
   - Qwen3/Qwen2.5-Omni + OPRO

2. **Fine-Tuned Methods**
   - Qwen2-Audio + LoRA (current best: 83.3%)
   - Qwen2-Audio + LoRA + OPRO post-FT

3. **Classical Baselines**
   - Silero VAD (completed: 66.7%)
   - WebRTC VAD (if compilable)

---

## Task Breakdown

### 2.1 OPRO Post-FT (~3-4 hours)

**Concept**: Optimize prompt on the **frozen fine-tuned model**

**Why This Works**:
- Fine-tuned Qwen2-Audio still accepts text prompts
- OPRO can find better instruction phrasing
- Combines benefits of FT (learned features) + prompt optimization

**Steps**:

#### A. Prepare OPRO for Post-FT

Currently OPRO optimizes on base model. Need to adapt for FT model.

**Script to create**: `scripts/opro_post_ft.py`

Key changes from original OPRO:
```python
# Load fine-tuned model instead of base
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "checkpoints/ablations/LORA_attn_mlp/seed_42/final"  # Fine-tuned checkpoint
)

# Keep model frozen (no gradient updates)
for param in model.parameters():
    param.requires_grad = False

# Only optimize the prompt text
```

#### B. Run OPRO on Dev Set

```bash
# Use dev set (72 samples) for optimization
python scripts/opro_post_ft.py \
    --checkpoint checkpoints/ablations/LORA_attn_mlp/seed_42/final \
    --train_csv data/processed/grouped_split/dev_metadata.csv \
    --meta_prompt_file prompts/meta_prompt_post_ft.txt \
    --num_iterations 20 \
    --samples_per_iter 10 \
    --output_dir results/opro_post_ft
```

**Expected time**: 2-3 hours (72 samples × 20 iterations)

#### C. Evaluate Best Prompt on Test

```bash
# Get best prompt from OPRO
BEST_PROMPT=$(cat results/opro_post_ft/best_prompt.txt)

# Evaluate on test with best prompt
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/ablations/LORA_attn_mlp/seed_42/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --prompt "$BEST_PROMPT" \
    --temperature 1.0 \
    --output_csv results/comparisons/ft_opro_seed42.csv
```

**Expected improvement**: +2-5% over FT alone (based on original OPRO gains)

---

### 2.2 Qwen2-Audio Baselines (~2-3 hours)

**Goal**: Establish how well the base model performs with prompt optimization (no FT)

#### A. Zero-Shot Baseline

```bash
# Evaluate base model with default prompt
python scripts/evaluate_with_logits.py \
    --checkpoint Qwen/Qwen2-Audio-7B-Instruct \  # Base model from HF
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --prompt "You classify audio content. Choose one: A) SPEECH (human voice) B) NONSPEECH (music/noise/silence/animals). Answer with A or B ONLY." \
    --temperature 1.0 \
    --output_csv results/comparisons/base_zeroshot.csv
```

**Expected**: ~70-85% (based on previous experiments)

#### B. OPRO on Base Model

```bash
# Run OPRO on base model (not fine-tuned)
python scripts/opro_qwen_audio.py \
    --model_name Qwen/Qwen2-Audio-7B-Instruct \
    --train_csv data/processed/grouped_split/dev_metadata.csv \
    --meta_prompt_file prompts/meta_prompt_base.txt \
    --num_iterations 20 \
    --samples_per_iter 10 \
    --output_dir results/opro_base

# Evaluate best prompt on test
BEST_PROMPT=$(cat results/opro_base/best_prompt.txt)
python scripts/evaluate_with_logits.py \
    --checkpoint Qwen/Qwen2-Audio-7B-Instruct \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --prompt "$BEST_PROMPT" \
    --output_csv results/comparisons/base_opro.csv
```

**Expected**: ~85-95% (based on original OPRO results)

---

### 2.3 Qwen3 / Qwen2.5-Omni Baseline (~2 hours)

**Goal**: Compare against newer Qwen models

**Note**: Qwen3 might not exist yet. Check availability first.

```bash
# Check if Qwen2.5-Omni is available
huggingface-cli repo info Qwen/Qwen2.5-Omni

# If available, evaluate with OPRO-optimized prompt
python scripts/evaluate_with_logits.py \
    --checkpoint Qwen/Qwen2.5-Omni \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --prompt "$BEST_PROMPT" \  # Use prompt from step 2.2B
    --output_csv results/comparisons/qwen25_omni.csv
```

**Alternative if Qwen2.5-Omni unavailable**:
- Use Qwen2-Audio-7B-Instruct (latest version)
- Or skip this comparison

---

### 2.4 Create Comprehensive Comparison Table (~30 min)

**Script to create**: `scripts/create_comparison_table.py`

```bash
python scripts/create_comparison_table.py \
    --prediction_csvs \
        results/comparisons/base_zeroshot.csv \
        results/comparisons/base_opro.csv \
        results/comparisons/ft_seed42.csv \
        results/comparisons/ft_opro_seed42.csv \
        results/baselines/silero_vad_predictions.csv \
    --labels \
        "Qwen2-Audio (zero-shot)" \
        "Qwen2-Audio + OPRO" \
        "Qwen2-Audio + LoRA" \
        "Qwen2-Audio + LoRA + OPRO" \
        "Silero VAD" \
    --output_table results/final_comparison_table.md \
    --output_plot results/final_comparison_plot.png
```

**Output Format**:

| Method | Overall | SPEECH | NONSPEECH | ROC-AUC | Parameters Trained |
|--------|---------|--------|-----------|---------|-------------------|
| Silero VAD | 66.7% | 0.0% | 100.0% | - | 0 (pretrained) |
| Qwen2-Audio (zero-shot) | ~75% | ~50% | ~90% | ~0.95 | 0 |
| Qwen2-Audio + OPRO | ~90% | ~75% | ~97% | ~0.98 | 0 (prompt only) |
| **Qwen2-Audio + LoRA** | **83.3%** | **50.0%** | **100.0%** | **1.00** | 44M (0.52%) |
| **Qwen2-Audio + LoRA + OPRO** | **~88%** | **~65%** | **~100%** | **1.00** | 44M (0.52%) |

---

## Detailed Task: OPRO Post-FT Implementation

Since this is the most novel part, here's the detailed implementation:

### A. Create OPRO Post-FT Script

Key modifications from original OPRO:

```python
# scripts/opro_post_ft.py

import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel

def load_finetuned_model(checkpoint_dir):
    """Load fine-tuned LoRA model."""
    # Load base model
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    # Merge LoRA weights and unload (optional, for speed)
    model = model.merge_and_unload()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model

def evaluate_prompt_on_finetuned(model, processor, prompt, samples):
    """Evaluate a candidate prompt on fine-tuned model."""
    correct = 0
    for sample in samples:
        # Use evaluate_sample_logits with custom prompt
        result = evaluate_sample_logits(
            model, processor,
            sample['audio_path'],
            ids_A, ids_B,
            system_prompt="You classify audio content.",
            user_prompt=prompt  # Variable prompt
        )
        if result['prediction'] == sample['ground_truth_token']:
            correct += 1

    return correct / len(samples)

# Rest is same as original OPRO: generate prompts, evaluate, select best
```

### B. Meta-Prompt for Post-FT

`prompts/meta_prompt_post_ft.txt`:
```
Your task is to generate prompts for a FINE-TUNED audio classification model.

The model has been fine-tuned on speech detection and already has good features.
Your prompt should help it use those features optimally.

Previous prompts and their accuracy:
{prompt_history}

Generate a new prompt that improves accuracy. Requirements:
- Clear binary choice: A) SPEECH vs B) NONSPEECH
- Leverage model's learned features
- Guide decision-making for edge cases
- Keep it concise (< 50 words)

New prompt:
```

---

## Expected Outcomes

### Success Criteria

1. ✅ OPRO post-FT improves accuracy by ≥2% (83.3% → 85%+)
2. ✅ Base + OPRO achieves ~90% (validates prompt optimization works)
3. ✅ Complete comparison table with ≥4 methods
4. ✅ Statistical tests (McNemar) for pairwise comparisons
5. ✅ Publication-ready figures and tables

### Comparison Table (Expected)

| Method | Accuracy | Training Cost | Inference Cost | Best For |
|--------|----------|---------------|----------------|----------|
| Silero VAD | 66.7% | None | Very Fast | Real-time systems |
| Qwen2 Zero-shot | ~75% | None | Slow | Quick prototyping |
| Qwen2 + OPRO | ~90% | Prompt search | Slow | No training data |
| Qwen2 + LoRA | 83.3% | Medium | Slow | Small datasets |
| **Qwen2 + LoRA + OPRO** | **~88%** | Medium | Slow | **Best accuracy** |

---

## Execution Order (Recommended)

### Parallel Track 1 (No Model Loading)
```bash
# 1. Create comparison table script
# 2. Create OPRO post-FT script
# 3. Update meta-prompts
```

### Sequential Track 2 (Requires Model Loading - Do ONE at a time due to VRAM)
```bash
# Step 1: OPRO on base model (2 hours)
python scripts/opro_qwen_audio.py ...

# Step 2: OPRO on fine-tuned model (2 hours)
python scripts/opro_post_ft.py ...

# Step 3: Evaluate Qwen2.5-Omni if available (30 min)
python scripts/evaluate_with_logits.py ...

# Step 4: Generate comparison table (5 min)
python scripts/create_comparison_table.py ...
```

**Total Time**: ~5-6 hours sequential

---

## Scripts to Create

1. ✅ `scripts/opro_qwen_audio.py` - Already exists
2. ⏳ `scripts/opro_post_ft.py` - Need to create (adaptation)
3. ⏳ `scripts/create_comparison_table.py` - Need to create
4. ⏳ `scripts/mcnemar_comparison.py` - Need to update for new results
5. ⏳ `prompts/meta_prompt_post_ft.txt` - Need to create

---

## Contingency Plans

### If VRAM Issues Persist

**Option A**: Run OPRO on CPU
- Slower but feasible
- ~4-6 hours instead of 2-3 hours

**Option B**: Reduce OPRO iterations
- 10 iterations instead of 20
- Faster but less optimal

**Option C**: Skip Qwen2.5-Omni
- Focus on Qwen2-Audio comparisons only

### If OPRO Post-FT Doesn't Improve

**Fallback**: Document that fine-tuning already optimizes the model enough
- This is a valid finding
- Shows FT >> prompt optimization for this task

---

## SPRINT 2 Completion Criteria

- [ ] OPRO post-FT script created and tested
- [ ] OPRO run on base model (dev set)
- [ ] OPRO run on fine-tuned model (dev set)
- [ ] Best prompts evaluated on test set
- [ ] Comparison table generated (≥4 methods)
- [ ] McNemar tests for statistical significance
- [ ] Final plots and figures created
- [ ] Results documented in SPRINT2_FINAL_REPORT.md

---

**Ready to Start**: Yes (all prerequisites met)

**Estimated Completion**: 6-8 hours (mostly computational)

**Next Step**: Create `scripts/opro_post_ft.py`
