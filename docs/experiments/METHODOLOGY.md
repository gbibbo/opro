# Experiment Methodology: 4-Conditions Pipeline

## Overview

This document describes the methodology for the speech detection experiments using Qwen2-Audio with LoRA fine-tuning and OPRO prompt optimization, evaluated across 4 psychoacoustic conditions.

## 1. Data Sources

### Speech Data: VoxConverse
- **Source**: VoxConverse speaker diarization dataset
- **Format**: Multi-speaker audio conversations
- **Sampling**: Random 1000ms segments verified with Silero VAD

### Non-Speech Data: ESC-50
- **Source**: ESC-50 Environmental Sound Classification dataset
- **Format**: 5-second environmental sound clips
- **Sampling**: Random 1000ms segments from environmental sounds

### Noise Data: MUSAN
- **Source**: MUSAN noise corpus
- **Type**: Babble noise for SNR degradation
- **Usage**: Mixed with clean clips at various SNR levels

## 2. Dataset Generation

### Step 1: Base Clip Extraction with Silero VAD Verification

**Script**: `scripts/generate_base_clips_verified.py`

**Configuration**:
```python
TARGET_SR = 16000           # Sample rate
CLIP_DURATION_MS = 1000     # Base clip duration
SPEECH_THRESHOLD = 0.8      # Require >80% speech ratio
```

**Process**:
1. Load VoxConverse audio files
2. Extract random 1000ms segments
3. Run Silero VAD to compute speech ratio
4. Keep only segments with >80% speech
5. Repeat until target count reached

**Verification**: Each SPEECH clip is guaranteed to contain >80% actual speech content, addressing the issue where VoxConverse segments may contain silence or non-speech segments.

### Step 2: 4-Conditions Expansion

**Script**: `scripts/generate_expanded_dataset_4conditions.py`

**Conditions Applied**:

#### Duration Variations (8 levels)
- 20ms, 40ms, 60ms, 80ms, 100ms, 200ms, 500ms, 1000ms
- All other conditions at baseline (clean, no filter, no reverb)

#### SNR Variations (6 levels)
- -10dB, -5dB, 0dB, 5dB, 10dB, 20dB
- Duration fixed at 1000ms, no filter, no reverb
- Noise source: MUSAN babble noise

#### Band Filter Variations (3 types)
- **Telephony**: 300-3400Hz bandpass
- **LP3400**: 3400Hz lowpass
- **HP300**: 300Hz highpass
- Duration fixed at 1000ms, clean, no reverb

#### Reverb Variations (3 T60 levels)
- T60 = 0.2s (small room)
- T60 = 0.6s (medium room)
- T60 = 1.1s (large/reverberant room)
- Duration fixed at 1000ms, clean, no filter

**Total Conditions**: 8 + 6 + 3 + 3 = 20 per base clip

## 3. Train/Dev/Test Splits

**Split Ratios**:
- Train: 60%
- Dev: 20%
- Test: 20%

**Stratification**: Balanced by ground truth label (SPEECH/NONSPEECH)

**Sample Counts** (for 200 base clips per class):
- Train: ~4800 samples
- Dev: ~1600 samples
- Test: ~1600 samples

## 4. Model Training

### LoRA Fine-tuning

**Script**: `scripts/finetune_qwen_audio.py`

**Configuration**:
```bash
--seed 42
--num_epochs 3
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
--learning_rate 5e-5
--lora_r 16
--lora_alpha 32
```

**Base Model**: `Qwen/Qwen2-Audio-7B-Instruct`

**Target Modules**: Attention layers (q_proj, k_proj, v_proj, o_proj)

## 5. OPRO Prompt Optimization

### Algorithm

**Script**: `scripts/opro_classic_optimize.py`

OPRO (Optimization by PROmpting) uses an LLM to iteratively generate and refine prompts based on evaluation feedback.

**Configuration**:
```bash
--optimizer_llm "Qwen/Qwen2.5-7B-Instruct"
--optimizer_temperature 0.9
--num_iterations 30
--candidates_per_iter 5
--top_k 15
--early_stopping 7
--max_eval_samples 1000
--sample_strategy stratified
```

**Initial Prompts**: `config/initial_prompts_diverse.json`

**Process**:
1. Initialize with diverse seed prompts
2. Evaluate each prompt on dev set (stratified sample)
3. LLM generates new candidate prompts based on top-k performers
4. Repeat until convergence or early stopping

### Two OPRO Runs

1. **OPRO-BASE**: Optimize prompt for base Qwen2-Audio (no LoRA)
2. **OPRO-LoRA**: Optimize prompt for LoRA fine-tuned model

## 6. Evaluation

### Metrics

**Primary**: Balanced Accuracy (BA)
```
BA = (Speech_Accuracy + Nonspeech_Accuracy) / 2
```

**Secondary**: Per-class accuracy

### Evaluation Matrix

| # | Model | Prompt | Output File |
|---|-------|--------|-------------|
| 1 | BASE (no LoRA) | Original | eval_base_original.csv |
| 2 | BASE (no LoRA) | OPRO-BASE | eval_base_opro.csv |
| 3 | LoRA | Original | eval_lora_original.csv |
| 4 | LoRA | OPRO-LoRA | eval_lora_opro.csv |

### Original Prompt
```
Is this short clip speech or noise? Your answer should be SPEECH or NON-SPEECH.
```

## 7. SLURM Jobs

### Job Sequence

```bash
# 1. Generate dataset
sbatch slurm/generate_verified_dataset.job

# 2. Train LoRA
JOB1=$(sbatch --parsable slurm/train_lora_verified.job 42)

# 3. OPRO BASE (can run in parallel with training)
JOB2=$(sbatch --parsable slurm/opro_base_verified.job 42)

# 4. OPRO LoRA (depends on training)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 slurm/opro_lora_verified.job 42)

# 5. Evaluation (depends on both OPRO jobs)
sbatch --dependency=afterok:$JOB2:$JOB3 slurm/eval_verified.job 42
```

### Resource Requirements

| Job | GPUs | CPUs | Memory | Time |
|-----|------|------|--------|------|
| generate_verified_dataset | 1 | 8 | 48GB | 4h |
| train_lora_verified | 1 | 8 | 48GB | 16h |
| opro_base_verified | 1 | 8 | 48GB | 8h |
| opro_lora_verified | 1 | 8 | 48GB | 8h |
| eval_verified | 1 | 8 | 48GB | 6h |

## 8. Reproducibility

### Random Seeds
- Dataset generation: seed=42
- LoRA training: seed=42 (also tested 123, 456, 789, 2024)
- OPRO optimization: seed=42

### Environment
```bash
conda activate opro
pip install bitsandbytes peft accelerate transformers pandas scikit-learn
```

### Key Dependencies
- transformers >= 4.37.0
- peft >= 0.7.0
- torch >= 2.0.0
- soundfile
- librosa
- scipy

## 9. Results Summary

### Non-Verified Dataset Results

| Configuration | Balanced Accuracy |
|--------------|------------------|
| BASE + Original | 70.83% |
| BASE + OPRO | 71.67% |
| LoRA + Original | 78.54% |
| **LoRA + OPRO** | **86.88%** |

### OPRO-Discovered Prompts

**For BASE model**:
```
Identify spoken words: A) SPEECH B) NON-SPEECH.
```

**For LoRA model**:
```
Short: Speech? YES or NO.
```

## 10. Output Structure

```
results/
├── eval_verified_seed42/
│   ├── eval_base_original.csv
│   ├── eval_base_opro.csv
│   ├── eval_lora_original.csv
│   ├── eval_lora_opro.csv
│   ├── prompt_original.txt
│   ├── prompt_opro_base.txt
│   ├── prompt_opro_lora.txt
│   └── results_summary.json
├── opro_verified_base_seed42/
│   ├── best_prompt.txt
│   ├── history.json
│   └── optimization_log.txt
└── opro_verified_lora_seed42/
    ├── best_prompt.txt
    ├── history.json
    └── optimization_log.txt
```

---

*Document Version: 1.0*
*Created: 2025-11-30*
