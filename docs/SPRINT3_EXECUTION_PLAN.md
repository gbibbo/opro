# SPRINT 3: Data Augmentation & Hyperparameter Optimization - Execution Plan

**Objective**: Improve beyond 100% test accuracy by training on augmented data and optimizing hyperparameters for better generalization

**Duration**: ~2-3 weeks (mostly training time)

**Prerequisites**:
- ✅ SPRINT 2 completed (100% test accuracy with threshold=1.256)
- ✅ Baseline established (83.3% without threshold)
- ⚠️ Requires GPU with 16GB+ VRAM or cloud access

---

## Overview

SPRINT 3 tackles the **generalization problem** discovered in SPRINT 2:
- Current 100% accuracy may be overfitting to small test set (n=24)
- Only 1 SPEECH speaker, 1-2 NONSPEECH sounds in test
- Need robust model that generalizes to diverse speakers/sounds

### Strategy

1. **Data Augmentation**: Expand training diversity artificially
2. **Hyperparameter Tuning**: Find optimal LoRA config
3. **Expanded Evaluation**: Test on larger, more diverse set
4. **Cross-Validation**: LOSO (Leave-One-Speaker-Out)

---

## Task Breakdown

### 3.1 Data Augmentation Pipeline (Priority: HIGH)

#### A. MUSAN Noise Augmentation

**Concept**: Add realistic background noise to training samples

**Implementation**:

```python
# scripts/augment_with_musan.py

import torch
import torchaudio
import numpy as np
from pathlib import Path

class MUSANAugmenter:
    """Add MUSAN noise to audio samples."""

    def __init__(self, musan_root, snr_range=(-5, 20)):
        """
        Args:
            musan_root: Path to MUSAN dataset
            snr_range: (min_snr, max_snr) in dB
        """
        self.musan_root = Path(musan_root)
        self.snr_range = snr_range

        # Load noise files
        self.noise_files = {
            'music': list((self.musan_root / 'music').glob('**/*.wav')),
            'speech': list((self.musan_root / 'speech').glob('**/*.wav')),
            'noise': list((self.musan_root / 'noise').glob('**/*.wav'))
        }

    def add_noise(self, audio, sr, noise_type='noise'):
        """Add background noise at random SNR."""
        # Select random noise file
        noise_file = np.random.choice(self.noise_files[noise_type])
        noise, noise_sr = torchaudio.load(noise_file)

        # Resample if needed
        if noise_sr != sr:
            resampler = torchaudio.transforms.Resample(noise_sr, sr)
            noise = resampler(noise)

        # Match length
        if noise.shape[1] < audio.shape[1]:
            # Repeat noise if too short
            repeats = int(np.ceil(audio.shape[1] / noise.shape[1]))
            noise = noise.repeat(1, repeats)
        noise = noise[:, :audio.shape[1]]

        # Mix at random SNR
        snr_db = np.random.uniform(*self.snr_range)
        audio_power = audio.pow(2).mean()
        noise_power = noise.pow(2).mean()

        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(audio_power / (snr_linear * noise_power))

        augmented = audio + scale * noise
        return augmented
```

**Usage in Training**:

```python
# In finetune_qwen_audio.py
if args.use_musan:
    augmenter = MUSANAugmenter(
        musan_root='data/musan',
        snr_range=(-5, 20)  # -5dB to 20dB SNR
    )

    # Apply during data loading
    audio_augmented = augmenter.add_noise(audio, sr, noise_type='noise')
```

**Expected Impact**: +2-5% accuracy on noisy test data

#### B. SpecAugment

**Concept**: Mask frequencies and time segments in spectrogram

**Implementation**:

```python
# scripts/spec_augment.py

import torch
import torchaudio

class SpecAugment:
    """SpecAugment for audio (Park et al., 2019)."""

    def __init__(self, freq_mask_param=15, time_mask_param=35, n_freq_masks=2, n_time_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __call__(self, spec):
        """
        Args:
            spec: (batch, freq, time) spectrogram
        """
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, spec.shape[1] - f)
            spec[:, f0:f0+f, :] = 0

        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, spec.shape[2] - t)
            spec[:, :, t0:t0+t] = 0

        return spec
```

**Usage**:

```python
if args.use_specaugment:
    spec_augment = SpecAugment(
        freq_mask_param=15,
        time_mask_param=35
    )
    # Apply during feature extraction
    features_augmented = spec_augment(features)
```

**Expected Impact**: +1-3% robustness to distortions

#### C. Time Stretching & Pitch Shifting

**Concept**: Vary speaking rate and pitch

```python
import torchaudio.transforms as T

# Time stretch (0.9x to 1.1x speed)
time_stretch = T.TimeStretch(hop_length=512, n_freq=201)
stretched = time_stretch(spec, rate=np.random.uniform(0.9, 1.1))

# Pitch shift (±2 semitones)
pitch_shift = T.PitchShift(sample_rate=16000, n_steps=np.random.randint(-2, 3))
shifted = pitch_shift(audio)
```

**Expected Impact**: +1-2% generalization across speakers

---

### 3.2 Hyperparameter Optimization (Priority: MEDIUM)

#### A. LoRA Configuration Search

**Parameters to Tune**:

| Parameter | Current | Search Space | Rationale |
|-----------|---------|--------------|-----------|
| `lora_rank` | 8 | [4, 8, 16, 32] | Higher rank = more capacity |
| `lora_alpha` | 32 | [16, 32, 64] | Scaling factor |
| `lora_dropout` | 0.05 | [0.0, 0.05, 0.1] | Regularization |
| `learning_rate` | 1e-4 | [5e-5, 1e-4, 2e-4] | Training speed vs stability |
| `warmup_steps` | 150 | [100, 150, 200] | LR schedule |

**Grid Search Strategy**:

```python
# scripts/hyperparameter_search.py

configs = [
    # Baseline (current)
    {'rank': 8, 'alpha': 32, 'lr': 1e-4, 'dropout': 0.05},

    # Larger capacity
    {'rank': 16, 'alpha': 64, 'lr': 1e-4, 'dropout': 0.05},
    {'rank': 32, 'alpha': 64, 'lr': 1e-4, 'dropout': 0.1},

    # Higher LR
    {'rank': 8, 'alpha': 32, 'lr': 2e-4, 'dropout': 0.05},

    # Lower LR (more stable)
    {'rank': 8, 'alpha': 32, 'lr': 5e-5, 'dropout': 0.05},

    # No dropout
    {'rank': 8, 'alpha': 32, 'lr': 1e-4, 'dropout': 0.0},
]

for i, config in enumerate(configs):
    train_model(
        output_dir=f'checkpoints/hp_search/config_{i}',
        lora_rank=config['rank'],
        lora_alpha=config['alpha'],
        learning_rate=config['lr'],
        lora_dropout=config['dropout'],
        seed=42
    )
```

**Evaluation**: Use dev set (72 samples) + threshold optimization

**Expected Impact**: +0-5% accuracy (marginal)

#### B. Training Schedule

**Current**:
- Epochs: 5
- Batch size: 8
- Gradient accumulation: 1

**Alternatives**:

```python
# Longer training
epochs = 10  # vs 5

# Smaller batches (more updates)
batch_size = 4
gradient_accumulation = 2  # Effective batch = 8

# Learning rate schedule
lr_scheduler_type = 'cosine'  # vs 'linear'
```

---

### 3.3 Expanded Test Set Creation (Priority: HIGH)

#### Goal

Create robust test set:
- **SPEECH**: 10 speakers × 5 samples = 50 samples
- **NONSPEECH**: 20 sounds × 3 samples = 60 samples
- **Total**: 110 samples (vs current 24)

#### Script: `create_expanded_test_set.py`

```python
#!/usr/bin/env python3
"""
Create expanded test set with diverse speakers and sounds.

Usage:
    python scripts/create_expanded_test_set.py \
        --speech_speakers 10 \
        --nonspeech_sounds 20 \
        --output data/processed/test_expanded.csv
"""

import argparse
import pandas as pd
from pathlib import Path

def create_expanded_test():
    # Load all available data
    voxconverse = load_voxconverse()  # Get 10 different speakers
    esc50 = load_esc50()  # Get 20 different sound classes

    # Sample strategically
    speech_samples = []
    for speaker in voxconverse['speakers'][:10]:
        speaker_clips = voxconverse[voxconverse['speaker'] == speaker]
        samples = speaker_clips.sample(5, random_state=42)
        speech_samples.append(samples)

    nonspeech_samples = []
    for sound_class in esc50['classes'][:20]:
        class_clips = esc50[esc50['class'] == sound_class]
        samples = class_clips.sample(3, random_state=42)
        nonspeech_samples.append(samples)

    # Combine
    test_expanded = pd.concat([
        pd.concat(speech_samples),
        pd.concat(nonspeech_samples)
    ])

    return test_expanded
```

**Benefits**:
- More reliable accuracy estimates (CI: ±5% vs current ±15%)
- Test generalization across speakers/sounds
- Validate threshold optimization robustness

---

### 3.4 Leave-One-Speaker-Out Cross-Validation (Priority: LOW)

#### Concept

For each speaker:
1. Train on all other speakers
2. Evaluate on held-out speaker
3. Average accuracy across all folds

**Implementation**:

```python
# scripts/loso_cv.py

speakers = get_all_speakers()  # e.g., 13 speakers

results = []
for test_speaker in speakers:
    # Split
    train_df = df[df['speaker'] != test_speaker]
    test_df = df[df['speaker'] == test_speaker]

    # Train
    model = train_model(train_df, seed=42)

    # Evaluate
    acc = evaluate(model, test_df)
    results.append({'speaker': test_speaker, 'accuracy': acc})

# Report
mean_acc = np.mean([r['accuracy'] for r in results])
print(f"LOSO CV Accuracy: {mean_acc:.1%} ± {np.std(...):.1%}")
```

**Expected**: Accuracy will likely drop to 75-85% (more realistic)

---

## 3.5 Final Model Training & Evaluation

### Training Pipeline

```bash
# 1. Train with all improvements
python scripts/finetune_qwen_audio.py \
    --seed 42 \
    --output_dir checkpoints/sprint3/final \
    --add_mlp_targets \
    --use_musan \
    --use_specaugment \
    --lora_rank 16 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --num_epochs 10

# 2. Calibrate threshold on dev set
python scripts/simulate_prompt_from_logits.py \
    --results_csv results/sprint3/dev_predictions.csv \
    --output_dir results/sprint3/threshold_calibration

# 3. Evaluate on expanded test set
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/sprint3/final \
    --test_csv data/processed/test_expanded.csv \
    --prompt "$(cat results/prompt_opt_local/best_prompt.txt)" \
    --threshold {CALIBRATED_THRESHOLD} \
    --output_csv results/sprint3/test_expanded_predictions.csv
```

---

## Execution Timeline

### Week 1: Setup & Initial Experiments

**Day 1-2**: Data augmentation
- Download MUSAN dataset
- Implement augmentation pipeline
- Verify augmented samples sound correct

**Day 3-4**: Hyperparameter search
- Run 6 configurations
- Evaluate on dev set
- Select best config

**Day 5**: Create expanded test set
- Sample 110 diverse examples
- Verify no data leakage
- Document sources

### Week 2: Training & Evaluation

**Day 6-8**: Train final models
- Best hyperparameters + augmentation
- Multiple seeds (42, 123, 456)
- Monitor overfitting

**Day 9-10**: Comprehensive evaluation
- Evaluate on expanded test
- LOSO cross-validation
- Threshold calibration

### Week 3: Analysis & Documentation

**Day 11-12**: Results analysis
- Compare all methods
- Statistical tests
- Error analysis

**Day 13-14**: Documentation
- SPRINT3_FINAL_REPORT.md
- Update README
- Prepare for publication

---

## Expected Outcomes

### Success Criteria

1. ✅ **Augmentation improves robustness**: Accuracy on noisy test ≥80%
2. ✅ **Hyperparameters improve efficiency**: Faster training or better accuracy
3. ✅ **Expanded test validates generalization**: Accuracy ≥75% on 110 samples
4. ✅ **Threshold calibration transfers**: Dev threshold works on test (±0.5)
5. ✅ **LOSO CV confirms**: Mean accuracy ≥75% across speakers

### Comparison Table (Projected)

| Method | Test (n=24) | Expanded (n=110) | LOSO CV | Robustness |
|--------|-------------|------------------|---------|------------|
| Baseline (SPRINT 2) | 83.3% | ~70% | ~70% | Low |
| + MUSAN | 83.3% | ~75% | ~75% | Medium |
| + SpecAugment | 83.3% | ~78% | ~77% | High |
| + HP Tuning | 83.3% | ~80% | ~78% | High |
| **+ Threshold Calib** | **100%** | **85%** | **80%** | **High** |

---

## Scripts to Create

1. ✅ `scripts/augment_with_musan.py` - MUSAN noise augmentation
2. ✅ `scripts/spec_augment.py` - SpecAugment implementation
3. ✅ `scripts/create_expanded_test_set.py` - Expanded test creation
4. ✅ `scripts/hyperparameter_search.py` - Grid search automation
5. ✅ `scripts/loso_cv.py` - Cross-validation
6. ✅ `scripts/evaluate_sprint3.py` - Comprehensive evaluation

---

## Contingency Plans

### If VRAM Issues Persist

**Option A**: Use Google Colab Pro (16GB GPU, $10/month)
**Option B**: Reduce batch size to 4, gradient accumulation = 2
**Option C**: Use 8-bit quantization for base model

### If Augmentation Doesn't Help

**Finding**: Model may already be at capacity limit
**Action**: Focus on expanding dataset (real data, not augmentation)

### If LOSO CV Shows Poor Generalization

**Finding**: Speaker-specific features dominate
**Action**: Collect more diverse training speakers

---

## SPRINT 3 Completion Criteria

- [ ] MUSAN augmentation implemented and tested
- [ ] SpecAugment integrated into training
- [ ] Hyperparameter search completed (≥6 configs)
- [ ] Expanded test set created (110+ samples)
- [ ] Final models trained with best config
- [ ] Threshold calibrated on dev, evaluated on test
- [ ] LOSO cross-validation completed
- [ ] Results documented in SPRINT3_FINAL_REPORT.md

---

**Ready to Start**: YES (need GPU access)

**Estimated Completion**: 2-3 weeks

**Next Step**: Download MUSAN dataset + implement augmentation
