# Changelog

All notable changes to the Qwen2-Audio Fine-Tuning for Speech Detection project.

## [Unreleased]

### Added
- Loss masking implementation (compute loss only on A/B token)
  - Expected +5-10% accuracy improvement
  - Better gradient signal during training

### Planned
- Dataset scaling to 1-3k clips with factorial balance
- NONSPEECH hygiene validation with WebRTC VAD
- SpecAugment for training robustness
- OPRO prompt optimization on dev set

---

## [v0.3.0] - 2025-10-19 - Phase 2 Complete

### Added
- **Constrained A/B decoding** in test script
  - Detects all token variants for "A" and "B"
  - Uses `prefix_allowed_tokens_fn` to force only A/B output
  - Eliminates tokenizer variability
- **Logits-based confidence scores**
  - Computes prob(A) and prob(B) via softmax over allowed tokens
  - Reports confidence metrics (overall, correct, wrong)
  - Enables future threshold optimization

### Changed
- **Peak normalization** (replaced RMS normalization)
  - Normalizes by peak amplitude (0.9 with 3dB headroom)
  - **Preserves SNR** as discriminative feature
  - RMS range: 0.053-0.203 (~4x) confirms SNR preservation
- **Improved sampling_rate handling**
  - Explicit `sampling_rate=target_sr` in all processor() calls
  - Fixes applied to training, test, and inference

### Fixed
- **sampling_rate warnings eliminated** in training
  - Training now completes with zero warnings
  - Prevents silent feature extraction errors
- **Confidence score calculation bug**
  - Fixed tensor dimension error in logits aggregation
  - Now correctly computes prob_A and prob_B

### Results
- **Accuracy**: 62.5% (20/32) on challenging test set
- **SPEECH**: 62.5% (10/16)
- **NONSPEECH**: 62.5% (10/16)
- **Training Loss**: 8.69 (improved from 10.17, -14.5%)
- **Perfect balance**: No bias towards SPEECH or NONSPEECH
- **Confidence gap**: 0.076 (correct=0.651, wrong=0.575)

### Technical Details
- Model: Qwen2-Audio-7B-Instruct
- Quantization: 4-bit (BitsAndBytes NF4)
- LoRA: r=16, α=32, 0.25% trainable params (20.7M/8.4B)
- Dataset: 160 clips (128 train, 32 test)
- Durations: 200-1000ms
- SNR range: 0-20dB

---

## [v0.2.0] - 2025-10-18 - Audio Integration Fixes

### Added
- **Peak normalization script** (`create_normalized_dataset.py`)
  - Normalizes by peak instead of RMS
  - Preserves relative energy differences (SNR)
- **Inspection script** (`inspect_clean_clips.py`)
  - Analyzes RMS, activity levels, duration distribution

### Changed
- **Audio parameter name** in processor
  - Changed from `audios=[audio]` to `audio=[audio]`
  - Critical fix - was causing audio to be ignored
- **Disabled custom auto-padding**
  - Let processor handle padding natively
  - Removed noise padding that diluted signal

### Fixed
- **Audio features not arriving at model**
  - Model was only seeing text, not audio
  - Fixed by using correct parameter name `audio=`
- **Low RMS in processed clips**
  - Custom padding was adding 70% silence
  - Removed custom padding logic

### Results
- **Accuracy**: 65.6% (21/32) with RMS normalization
  - SPEECH: 68.8%
  - NONSPEECH: 62.5%
- **Improvement**: +15.6% over 50% baseline

---

## [v0.1.0] - 2025-10-17 - Initial Fine-Tuning Pipeline

### Added
- **Clean dataset creation** (`create_clean_dataset.py`)
  - Extracts center portion of clips (removes noise padding)
  - Filters by SNR ≥10dB, duration ≥500ms
  - Balances classes (equal SPEECH/NONSPEECH)
  - 80/20 train/test split
- **LoRA fine-tuning script** (`finetune_qwen_audio.py`)
  - 4-bit quantization with BitsAndBytes
  - LoRA adapters (r=16, α=32)
  - 3 epochs, batch size 4, gradient accumulation 4
- **Evaluation script** (`test_finetuned_model.py`)
  - Quick test on 10 samples
  - Full evaluation if >90% accuracy

### Technical Details
- Dataset: 40 clips (32 train, 8 test) with strict filters
- Training: ~2-4 hours on single GPU
- Storage: ~500MB for model + adapters

### Results
- Initial training showed learning (loss decreased)
- Baseline accuracy: 50% (model always predicting "B")
- Discovered audio wasn't being processed correctly

---

## [v0.0.1] - 2025-10-16 - Project Setup

### Added
- Project structure
- Initial Qwen2AudioClassifier wrapper
- Basic A/B prompting format
- Constrained decoding option (not yet working)

### Known Issues
- Audio parameter incorrect (`audios=` instead of `audio=`)
- Custom auto-padding causing low RMS
- No sampling_rate passed to feature extractor
- Model stuck at 50% accuracy (always predicts B)

---

## Version History Summary

| Version | Date | Accuracy | Key Achievement |
|---------|------|----------|-----------------|
| v0.0.1 | 2025-10-16 | 50.0% | Initial setup |
| v0.1.0 | 2025-10-17 | 50.0% | Fine-tuning pipeline |
| v0.2.0 | 2025-10-18 | 65.6% | Audio integration fixes |
| v0.3.0 | 2025-10-19 | 62.5% | Peak norm + constrained decoding |
| **Next** | TBD | **~75%** | **Loss masking** |

---

## Breaking Changes

### v0.3.0
- `create_normalized_dataset.py` now uses **peak normalization** instead of RMS
  - Re-run normalization step if using old pipeline
  - Old RMS-normalized clips will have destroyed SNR info

### v0.2.0
- Processor call signature changed
  - Must use `audio=[audio]` not `audios=[audio]`
  - Breaking change for any custom inference code

---

## Deprecations

### v0.3.0
- `normalize_audio_rms()` marked as legacy
  - Use `normalize_audio_peak()` instead
  - RMS normalization destroys SNR as discriminative feature

### v0.2.0
- Custom auto-padding removed
  - `auto_pad` parameter deprecated
  - Processor handles padding natively with `padding=True`

---

## Migration Guide

### From v0.2.0 to v0.3.0

1. **Re-generate normalized dataset**:
   ```bash
   python scripts/create_normalized_dataset.py
   ```
   This will apply peak normalization (preserves SNR)

2. **Update test scripts** to use constrained decoding:
   ```python
   from scripts.test_normalized_model import get_ab_token_ids, make_ab_prefix_fn

   ids_A, ids_B = get_ab_token_ids(processor.tokenizer)
   prefix_fn = make_ab_prefix_fn(ids_A, ids_B)

   outputs = model.generate(
       **inputs,
       prefix_allowed_tokens_fn=prefix_fn,
       output_scores=True,
       return_dict_in_generate=True,
   )
   ```

3. **Add confidence computation**:
   ```python
   scores = outputs.scores[0]
   logits_A = scores[0, ids_A]
   logits_B = scores[0, ids_B]
   all_logits = torch.cat([logits_A, logits_B])
   probs_all = torch.softmax(all_logits, dim=0)
   prob_A = probs_all[:len(ids_A)].sum().item()
   prob_B = probs_all[len(ids_A):].sum().item()
   ```

### From v0.1.0 to v0.2.0

1. **Update processor calls**:
   ```python
   # OLD (WRONG)
   inputs = processor(text=text, audios=[audio], ...)

   # NEW (CORRECT)
   inputs = processor(text=text, audio=[audio], sampling_rate=16000, ...)
   ```

2. **Remove auto-padding**:
   ```python
   # OLD
   classifier = Qwen2AudioClassifier(..., auto_pad=True)

   # NEW
   classifier = Qwen2AudioClassifier(...)  # auto_pad removed
   ```

3. **Re-create clean dataset** with relaxed filters:
   ```bash
   python scripts/create_clean_dataset.py  # Now uses SNR≥0dB, duration≥200ms
   ```

---

## Contributors

- **Gabi** - Project lead, implementation, debugging
- **Claude (AI Assistant)** - Code reviews, optimization suggestions, documentation

---

## License

[Specify your license here]

---

**Last Updated**: 2025-10-19
**Current Version**: v0.3.0
**Next Release**: v0.4.0 (Loss Masking)
