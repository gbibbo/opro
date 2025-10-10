# Implementation Notes: Qwen2-Audio Speech Detection

**Date:** 2025-10-10
**Model:** Qwen2-Audio-7B-Instruct (4-bit quantization)
**Task:** Binary speech detection (SPEECH vs NONSPEECH)

---

## Summary

Successfully implemented and validated Qwen2-Audio for speech detection in short audio segments (20-1000ms) using:
- **Multiple choice prompting** (A/B/C/D format)
- **Noise padding to 2000ms** for optimal performance
- **4-bit quantization** for 8GB VRAM compatibility

**Final Performance:** 85% overall accuracy across all durations (20-1000ms)

---

## Key Findings

### 1. Prompt Strategy: Multiple Choice (A/B/C/D)

After testing 6 different prompt strategies, multiple choice format achieved the best results:

**System Prompt:**
```
You classify audio content.
```

**User Prompt:**
```
What best describes this audio?
A) Human speech or voice
B) Music
C) Noise or silence
D) Animal sounds

Answer with ONLY the letter (A, B, C, or D).
```

**Why it works:**
- Qwen has bias against binary YES/NO responses
- Multiple choice provides more context
- Clearer decision boundaries

### 2. Audio Padding Strategy

**Problem:** Qwen2-Audio performs poorly on audio segments <2000ms

**Solution:** Pad all segments to 2000ms with low-amplitude noise (0.0001)

**Implementation:**
```python
# Pad audio to 2000ms, centering original audio
[LOW_NOISE_LEFT] + [ORIGINAL_AUDIO] + [LOW_NOISE_RIGHT] = 2000ms total
```

**Results by duration (with 2000ms padding):**

| Duration | Accuracy | Notes |
|----------|----------|-------|
| 20ms | 53.3% | Challenging due to insufficient speech content |
| 40ms | 73.3% | Improvement with padding |
| 60ms | 83.3% | Good performance |
| 80ms | **96.7%** | Excellent - minimum reliable threshold |
| 100ms | 90.0% | Very good |
| 200ms | 93.3% | Excellent |
| 500ms | 93.3% | Excellent |
| 1000ms | **96.7%** | Excellent |

**Minimum reliable threshold:** ≥80ms (96.7% accuracy)

### 3. Critical Bug Fixes

**Bug 1: Response Decoding**
- **Problem:** Model output included full prompt + response, not just response
- **Cause:** Decoding all tokens instead of only generated ones
- **Fix:** Decode only tokens after input length
```python
input_length = inputs['input_ids'].shape[1]
generated_tokens = outputs[:, input_length:]
output_text = processor.batch_decode(generated_tokens, ...)[0]
```

**Bug 2: Processor Parameter**
- **Problem:** Missing `sampling_rate` parameter caused warnings
- **Fix:** Explicitly pass `sampling_rate=16000` to processor

**Bug 3: Dataset Contamination**
- **Problem:** ESC-50 dataset contained ambiguous sounds (human/animal)
- **Solution:** Filtered to 23 clean environmental sound categories (376 samples)

### 4. Performance Characteristics

**Hardware:** RTX 4070 Laptop (8GB VRAM)
- 4-bit quantization enabled
- Average latency: ~1.9s per sample (with 2000ms padding)
- Samples per minute: ~32

**Comparison with Silero-VAD:**
- Silero: ~100ms minimum, <100ms latency
- Qwen: ~80ms minimum, ~1900ms latency
- Qwen advantage: Interpretable, handles edge cases better
- Silero advantage: 19x faster

---

## Configuration

### Final Model Configuration

```python
from qsm.models import Qwen2AudioClassifier

model = Qwen2AudioClassifier(
    device="cuda",
    torch_dtype="float16",
    load_in_4bit=True,
    auto_pad=True,              # Enable automatic padding
    pad_target_ms=2000,         # Pad to 2000ms
    pad_noise_amplitude=0.0001, # Very low noise to avoid interference
)
```

### Default Prompts

The classifier uses validated multiple choice prompts by default. Custom prompts can be set:

```python
model.set_prompt(
    system_prompt="Your custom system prompt",
    user_prompt="Your custom user prompt"
)
```

---

## Evaluation Results

### Extended Evaluation (240 samples, 30 per duration)

**Overall Accuracy:** 85.0% (204/240 correct)

**Performance Tiers:**
- Excellent (≥95%): 80ms, 1000ms
- Good (80-94%): 60ms, 100ms, 200ms, 500ms
- Partial (60-79%): 40ms
- Poor (<60%): 20ms

**Key Insight:** The 12.5x improvement from 1000ms → 80ms minimum threshold is achieved through noise padding strategy.

---

## Lessons Learned

1. **LLM audio models need context:** Even with 40ms frame resolution, Qwen needs ~2000ms total duration for stable predictions

2. **Padding content doesn't matter:** Low-amplitude noise works as well as silence or repeated audio - the model uses temporal context, not content

3. **Prompt engineering is critical:** 50% accuracy with binary prompts → 85% with multiple choice

4. **Small test sets mislead:** Initial 3-sample tests showed 100% on some durations, but 30-sample tests revealed true performance

5. **Segmentation matters:** Some 1000ms samples consistently fail - likely due to excessive silence, overlapping speakers, or other quality issues

---

## Next Steps (Sprint 5+)

1. **Threshold Analysis:**
   - Generate psychometric curves (accuracy vs duration)
   - Compare with Silero-VAD baseline
   - Identify optimal duration ranges

2. **OPRO Optimization (Sprint 6):**
   - Use current config as baseline
   - Optimize prompts for <80ms segments
   - Target: 60-70% → 80%+ on very short segments

3. **Production Integration:**
   - Integrate with AVA-Speech dataset
   - Deploy in DIHARD evaluation pipeline
   - Benchmark against other speech detection methods

---

## Files

**Core Implementation:**
- `src/qsm/models/qwen_audio.py` - Main classifier wrapper
- `src/qsm/models/__init__.py` - Module exports

**Evaluation:**
- `scripts/evaluate_extended.py` - Full evaluation script (240 samples)
- `scripts/run_qwen_inference.py` - Basic inference script

**Results:**
- `results/qwen_extended_evaluation_with_padding.parquet` - Detailed results
- `results/qwen_extended_summary.parquet` - Summary statistics

**Documentation:**
- `README.md` - Project overview
- `IMPLEMENTATION_NOTES.md` - This file
- `SPRINT_4_SETUP.md` - Sprint 4 setup guide

---

## References

- Qwen2-Audio Paper: https://arxiv.org/abs/2407.10759
- HuggingFace Model: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
- VoxConverse Dataset: http://www.robots.ox.ac.uk/~vgg/data/voxconverse/
- ESC-50 Dataset: https://github.com/karolpiczak/ESC-50

---

**Status:** ✅ Implementation complete and validated. Ready for Sprint 5 (Threshold Analysis).
