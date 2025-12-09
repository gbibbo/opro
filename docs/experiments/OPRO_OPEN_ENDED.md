# OPRO with Open-Ended Prompts

## Overview

This document describes the enhanced OPRO (Optimization by PROmpting) implementation that supports **both constrained and open-ended prompts** for speech detection.

## Motivation

The original OPRO implementation restricted prompts to **label-style instructions** that explicitly require `SPEECH` or `NON-SPEECH` keywords. This ensured deterministic mapping of model outputs to binary labels.

However, this restriction prevented OPRO from exploring **open-ended prompts** like:
- "What do you hear in this audio?"
- "Describe the sound in this clip."
- "Is there human speech here?"

Open-ended prompts may elicit more natural responses from the model and could potentially improve performance, especially on degraded audio where explicit binary choices might be confusing.

## Implementation

### 1. Prompt Sanitization

Modified `sanitize_prompt()` to accept an `allow_open_ended` parameter:

```python
def sanitize_prompt(prompt: str, allow_open_ended: bool = False) -> tuple[str, bool]:
    """
    Sanitize and validate prompt candidate.

    Args:
        prompt: Prompt text to validate
        allow_open_ended: If True, allow open-ended prompts without explicit SPEECH/NON-SPEECH keywords

    Returns:
        (cleaned_prompt, is_valid)
    """
```

**Constrained mode** (default, `allow_open_ended=False`):
- Requires explicit `SPEECH` and `NON-SPEECH` (or `NONSPEECH`) keywords
- Ensures deterministic output parsing

**Open-ended mode** (`allow_open_ended=True`):
- Allows questions about audio without explicit keywords
- Validates that prompt references audio/sound
- Accepts keywords: AUDIO, SOUND, HEAR, LISTEN, WHAT, DESCRIBE, IDENTIFY, CLASSIFY

### 2. Response Normalization

Enhanced `normalize_to_binary()` to handle open-ended responses via **synonym matching**:

**Priority Order**:
1. **Exact verbalizers**: `SPEECH`, `NONSPEECH`, `NON-SPEECH`, `NO SPEECH`
2. **Letter mapping**: A/B/C/D (if provided)
3. **Yes/No**: Word-boundary matched to avoid false positives
4. **Synonyms** (for open-ended):
   - **SPEECH**: voice, voices, talking, spoken, speaking, speaker, conversation, words, utterance, vocal, dialogue, syllables, phonemes
   - **NONSPEECH**: music, musical, song, melody, instrumental, beep, tone, noise, static, hiss, white noise, silence, quiet, nothing, ambient, environmental, background, click, clock, tick

**Improvements**:
- Check `NONSPEECH` before `SPEECH` to avoid substring matching issues
- Use word boundaries for Yes/No patterns (avoids "SI" matching "SILENCE")
- Robust handling of negations

### 3. OPRO Meta-Prompt

Updated `build_meta_prompt()` to include open-ended examples when enabled:

```
STYLE EXAMPLES (use these as inspiration for DIVERSITY):
- Constrained Binary: "Does this audio contain human speech? Answer: SPEECH or NON-SPEECH."
- A/B Choice: "Choose: A) SPEECH B) NON-SPEECH. Answer with A or B."
- Open Question: "What do you hear in this audio?"
- Open Identification: "Describe what sound is in this clip."
- Open Classification: "Listen carefully and tell me what type of audio this is."
- Open Yes/No: "Is there human speech in this audio?"
- Minimal Open: "What is this sound?"
- Contextual Open: "This may be noisy or brief. What do you hear?"

NOTE: For OPEN-ENDED prompts (questions without explicit SPEECH/NON-SPEECH keywords):
- Responses mentioning "speech", "voice", "talking", "spoken" → mapped to SPEECH
- Responses mentioning "music", "noise", "silence", "beep" → mapped to NONSPEECH
- The model's answer will be automatically classified using synonym matching
```

### 4. Command-Line Interface

Added `--allow_open_ended` flag:

```bash
# Constrained mode only (default)
python scripts/opro_classic_optimize.py \
    --manifest data/processed/expanded_4conditions_verified/dev_metadata.csv \
    --output_dir results/opro_constrained \
    --checkpoint checkpoints/qwen_lora_verified_seed42/final

# Open-ended + Constrained mode
python scripts/opro_classic_optimize.py \
    --manifest data/processed/expanded_4conditions_verified/dev_metadata.csv \
    --output_dir results/opro_open_ended \
    --checkpoint checkpoints/qwen_lora_verified_seed42/final \
    --allow_open_ended
```

## Testing

Comprehensive test suite in `scripts/test_open_ended_normalization.py`:

```bash
python scripts/test_open_ended_normalization.py
```

**Test Cases** (20 total):
- ✅ 7 speech responses ("I hear someone talking", "There's a human voice", etc.)
- ✅ 7 non-speech responses ("I hear music", "Silence", "Background noise", etc.)
- ✅ 4 constrained responses ("SPEECH", "NON-SPEECH", "A) SPEECH", etc.)
- ✅ 2 ambiguous responses ("I don't know", "Unclear" → None)

**Results**: 100% pass rate (20/20)

## Metrics Consistency

All existing metrics are preserved and calculated correctly:
- **BA_clip**: Balanced accuracy at clip level
- **BA_conditions**: Macro-average BA across psychoacoustic conditions
- **BA by dimension**: Duration, SNR, Filter, Reverb
- **Confusion matrix**: Speech/Nonspeech accuracies
- **Reward function**: `R = BA_clip + 0.25×BA_conditions - 0.05×len/100`

## Example Prompts

### Constrained Prompts (default mode)
```
"Does this audio contain human speech? Answer: SPEECH or NON-SPEECH."
"Choose: A) SPEECH B) NON-SPEECH"
"Classify as SPEECH or NON-SPEECH"
```

### Open-Ended Prompts (with --allow_open_ended)
```
"What do you hear in this audio?"
"Is there human voice in this clip?"
"Describe the sound"
"What type of audio is this?"
```

### Hybrid Approach
With `--allow_open_ended`, OPRO can explore **both** constrained and open-ended prompts, allowing the optimizer to discover which approach works better for each psychoacoustic condition.

## Expected Outcomes

### Potential Advantages of Open-Ended Prompts
1. **More natural responses** from the model (less constrained)
2. **Better handling of degraded audio** (less binary, more descriptive)
3. **Richer signal** for OPRO to optimize (more diverse responses)

### Potential Disadvantages
1. **Higher parsing ambiguity** (synonym matching is probabilistic)
2. **Lower confidence** (0.8 vs 1.0 for exact matches)
3. **Longer responses** (higher token cost, more variability)

### Hypothesis
Open-ended prompts may perform better on **ambiguous** or **degraded** audio (low SNR, short duration) where binary choices force the model into uncertain predictions.

## Future Work

1. **Expand synonym dictionaries** with domain-specific terms
2. **Add multilingual support** for Spanish, French, etc.
3. **Semantic similarity** using embeddings instead of keyword matching
4. **Confidence calibration** for synonym-based predictions
5. **A/B testing** on verified dataset to compare constrained vs open-ended

## References

- Original OPRO paper: [Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)
- Qwen2-Audio: [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)
- Normalization code: `src/qsm/utils/normalize.py`
- OPRO implementation: `scripts/opro_classic_optimize.py`

---

*Created: 2025-12-08*
*Version: 1.0*
