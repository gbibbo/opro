# Fine-Tuning Qwen2-Audio for Speech Detection

This guide shows how to fine-tune Qwen2-Audio-7B-Instruct to improve speech detection accuracy on noisy audio clips.

## Problem Statement

The base Qwen2-Audio model works well on clean audio, but struggles with audio that has noise padding or low SNR. Our dataset (`snr_duration_crossed`) contains clips with 1s of content + 1s of noise padding, which confuses the model.

**Solution**: Fine-tune with LoRA/QLoRA on clean clips (without noise padding).

## Prerequisites

Install required packages:

```bash
pip install peft transformers accelerate bitsandbytes
```

## Step 1: Create Clean Dataset

Extract the center portion of audio clips (removing noise padding):

```bash
python scripts/create_clean_dataset.py
```

This will:
- Load metadata from `data/processed/snr_duration_crossed/metadata.csv`
- Filter to samples with SNR >= +10dB and duration >= 500ms
- Balance classes (equal SPEECH and NONSPEECH samples)
- Extract center portion of each clip (removing padding)
- Save cleaned clips to `data/processed/clean_clips/`
- Create train/test split (80/20)

**Output:**
- `data/processed/clean_clips/*.wav` - Cleaned audio files
- `data/processed/clean_clips/train_metadata.csv` - Training set
- `data/processed/clean_clips/test_metadata.csv` - Test set
- `data/processed/clean_clips/clean_metadata.csv` - Full dataset

Expected: ~2000-4000 balanced samples (depends on your dataset)

## Step 2: Fine-Tune the Model

Run fine-tuning with LoRA:

```bash
python scripts/finetune_qwen_audio.py
```

This will:
- Load Qwen2-Audio-7B-Instruct in 4-bit quantization
- Apply LoRA adapters (r=16, alpha=32)
- Train for 3 epochs on clean clips
- Save checkpoints to `checkpoints/qwen2_audio_speech_detection/`

**Training time**: ~2-4 hours on a single GPU (depends on dataset size)

**Configuration** (in `finetune_qwen_audio.py`):
- Batch size: 4
- Gradient accumulation: 4 (effective batch size = 16)
- Learning rate: 2e-4
- LoRA rank: 16
- Target modules: q_proj, v_proj, k_proj, o_proj

## Step 3: Test Fine-Tuned Model

Evaluate on test set:

```bash
python scripts/test_finetuned_model.py
```

This will:
- Load fine-tuned model from `checkpoints/qwen2_audio_speech_detection/final/`
- Run quick test on 10 samples
- If accuracy > 90%, run full evaluation

**Expected Results:**
- Clean clips: 95-100% accuracy
- Noisy clips (original): 70-85% accuracy (improvement from 50%)

## Step 4: Use Fine-Tuned Model in Production

To use the fine-tuned model with your existing code:

```python
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Load base model
base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct",
    load_in_4bit=True,
    device_map="auto",
)

# Load LoRA weights
model = PeftModel.from_pretrained(
    base_model,
    "checkpoints/qwen2_audio_speech_detection/final"
)

# Use with classifier
from src.qsm.models.qwen_audio import Qwen2AudioClassifier

classifier = Qwen2AudioClassifier(
    model_name="Qwen/Qwen2-Audio-7B-Instruct",
    device="cuda",
    load_in_4bit=True,
    constrained_decoding=True,
)

# Replace model with fine-tuned version
classifier.model = model

# Predict
result = classifier.predict("audio.wav")
print(f"Label: {result.label}, Confidence: {result.confidence}")
```

## Troubleshooting

### Low accuracy after fine-tuning (<70%)

Try:
- Train for more epochs (5-10)
- Increase LoRA rank (r=32 or r=64)
- Use more training data
- Check data quality (are labels correct?)

### Out of memory errors

Reduce:
- Batch size (e.g., batch_size=2)
- LoRA rank (e.g., r=8)
- Max audio length (e.g., max_audio_length=15)

### Model still fails on very noisy audio

This is expected. Fine-tuning helps but won't solve all cases with extreme noise (SNR < 0dB). Consider:
- Fine-tuning on noisy samples directly
- Using data augmentation (add more noise during training)
- Using a larger model (13B instead of 7B)

## Advanced: Fine-Tune on Noisy Data

If you want the model to handle noisy audio better:

1. **Don't extract clean clips** - use original `snr_duration_crossed` files
2. Modify `finetune_qwen_audio.py`:
   ```python
   config.train_csv = project_root / "data" / "processed" / "snr_duration_crossed" / "metadata.csv"
   ```
3. Filter to samples with SNR >= 0dB (not >= 10dB)
4. Train for more epochs (5-10)

This teaches the model to handle noise but may reduce accuracy on clean audio.

## Files Created

```
scripts/
├── create_clean_dataset.py      # Extract clean clips
├── finetune_qwen_audio.py        # Fine-tune with LoRA
└── test_finetuned_model.py       # Evaluate fine-tuned model

data/processed/clean_clips/
├── *.wav                         # Cleaned audio files
├── train_metadata.csv            # Training set
├── test_metadata.csv             # Test set
└── clean_metadata.csv            # Full dataset

checkpoints/qwen2_audio_speech_detection/
├── checkpoint-200/               # Intermediate checkpoints
├── checkpoint-400/
└── final/                        # Final LoRA weights
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Next Steps

After fine-tuning:
1. ✅ Test on clean clips (should be 95-100%)
2. Test on original noisy clips (should improve from 50% to 70-85%)
3. Deploy fine-tuned model in your application
4. Monitor performance on real data
5. Collect failure cases for further fine-tuning

## References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
