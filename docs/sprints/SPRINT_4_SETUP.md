# Sprint 4: Qwen2-Audio Inference Setup

## Status
✅ **Infrastructure Complete** - Ready for GPU inference

## What's Been Implemented

### 1. Model Wrapper (`src/qsm/models/qwen_audio.py`)
- `Qwen2AudioClassifier`: Full wrapper for Qwen2-Audio models
- Supports binary SPEECH/NONSPEECH classification
- Customizable prompts
- Automatic response parsing
- Latency tracking

### 2. Inference Script (`scripts/run_qwen_inference.py`)
- Command-line interface for batch evaluation
- Supports all segment datasets
- Generates detailed metrics (accuracy, F1, precision, recall)
- Saves results to parquet for analysis
- Optional segment limiting for testing

### 3. Dataset Ready
- **1,016 clean segments** (640 SPEECH + 376 NONSPEECH)
- All durations: 20, 40, 60, 80, 100, 200, 500, 1000 ms
- Organized by dataset (AVA-Speech, VoxConverse, ESC-50)

## Hardware Requirements

### Minimum (for 7B model):
- **GPU**: NVIDIA GPU with ≥16GB VRAM (RTX 3090, A100, etc.)
- **RAM**: 32GB system RAM
- **Storage**: ~15GB for model weights

### Recommended:
- **GPU**: NVIDIA A100 (40GB) or better
- **RAM**: 64GB system RAM
- **CUDA**: Version 11.8 or higher

## Running Inference

### 1. Quick Test (2 segments)
```bash
# Test on AVA-Speech with GPU
python scripts/run_qwen_inference.py \
    --segments-dir data/segments/ava_speech/train \
    --limit 2 \
    --device cuda \
    --dtype float16

# Expected output:
# - Model loads in ~2-3 minutes
# - Inference: ~5-10 seconds per segment
# - Results saved to results/qwen_inference/ava_speech/
```

### 2. Full Dataset Evaluation

#### AVA-Speech (320 SPEECH segments)
```bash
python scripts/run_qwen_inference.py \
    --segments-dir data/segments/ava_speech/train \
    --device cuda \
    --dtype float16
```

#### VoxConverse (320 SPEECH segments)
```bash
python scripts/run_qwen_inference.py \
    --segments-dir data/segments/voxconverse/dev \
    --device cuda \
    --dtype float16
```

#### ESC-50 Clean (376 NONSPEECH segments)
```bash
python scripts/run_qwen_inference.py \
    --segments-dir data/segments/esc50/nonspeech \
    --device cuda \
    --dtype float16
```

### 3. Expected Runtime
- **Per segment**: 3-10 seconds (depending on GPU and duration)
- **Full dataset (1,016 segments)**: ~1.5-3 hours total

## Command-Line Options

```bash
python scripts/run_qwen_inference.py --help

Options:
  --segments-dir PATH          Directory with segments and metadata [REQUIRED]
  --metadata-file STR          Metadata filename (default: segments_metadata.parquet)
  --output-dir PATH            Output directory (default: results/qwen_inference)
  --model-name STR             HuggingFace model (default: Qwen/Qwen2-Audio-7B-Instruct)
  --device {cuda,cpu}          Device (default: cuda)
  --dtype {auto,float16,float32}  Model precision (default: auto)
  --limit INT                  Max segments to process (for testing)
```

## Output Format

Results are saved as parquet files with the following schema:
```python
{
    "segment_id": int,           # Original segment index
    "true_label": str,           # Ground truth ("SPEECH" or "NONSPEECH")
    "pred_label": str,           # Prediction ("SPEECH", "NONSPEECH", or "UNKNOWN")
    "confidence": float,         # Confidence score (0.0-1.0)
    "latency_ms": float,         # Inference time in milliseconds
    "duration_ms": int,          # Segment duration
    "dataset": str,              # Source dataset
    "condition": str,            # Audio condition/category
    "raw_output": str,           # Raw model text output
}
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Use float16 precision
python scripts/run_qwen_inference.py ... --dtype float16

# Or use smaller model (if available)
python scripts/run_qwen_inference.py ... --model-name Qwen/Qwen2-Audio-7B
```

### Slow Inference
- Ensure CUDA is properly installed and detected
- Use float16 precision for 2x speedup
- Process in batches with --limit flag

### Model Download Issues
```bash
# Pre-download model
huggingface-cli download Qwen/Qwen2-Audio-7B-Instruct

# Use local cache
export HF_HOME=/path/to/huggingface/cache
```

## Next Steps After Inference

1. **Compare with Silero-VAD baseline**
   ```python
   import pandas as pd

   # Load results
   qwen = pd.read_parquet('results/qwen_inference/ava_speech/qwen2_audio_*.parquet')
   silero = pd.read_parquet('results/vad_baseline/ava_speech/silero_vad_*.parquet')

   # Compare metrics...
   ```

2. **Analyze performance by duration**
   - Which durations does Qwen2-Audio handle best?
   - Compare with Silero-VAD's performance curve

3. **Generate psychometric curves**
   - Plot accuracy vs duration
   - Identify temporal threshold for reliable detection

4. **Sprint 5: Threshold Analysis & Optimization**

## Model Architecture Notes

### Qwen2-Audio-7B-Instruct
- **Parameters**: 7 billion
- **Audio Encoder**: Whisper-like architecture
- **Context Window**: Handles audio up to several seconds
- **Training**: Multi-task audio understanding (speech, music, events)

### Why This Model?
- State-of-the-art audio understanding
- Instruction-following capability (good for prompting)
- Open-source and accessible
- Designed for diverse audio tasks (not just ASR)

## Alternative Models (if needed)

If Qwen2-Audio is too large:
1. **Whisper-Large-v3** (OpenAI) - Faster, ASR-focused
2. **Wav2Vec2** (Meta) - Efficient, classification-focused
3. **HuBERT** (Meta) - Good for audio classification

Update `--model-name` parameter to switch models.

## References

- [Qwen2-Audio GitHub](https://github.com/QwenLM/Qwen2-Audio)
- [Qwen2-Audio Paper](https://arxiv.org/abs/2407.10759)
- [HuggingFace Model Card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
