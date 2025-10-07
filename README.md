# Qwen Speech Minimum (QSM)

Temporal threshold measurement and optimization for speech detection in Qwen models.

## Overview

This project investigates the minimum temporal resolution at which Qwen Audio models can reliably detect speech, with focus on:

- **Psychometric curves** measuring detection threshold (DT50/DT75)
- **Prompt optimization** using OPRO and DSPy
- **LoRA fine-tuning** for short segments (≤200ms)
- **Comparison** with traditional VAD baselines (WebRTC, Silero)

## Key Features

### Prototype Mode
The project includes a **PROTOTYPE_MODE** setting in `config.yaml`:

```yaml
PROTOTYPE_MODE: true  # Set to false for full datasets
PROTOTYPE_SAMPLES: 5  # Number of examples per dataset
```

- **PROTOTYPE_MODE=true**: Downloads and uses only 5 examples per dataset (for development)
- **PROTOTYPE_MODE=false**: Downloads and processes full datasets (for production)

This allows rapid development without requiring large storage or compute resources.

### Datasets with High-Precision Ground Truth

1. **AVA-Speech** - Frame-level labels (clean/music/noise)
2. **DIHARD II/III** - RTTM with precise onset/offset
3. **VoxConverse** - RTTM v0.3 with fixes
4. **AMI** - Word-level forced alignment
5. **AVA-ActiveSpeaker** - Frame-level speaking labels

### Target Durations

Based on Qwen2-Audio's ~40ms effective frame resolution:
```
[20, 40, 60, 80, 100, 150, 200, 300, 500, 1000] ms
```

## Project Structure

```
qwen-speech-min/
├── config.yaml              # Global configuration (PROTOTYPE_MODE here!)
├── pyproject.toml          # Dependencies and build config
├── src/
│   └── qsm/                # Main package
│       ├── data/           # Loaders and slicing
│       ├── vad/            # VAD baselines
│       ├── qwen/           # Qwen model wrappers
│       ├── prompts/        # OPRO/DSPy optimization
│       ├── eval/           # Metrics and psychometric curves
│       ├── train/          # LoRA fine-tuning
│       └── viz/            # Visualization
├── scripts/                # CLI tools
│   ├── download_datasets.py
│   ├── smoke_test.py
│   └── ...
├── tests/                  # Unit tests
├── configs/                # Dataset configs
│   └── datasets/
└── data/                   # Data storage
    ├── raw/                # Downloaded datasets
    ├── processed/          # Processed annotations
    └── segments/           # Sliced audio segments
```

## Quick Start

### 1. Environment Setup

The project requires Python 3.11+ and uses conda for environment management.

```bash
# Activate your conda environment
conda activate opro

# Install the package and dependencies
pip install -e .

# For development (includes pytest, ruff, black, etc.)
pip install -e ".[dev]"
```

### 2. Download Prototype Data

In **PROTOTYPE_MODE**, this downloads only 5 examples per dataset:

```bash
python scripts/download_datasets.py --datasets all
```

This will:
- Create mock/sample data for each dataset
- Generate dataset configuration YAMLs in `configs/datasets/`
- Set up directory structure in `data/`

### 3. Run Smoke Test

Verify the installation:

```bash
python scripts/smoke_test.py
```

This runs in <30 seconds and validates:
- Configuration loading
- Data structure creation
- Slicing functionality
- Directory setup

### 4. Run Tests

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=qsm --cov-report=html

# Run specific test file
pytest tests/test_loaders.py -v
```

## Configuration

### Global Config (`config.yaml`)

Key settings:

```yaml
# Control dataset size
PROTOTYPE_MODE: true        # true = 5 examples, false = full datasets
PROTOTYPE_SAMPLES: 5

# Model settings
models:
  qwen2_audio:
    name: "Qwen/Qwen2-Audio-7B-Instruct"
    effective_frame_ms: 40  # ~40ms per output frame

# Target durations
durations_ms: [20, 40, 60, 80, 100, 150, 200, 300, 500, 1000]

# Reproducibility
seed: 42
```

### Dataset Configs (`configs/datasets/*.yaml`)

Each dataset has its own config with paths to annotations and audio:

```yaml
# configs/datasets/voxconverse.yaml
name: voxconverse
version: "0.3"
rttm_path:
  train: "./data/raw/voxconverse/train"
  dev: "./data/raw/voxconverse/dev"
  test: "./data/raw/voxconverse/test"
```

## Development Workflow

### Sprint 0: Infrastructure ✓
- [x] Project structure
- [x] Configuration system with PROTOTYPE_MODE
- [x] Data loaders skeleton
- [x] Download scripts
- [x] Testing framework
- [x] Smoke test

### Sprint 1: Dataset Ingestion (NEXT)
- [ ] Implement RTTM loaders (DIHARD, VoxConverse)
- [ ] Implement AVA-Speech loader
- [ ] Implement AMI alignment loader
- [ ] Validate ground truth precision
- [ ] Build unified FrameTable

### Sprint 2: Segment Slicing
- [ ] Extract segments at target durations
- [ ] Balance by condition (clean/music/noise)
- [ ] Export WAV + metadata

### Sprint 3-12: See roadmap in docs

## Usage Examples

### Loading a Dataset

```python
from qsm.data import load_dataset

# Load with automatic prototype limiting if PROTOTYPE_MODE=true
frame_table = load_dataset("voxconverse", split="dev")

print(f"Loaded {len(frame_table.data)} segments")
print(f"Speech segments: {len(frame_table.speech_segments)}")
```

### Slicing Segments

```python
from qsm.data import create_segments
from pathlib import Path

metadata_df = create_segments(
    frame_table=frame_table,
    audio_root=Path("data/raw/voxconverse/audio/dev"),
    output_dir=Path("data/segments/voxconverse_dev"),
    durations_ms=[40, 100, 200],
    max_segments_per_config=100
)
```

### Checking Prototype Mode

```python
from qsm import PROTOTYPE_MODE, PROTOTYPE_SAMPLES

if PROTOTYPE_MODE:
    print(f"Running in PROTOTYPE_MODE with {PROTOTYPE_SAMPLES} samples")
else:
    print("Running in FULL mode")
```

## Switching to Full Datasets

When ready to move from prototyping to full experiments:

1. **Update config.yaml:**
   ```yaml
   PROTOTYPE_MODE: false
   ```

2. **Download full datasets:**
   ```bash
   python scripts/download_datasets.py --force-full --datasets all
   ```

3. **Follow download instructions** printed for each dataset (many require licenses)

4. **Re-run processing** - all scripts respect the PROTOTYPE_MODE setting

## Architecture Notes

### Qwen2-Audio Temporal Resolution

- Mel spectrogram: **25ms window / 10ms hop**
- Pooling: **×2 stride**
- Effective resolution: **~40ms per output frame**
- Recommended max duration: **30 seconds**

This informs our target durations and expected psychometric thresholds.

### Data Precision Requirements

- **Frame-level** (AVA-Speech): ~40ms precision
- **RTTM** (DIHARD/VoxConverse): Onset/offset timestamps
- **Word-level** (AMI): ~10ms alignment steps

All loaders convert to unified `FrameTable` format for consistent processing.

## Testing

### Unit Tests

```bash
pytest tests/test_loaders.py    # Data loading
pytest tests/test_slicing.py    # Segment slicing
```

### Smoke Test

```bash
python scripts/smoke_test.py    # Quick validation (<30s)
```

### CI/CD

GitHub Actions runs:
- Linting (ruff, black)
- Unit tests
- Smoke test (with timeout)

## Dependencies

Core:
- `torch`, `torchaudio` - PyTorch (GPU support)
- `transformers` - Hugging Face models
- `pyannote.core`, `pyannote.database` - Audio annotations
- `librosa`, `soundfile` - Audio processing
- `webrtcvad` - WebRTC VAD baseline
- `peft`, `accelerate` - LoRA fine-tuning

Development:
- `pytest` - Testing
- `ruff`, `black` - Linting/formatting
- `jupyter` - Notebooks

See [pyproject.toml](pyproject.toml) for full dependency list.

## License

Apache-2.0 (matching Qwen model licenses)

## References

### Models
- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) - Audio LALM
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) - Omni-modal comparison

### Datasets
- [AVA-Speech](https://research.google.com/ava/download.html)
- [DIHARD](https://dihardchallenge.github.io/dihard3/)
- [VoxConverse](https://github.com/joonson/voxconverse)
- [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [AVA-ActiveSpeaker](https://github.com/cvdfoundation/ava-dataset)

### Methods
- [OPRO](https://arxiv.org/abs/2309.03409) - Prompt optimization
- [DSPy](https://dspy.ai) - Programming with LLMs
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad)
- [Silero VAD](https://github.com/snakers4/silero-vad)

## Citation

```bibtex
@software{qwen_speech_min,
  title={Qwen Speech Minimum: Temporal Threshold Measurement for Speech Detection},
  author={OPRO Qwen Team},
  year={2025}
}
```

## Contributing

1. Install dev dependencies: `pip install -e ".[dev]"`
2. Run tests: `pytest -v`
3. Format code: `black src/ tests/`
4. Lint: `ruff check src/ tests/`
5. Ensure smoke test passes: `python scripts/smoke_test.py`

## Troubleshooting

### Import Errors

If you get import errors, ensure the package is installed:
```bash
pip install -e .
```

### Data Not Found

Check that:
1. `PROTOTYPE_MODE` is set correctly in `config.yaml`
2. You've run `python scripts/download_datasets.py`
3. Dataset configs exist in `configs/datasets/`

### GPU Out of Memory

For prototyping, the 5-example limit should prevent OOM. For full datasets, adjust batch sizes in `config.yaml`.

---

**Status**: Sprint 0 Complete ✓ | Ready for Sprint 1 (Dataset Ingestion)
