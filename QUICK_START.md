# Quick Start Guide

## 🚀 Setup (First Time)

```bash
# 1. Navigate to project
cd "C:\VS projects\OPRO Qwen"

# 2. Activate conda environment
conda activate opro

# 3. Install package
pip install -e .

# 4. Verify installation
python scripts/smoke_test.py
```

## 📊 Working with Prototype Data (5 samples per dataset)

### Download Prototype Datasets

```bash
python scripts/download_datasets.py --datasets all
```

This creates mock data for development (controlled by `PROTOTYPE_MODE: true` in config.yaml).

### Check What's Downloaded

```bash
# List raw data
ls data/raw/

# Check specific dataset
ls data/raw/voxconverse/
```

## 🔄 Switch to Full Datasets (Later)

### Step 1: Update config.yaml

```yaml
PROTOTYPE_MODE: false  # Change from true to false
```

### Step 2: Download Full Datasets

```bash
python scripts/download_datasets.py --force-full --datasets all
```

**Note**: Many datasets require licenses. Follow the instructions printed by the script.

## 📝 Common Commands

### Run Tests

```bash
# All tests
pytest -v

# Specific test file
pytest tests/test_loaders.py -v

# With coverage
pytest --cov=qsm --cov-report=html
```

### Create Segments

```bash
# Create segments at target durations
python scripts/make_segments.py \
  --dataset voxconverse \
  --split dev \
  --durations 40 100 200
```

### Check Configuration

```python
from qsm import CONFIG, PROTOTYPE_MODE, PROTOTYPE_SAMPLES

print(f"Prototype mode: {PROTOTYPE_MODE}")
print(f"Samples per dataset: {PROTOTYPE_SAMPLES}")
print(f"Target durations: {CONFIG['durations_ms']}")
```

## 🔍 Verify Everything is Working

### Quick Health Check

```bash
python scripts/smoke_test.py
```

Expected: All tests pass ✓

### Load Sample Data

```python
from qsm.data import load_dataset

# Load prototype data (automatically limited to 5 examples)
frame_table = load_dataset("voxconverse", split="dev")

print(f"Loaded {len(frame_table.data)} segments")
print(f"Speech: {len(frame_table.speech_segments)}")
print(f"Non-speech: {len(frame_table.nonspeech_segments)}")
```

### Create Test Segments

```python
from qsm.data import create_segments
from pathlib import Path

metadata_df = create_segments(
    frame_table=frame_table,
    audio_root=Path("data/raw/voxconverse/audio/dev"),
    output_dir=Path("data/segments/voxconverse_dev_test"),
    durations_ms=[100, 200],
    max_segments_per_config=10
)

print(f"Created {len(metadata_df)} segments")
```

## 📁 Project Structure

```
OPRO Qwen/
├── config.yaml          ← CHANGE PROTOTYPE_MODE HERE
├── src/qsm/             ← Main package code
├── scripts/             ← Executable scripts
│   ├── download_datasets.py
│   ├── make_segments.py
│   └── smoke_test.py
├── tests/               ← Unit tests
├── data/                ← Data storage
│   ├── raw/             ← Downloaded datasets
│   ├── processed/       ← Processed annotations
│   └── segments/        ← Sliced segments
└── configs/datasets/    ← Auto-generated dataset configs
```

## ⚙️ Important Configuration

### config.yaml (Main Settings)

```yaml
# Control dataset size
PROTOTYPE_MODE: true     # ← CHANGE THIS to switch modes
PROTOTYPE_SAMPLES: 5

# Target durations for segments (ms)
durations_ms: [20, 40, 60, 80, 100, 150, 200, 300, 500, 1000]

# Qwen2-Audio effective frame resolution
models:
  qwen2_audio:
    effective_frame_ms: 40  # ~40ms per output frame

# Reproducibility
seed: 42
```

## 🐛 Troubleshooting

### Can't import qsm

```bash
pip install -e .
```

### WebRTC VAD error on Windows

This is normal. WebRTC VAD is optional and requires C++ Build Tools. Skip it for now:

```bash
# Already done automatically - webrtcvad is in optional dependencies
```

### No data in data/raw/

```bash
python scripts/download_datasets.py --datasets all
```

### Tests failing

```bash
# Check installation
python scripts/smoke_test.py

# Reinstall if needed
pip install -e . --force-reinstall --no-deps
```

## 📚 Next Steps

### For Development (Sprint 1)

1. ✅ Complete installation: `pip install -e .`
2. ✅ Run smoke test: `python scripts/smoke_test.py`
3. ✅ Download data: `python scripts/download_datasets.py --datasets all`
4. 📋 Implement dataset loaders (see Sprint 1 tasks)
5. 📋 Test with prototype data
6. 📋 Validate ground truth precision

### For Production (Later)

1. Change `PROTOTYPE_MODE: false` in config.yaml
2. Download full datasets (follow license requirements)
3. Re-run processing pipeline
4. Run experiments

## 🎯 Current Status

- ✅ Sprint 0: Infrastructure - COMPLETE
- ⏭️ Sprint 1: Dataset Ingestion - NEXT

## 📖 Documentation

- [README.md](README.md) - Full project documentation
- [INSTALL.md](INSTALL.md) - Detailed installation guide
- [SPRINT_0_COMPLETE.md](SPRINT_0_COMPLETE.md) - Sprint 0 summary

## 💡 Tips

1. **Always activate conda environment first**: `conda activate opro`
2. **Check PROTOTYPE_MODE before processing**: Avoid accidentally processing full datasets
3. **Use smoke test to verify changes**: `python scripts/smoke_test.py`
4. **Run tests frequently**: `pytest -v`
5. **Check config.yaml for all global settings**

---

**You're ready to start Step 1 (Sprint 1)!** 🎉
