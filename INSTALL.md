# Installation Guide

## Prerequisites

- Python 3.11+
- Conda environment already created (named `opro`)
- GPU with CUDA support (optional but recommended)

## Installation Steps

### 1. Activate Conda Environment

```bash
conda activate opro
```

### 2. Install Core Package

```bash
cd "C:\VS projects\OPRO Qwen"
pip install -e .
```

**Note for Windows users**: This will install all dependencies except `webrtcvad`, which requires Microsoft C++ Build Tools. WebRTC VAD is optional for the prototype phase.

### 3. (Optional) Install WebRTC VAD

If you have Microsoft Visual C++ 14.0 or greater installed:

```bash
pip install -e ".[vad]"
```

Otherwise, download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.

### 4. (Optional) Install Development Tools

```bash
pip install -e ".[dev]"
```

This includes: pytest, ruff, black, mypy, jupyter, ipython

## Verification

### Quick Check

```bash
python -c "import qsm; print(f'✓ QSM installed: v{qsm.__version__}')"
```

### Run Smoke Test

```bash
python scripts/smoke_test.py
```

This should complete in <30 seconds and verify:
- Configuration loading
- Data structures
- Slicing functionality
- Directory setup

### Run Unit Tests

```bash
pytest -v
```

## Troubleshooting

### Issue: Cannot import qsm

**Solution**: Ensure package is installed in editable mode:
```bash
pip install -e .
```

### Issue: webrtcvad build error on Windows

**Solution**: This is expected. WebRTC VAD is optional and not required for prototyping. You can:
1. Install Microsoft C++ Build Tools and retry, or
2. Skip WebRTC VAD (Silero VAD will be used instead)

### Issue: PyTorch GPU not detected

**Solution**: Reinstall PyTorch with CUDA support:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with your CUDA version.

### Issue: Missing datasets

**Solution**: Download prototype datasets:
```bash
python scripts/download_datasets.py --datasets all
```

## Next Steps

After successful installation:

1. **Download Prototype Data**:
   ```bash
   python scripts/download_datasets.py --datasets all
   ```

2. **Verify Setup**:
   ```bash
   python scripts/smoke_test.py
   ```

3. **Run Tests**:
   ```bash
   pytest -v
   ```

4. **Start Development**: Follow README.md for usage examples

## Configuration

The project uses a global `config.yaml` with important settings:

```yaml
PROTOTYPE_MODE: true  # true = 5 samples, false = full datasets
PROTOTYPE_SAMPLES: 5
```

To switch to full datasets later, set `PROTOTYPE_MODE: false`.

## GPU Setup

The project is configured to use GPU if available. Check GPU status:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Package Structure

After installation, you should have:

```
C:\VS projects\OPRO Qwen\
├── src/qsm/          # Installed as editable package
├── scripts/          # Available as executables
├── tests/            # Run with pytest
├── configs/          # Configuration files
└── data/             # Data storage
```
