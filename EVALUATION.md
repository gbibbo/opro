# Sprint 0 Evaluation Checklist

This document provides criteria to evaluate that Sprint 0 (Infrastructure) has been correctly implemented.

## Overview

**Sprint 0 Goal:** Establish a reproducible skeleton from day 1.

**Status:** ✓ **COMPLETE** (See detailed checklist below)

---

## Acceptance Criteria

### 1. Repository Structure ✓

**Required directories:**
```
qwen-speech-min/
├── config.yaml              # Global configuration
├── pyproject.toml          # Dependencies and build config
├── requirements.txt        # Pip dependencies
├── README.md               # Project documentation
├── src/
│   └── qsm/                # Main package
│       ├── __init__.py
│       └── data/           # Loaders and slicing
│           ├── __init__.py
│           ├── loaders.py
│           └── slicing.py
├── scripts/                # CLI tools
│   ├── download_datasets.py
│   ├── make_segments.py
│   └── smoke_test.py
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_loaders.py
│   └── test_slicing.py
├── configs/                # Dataset configs
│   └── datasets/
└── data/                   # Data storage
    ├── raw/
    ├── processed/
    └── segments/
```

**Verification:**
- [ ] All directories exist or are created on first run
- [ ] Package structure follows Python best practices
- [ ] Clear separation between source, tests, configs, and data

---

### 2. Environment Management ✓

**Requirements:**
- Python 3.11+
- conda environment support
- All core dependencies specified

**Core dependencies:**
```
torch, torchaudio, transformers, datasets
soundfile, librosa, pyannote.core, pyannote.database
peft, accelerate, scikit-learn, scipy, matplotlib
pyyaml, tqdm, pandas, pyarrow
```

**Verification:**
```bash
# Check Python version
python --version  # Should be >= 3.11

# Install package
pip install -e .

# Verify imports
python -c "import qsm; print(qsm.__version__)"
```

**Expected:** No import errors, clean installation.

---

### 3. Configuration System ✓

**File:** `config.yaml`

**Required sections:**
- `PROTOTYPE_MODE` and `PROTOTYPE_SAMPLES` flags
- `data` paths (root, raw, processed, segments)
- `models` config (qwen2_audio, qwen3_omni)
- `vad` baselines (webrtc, silero)
- `durations_ms` target list
- `datasets` configuration
- `evaluation`, `prompt_optimization`, `fine_tuning` settings
- `seed` and `logging` config

**Verification:**
```python
from qsm import CONFIG, PROTOTYPE_MODE, PROTOTYPE_SAMPLES

assert "data" in CONFIG
assert "models" in CONFIG
assert "durations_ms" in CONFIG
assert isinstance(PROTOTYPE_MODE, bool)
assert isinstance(PROTOTYPE_SAMPLES, int)
```

**Expected:** All sections present, valid YAML syntax, sensible defaults.

---

### 4. Data Loaders (Skeleton) ✓

**File:** `src/qsm/data/loaders.py`

**Required classes/functions:**
- `FrameTable` - unified data container
- `load_rttm_dataset()` - RTTM loader
- `load_ava_speech()` - AVA-Speech loader
- `iter_intervals()` - iterate speech/nonspeech intervals

**Verification:**
```python
from qsm.data import FrameTable, load_rttm_dataset, iter_intervals
import pandas as pd

# Test FrameTable creation
data = pd.DataFrame({
    "uri": ["file_1"],
    "start_s": [0.0],
    "end_s": [1.0],
    "label": ["SPEECH"],
    "split": ["train"],
    "dataset": ["test"]
})
ft = FrameTable(data=data)
assert len(ft.data) == 1
assert len(ft.speech_segments) == 1
```

**Expected:** Classes instantiate, basic validation works.

---

### 5. Segment Slicing (Skeleton) ✓

**File:** `src/qsm/data/slicing.py`

**Required functions:**
- `slice_segments_from_interval()` - extract fixed-duration segments
- `balance_segments()` - balance by condition/duration
- `SegmentMetadata` - dataclass for metadata

**Verification:**
```python
from qsm.data.slicing import slice_segments_from_interval
from pyannote.core import Segment

interval = Segment(0.0, 1.0)
segments = slice_segments_from_interval(interval, duration_ms=100, mode="speech")

assert len(segments) == 10  # 1000ms / 100ms = 10
assert segments[0].duration == 0.1
```

**Expected:** Segments created with correct durations, no overlaps.

---

### 6. Testing Framework ✓

**Test files:**
- `tests/test_loaders.py` - data loader tests
- `tests/test_slicing.py` - slicing tests
- `scripts/smoke_test.py` - quick validation

**Verification:**
```bash
# Run all tests
pytest -v

# Run smoke test
python scripts/smoke_test.py
```

**Expected results:**
- All unit tests pass
- Smoke test completes in <30 seconds
- No import errors
- Clear pass/fail indicators

**Key tests:**
- FrameTable creation and validation
- RTTM loading (mock data)
- AVA-Speech loading (mock data)
- Segment slicing at various durations
- PROTOTYPE_MODE limiting
- Save/load functionality

---

### 7. Download Scripts ✓

**File:** `scripts/download_datasets.py`

**Features:**
- Support for `--datasets all` or individual datasets
- Respects `PROTOTYPE_MODE` from config.yaml
- Creates mock data when in prototype mode
- Generates dataset YAML configs
- Prints instructions for full downloads

**Verification:**
```bash
# In PROTOTYPE_MODE
python scripts/download_datasets.py --datasets voxconverse

# Check outputs
ls configs/datasets/voxconverse.yaml  # Config created
ls data/raw/voxconverse/  # Mock data created
```

**Expected:**
- Mock RTTM files generated
- Dataset configs written to `configs/datasets/`
- Clear instructions printed for full downloads

---

### 8. Code Quality (Basic) ✓

**Tools configured:**
- `ruff` - linting (E, F, I, N, W, UP rules)
- `black` - formatting (line length 100)
- `mypy` - type checking (optional)
- `pytest` - testing

**Configuration in:** `pyproject.toml`

**Verification:**
```bash
# Check formatting
black --check src/ tests/ scripts/

# Run linter
ruff check src/ tests/ scripts/

# Type checking (if mypy installed)
mypy src/
```

**Expected:** Minimal linting errors, consistent formatting.

---

## Testing Protocol

### Automated Tests

Run the complete test suite:

```bash
# 1. Activate environment
conda activate opro

# 2. Run unit tests with logging
pytest -v --log-cli-level=INFO --log-file=logs/pytest_output.log

# 3. Run smoke test with logging
python scripts/smoke_test.py 2>&1 | tee logs/smoke_test.log

# 4. Check code quality
ruff check src/ tests/ scripts/ 2>&1 | tee logs/ruff_output.log
black --check src/ tests/ scripts/ 2>&1 | tee logs/black_output.log
```

All logs are saved to `logs/` directory for review.

### Manual Verification

1. **Import test:**
   ```python
   import qsm
   from qsm import CONFIG, PROTOTYPE_MODE
   from qsm.data import FrameTable, load_rttm_dataset
   print("✓ All imports successful")
   ```

2. **Config test:**
   ```python
   from qsm import CONFIG
   print(f"Prototype mode: {CONFIG.get('PROTOTYPE_MODE')}")
   print(f"Durations: {CONFIG.get('durations_ms')}")
   ```

3. **Data structure test:**
   ```python
   from qsm.data import FrameTable
   import pandas as pd

   data = pd.DataFrame({
       "uri": ["test"],
       "start_s": [0.0],
       "end_s": [1.0],
       "label": ["SPEECH"],
       "split": ["train"],
       "dataset": ["test"]
   })
   ft = FrameTable(data=data)
   print(f"✓ Created FrameTable with {len(ft.data)} segments")
   ```

4. **Slicing test:**
   ```python
   from qsm.data.slicing import slice_segments_from_interval
   from pyannote.core import Segment

   segments = slice_segments_from_interval(
       Segment(0.0, 1.0),
       duration_ms=100,
       mode="speech"
   )
   print(f"✓ Created {len(segments)} segments of 100ms")
   ```

---

## Success Criteria Summary

**Sprint 0 is complete when:**

- ✓ `pytest -q` passes all tests
- ✓ `python scripts/smoke_test.py` completes successfully (<30s)
- ✓ `pip install -e .` installs without errors
- ✓ All core imports work (`import qsm`, `from qsm.data import ...`)
- ✓ Configuration loads and validates
- ✓ Directory structure is created
- ✓ Mock data can be generated via download script
- ✓ FrameTable can be created and manipulated
- ✓ Segments can be sliced at target durations
- ✓ Code quality checks pass (ruff, black)
- ✓ All logs are saved to `logs/` directory for review

---

## Known Limitations (By Design)

These are **intentional** for Sprint 0:

1. **No actual data downloaded** - Using mock/prototype data only
2. **Loaders are skeletons** - Full parsing of RTTM/AVA/AMI in Sprint 1
3. **No VAD implementations** - WebRTC/Silero in Sprint 3
4. **No Qwen inference** - Model wrappers in Sprint 4
5. **No prompt optimization** - OPRO/DSPy in Sprint 6
6. **No fine-tuning** - LoRA in Sprint 8

These will be implemented in subsequent sprints.

---

## Next Steps (Sprint 1)

Once Sprint 0 passes all criteria:

1. **T-103: Implement full RTTM loaders**
   - Parse DIHARD RTTM files
   - Parse VoxConverse RTTM files
   - Validate against official counts

2. **T-103: Implement AVA-Speech loader**
   - Parse CSV annotations
   - Map frame labels to SPEECH/NONSPEECH
   - Extract conditions (clean/music/noise)

3. **T-103: Implement AMI loader**
   - Parse forced alignment files
   - Convert word-level to frame-level
   - Handle overlapping speakers

4. **T-103: Build unified FrameTable**
   - Combine all datasets
   - Validate consistency
   - Export to parquet

---

## Troubleshooting

### Import errors
```bash
pip install -e .  # Reinstall package
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

### Test failures
```bash
pytest -v --tb=short  # Show detailed traceback
```

### Directory permissions
```bash
chmod -R u+w data/  # Ensure write permissions
```

---

## Logging Configuration

All test scripts now save output to `logs/` directory:

- `logs/pytest_output.log` - pytest test results
- `logs/smoke_test.log` - smoke test output
- `logs/ruff_output.log` - linting results
- `logs/black_output.log` - formatting check results
- `logs/test_run_YYYYMMDD_HHMMSS.log` - timestamped test runs

Logs are **cumulative** and include timestamps for tracking progress over time.

---

## Contact

For issues or questions:
- Check `README.md` for setup instructions
- Review logs in `logs/` directory
- Verify `config.yaml` settings
- Ensure `PROTOTYPE_MODE=true` for development

---

**Document version:** 1.0
**Last updated:** 2025-10-08
**Sprint status:** Sprint 0 ✓ COMPLETE | Ready for Sprint 1
