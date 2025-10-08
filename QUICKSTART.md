# Quick Start - Qwen Speech Minimum

**Sprint 0 Complete ✅** | All tests now save logs automatically

---

## 1-Minute Setup

```bash
# Install
pip install -e .

# Test (runs all tests + saves logs)
python scripts/run_all_tests.py

# Check logs
ls logs/
```

---

## Key Commands

### Run Tests

```bash
# All tests (recommended)
python scripts/run_all_tests.py

# Individual tests
python scripts/smoke_test.py    # Quick validation (<30s)
pytest -v                       # Unit tests
pytest tests/test_loaders.py    # Just loaders
pytest tests/test_slicing.py    # Just slicing
```

### View Logs

```bash
# Latest master log
ls -t logs/test_run_*.log | head -1 | xargs cat

# Specific logs
cat logs/smoke_test_*.log
cat logs/pytest_*.log
cat logs/test_loaders_*.log
cat logs/test_slicing_*.log
```

### Import Test

```python
import qsm
from qsm import CONFIG, PROTOTYPE_MODE
from qsm.data import FrameTable, load_rttm_dataset
print(f"Prototype mode: {PROTOTYPE_MODE}")
```

---

## Project Structure

```
qwen-speech-min/
├── config.yaml           # PROTOTYPE_MODE here!
├── src/qsm/              # Main package
├── scripts/              # CLI tools
│   ├── smoke_test.py
│   └── run_all_tests.py  # NEW: Run all tests
├── tests/                # Unit tests (with logging)
├── logs/                 # NEW: All test output (timestamped)
└── data/                 # Data storage
```

---

## Configuration

Edit `config.yaml`:

```yaml
PROTOTYPE_MODE: true      # true = 5 samples, false = full datasets
PROTOTYPE_SAMPLES: 5      # Number of samples in prototype mode

durations_ms: [20, 40, 60, 80, 100, 150, 200, 300, 500, 1000]
seed: 42
```

---

## Sprint 0 Checklist

✅ Install: `pip install -e .`
✅ Test: `python scripts/run_all_tests.py`
✅ Logs: Check `logs/` directory
✅ Imports: `import qsm` works
✅ Config: `config.yaml` loads
✅ Smoke test: Passes (<30s)
✅ Unit tests: All pass

**All criteria met? Sprint 0 complete! ✅**

---

## Documentation

- **SPRINT0_SUMMARY.md** - What was implemented
- **EVALUATION.md** - Detailed acceptance criteria
- **TESTING_GUIDE.md** - How to run tests with logging
- **README.md** - Full project documentation

---

## New Feature: Automatic Logging ✨

**All test scripts now save output to `logs/` automatically:**

```
logs/
├── test_run_YYYYMMDD_HHMMSS.log        # Master log (all tests)
├── smoke_test_YYYYMMDD_HHMMSS.log      # Smoke test
├── pytest_YYYYMMDD_HHMMSS.log          # Pytest
├── test_loaders_YYYYMMDD_HHMMSS.log    # Loader tests
└── test_slicing_YYYYMMDD_HHMMSS.log    # Slicing tests
```

**Benefits:**
- ✅ Run tests in terminal
- ✅ I can read logs automatically
- ✅ Timestamped for tracking
- ✅ No manual copy/paste needed

---

## Troubleshooting

### Missing dependencies
```bash
pip install -e .
```

### Import errors
```bash
pip install -e .
python -c "import qsm"
```

### Test failures
```bash
# Check logs
cat logs/test_run_*.log

# Run individually
pytest tests/test_loaders.py -v
```

### Unicode errors (Windows console)
```bash
set PYTHONIOENCODING=utf-8
```

**Note:** Log files are always saved correctly, even if console shows errors.

---

## Next Steps

1. ✅ **Sprint 0 Complete** - Infrastructure ready
2. **Sprint 1 Next** - Dataset ingestion
   - Implement full RTTM loaders
   - Implement AVA-Speech loader
   - Implement AMI loader
   - Build unified FrameTable

---

**Quick help:**
- Tests not passing? Check `TESTING_GUIDE.md`
- What was implemented? Check `SPRINT0_SUMMARY.md`
- Detailed criteria? Check `EVALUATION.md`
- Full docs? Check `README.md`

---

**Status:** Sprint 0 Complete ✅ | Ready for Sprint 1
