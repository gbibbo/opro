# Testing Guide - Qwen Speech Minimum

This guide explains how to evaluate that Sprint 0 is correctly implemented and how to run tests with logging.

## Quick Start

### 1. Install Dependencies

Before running tests, ensure all dependencies are installed:

```bash
# Activate your conda environment
conda activate opro

# Install the package with dependencies
pip install -e .

# For development tools (optional)
pip install -e ".[dev]"
```

### 2. Run All Tests with Logging

We've created a comprehensive test runner that automatically saves all output to log files:

```bash
python scripts/run_all_tests.py
```

This script will:
- Run the smoke test
- Run all unit tests with pytest
- Check code quality (ruff, black)
- Test imports
- Save all output to `logs/` directory with timestamps

### 3. Review Logs

After running tests, check the logs in the `logs/` directory:

```bash
# View the master log (summary of all tests)
cat logs/test_run_YYYYMMDD_HHMMSS.log

# View individual test logs
cat logs/smoke_test_YYYYMMDD_HHMMSS.log
cat logs/pytest_YYYYMMDD_HHMMSS.log
cat logs/ruff_YYYYMMDD_HHMMSS.log
cat logs/black_YYYYMMDD_HHMMSS.log
```

**All logs are timestamped** so you can track multiple test runs.

---

## Individual Test Scripts

You can also run tests individually. All scripts now automatically save logs:

### Smoke Test

Quick validation (<30 seconds):

```bash
python scripts/smoke_test.py
```

**Output:** `logs/smoke_test_YYYYMMDD_HHMMSS.log`

**Tests:**
- Core imports
- Configuration loading
- Data structure creation
- Slicing functionality
- Directory setup

### Unit Tests (pytest)

Comprehensive unit tests:

```bash
pytest -v
```

**Output:** Individual log files created by each test file:
- `logs/test_loaders_YYYYMMDD_HHMMSS.log`
- `logs/test_slicing_YYYYMMDD_HHMMSS.log`

**Tests:**
- FrameTable creation and validation
- RTTM loading (with mock data)
- AVA-Speech loading (with mock data)
- Segment slicing at various durations
- PROTOTYPE_MODE limiting
- Save/load functionality
- Balancing segments
- Edge cases (too-short intervals, etc.)

### Code Quality

Lint and format checks:

```bash
# Linting
ruff check src/ tests/ scripts/

# Formatting
black --check src/ tests/ scripts/

# Type checking (optional)
mypy src/
```

---

## What to Check

### Sprint 0 Success Criteria

Sprint 0 is complete when:

1. **✓ All dependencies install cleanly**
   ```bash
   pip install -e .
   ```

2. **✓ Core imports work**
   ```python
   import qsm
   from qsm import CONFIG, PROTOTYPE_MODE
   from qsm.data import FrameTable, load_rttm_dataset
   ```

3. **✓ Smoke test passes**
   ```bash
   python scripts/smoke_test.py
   # Should complete in <30 seconds with all tests passing
   ```

4. **✓ Unit tests pass**
   ```bash
   pytest -v
   # All tests should pass
   ```

5. **✓ Configuration loads correctly**
   ```python
   from qsm import CONFIG
   assert "data" in CONFIG
   assert "models" in CONFIG
   assert "durations_ms" in CONFIG
   ```

6. **✓ Directory structure created**
   ```bash
   ls data/raw data/processed data/segments
   # All directories should exist
   ```

7. **✓ Mock data can be generated**
   ```bash
   python scripts/download_datasets.py --datasets voxconverse
   # Should create mock RTTM files
   ```

8. **✓ Logs are saved automatically**
   ```bash
   ls logs/
   # Should see timestamped log files
   ```

---

## Log File Structure

All test scripts save output to timestamped log files in the `logs/` directory:

```
logs/
├── test_run_20251008_161131.log        # Master log (all tests)
├── smoke_test_20251008_161131.log      # Smoke test output
├── pytest_20251008_161131.log          # Unit test output
├── test_loaders_20251008_161131.log    # Loader tests (from pytest)
├── test_slicing_20251008_161131.log    # Slicing tests (from pytest)
├── ruff_20251008_161131.log            # Linting output
├── black_20251008_161131.log           # Formatting output
└── import_test_20251008_161131.log     # Import verification
```

### Log Format

All logs include:
- Timestamp
- Test name
- Pass/fail status
- Detailed output
- Error messages (if any)

### Example Log Entry

```
2025-10-08 16:11:31,645 - INFO - Starting test_loaders.py
2025-10-08 16:11:31,646 - INFO - Running test_frame_table_creation
2025-10-08 16:11:31,650 - INFO - ✓ test_frame_table_creation PASSED
```

---

## Troubleshooting

### Missing Dependencies

If you see `ModuleNotFoundError`, install dependencies:

```bash
pip install -e .
```

For development tools:

```bash
pip install -e ".[dev]"
```

### Import Errors

If imports fail, ensure the package is installed:

```bash
pip install -e .
```

### Test Failures

1. **Check the logs** in `logs/` directory for detailed error messages
2. **Run tests individually** to isolate issues:
   ```bash
   pytest tests/test_loaders.py -v
   pytest tests/test_slicing.py -v
   ```
3. **Check PROTOTYPE_MODE** in `config.yaml` is set to `true`

### Unicode Errors on Windows

If you see `UnicodeEncodeError` in console output (Windows only):
- This is a **console display issue only**
- The **log files are saved correctly** and readable
- The tests still run and pass/fail correctly
- To avoid console errors, set environment variable:
  ```bash
  set PYTHONIOENCODING=utf-8
  ```

---

## Next Steps

Once all Sprint 0 tests pass:

1. **Review logs** to ensure everything works correctly
2. **Check EVALUATION.md** for detailed acceptance criteria
3. **Move to Sprint 1** (Dataset Ingestion)
   - Implement full RTTM loaders
   - Implement AVA-Speech loader
   - Implement AMI loader
   - Build unified FrameTable

---

## Log Retention

Logs are kept indefinitely (not automatically deleted). To clean up old logs:

```bash
# Remove all logs
rm -rf logs/*.log

# Remove logs older than 7 days (Linux/Mac)
find logs/ -name "*.log" -mtime +7 -delete

# Keep only last 10 test runs (manual)
ls -t logs/test_run_*.log | tail -n +11 | xargs rm
```

---

## Summary

**Test execution:**
```bash
# One-command test runner (recommended)
python scripts/run_all_tests.py

# Or run individually
python scripts/smoke_test.py
pytest -v
```

**Check logs:**
```bash
# Latest master log
ls -t logs/test_run_*.log | head -1 | xargs cat

# Latest smoke test
ls -t logs/smoke_test_*.log | head -1 | xargs cat
```

**All test output is automatically saved** to `logs/` directory for review.

---

**Document version:** 1.0
**Last updated:** 2025-10-08
**Sprint status:** Sprint 0 Testing Infrastructure Complete
