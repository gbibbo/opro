#!/usr/bin/env python3
"""
Smoke test to verify basic functionality.

Runs in <30 seconds with minimal data to validate:
- Config loading
- Data structure creation
- Basic imports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all core modules can be imported."""
    logger.info("Testing imports...")

    try:
        import qsm
        from qsm import CONFIG, PROTOTYPE_MODE, PROTOTYPE_SAMPLES
        from qsm.data import FrameTable, load_dataset
        from qsm.data.slicing import slice_segments_from_interval

        logger.info(f"✓ Core imports successful")
        logger.info(f"  PROTOTYPE_MODE: {PROTOTYPE_MODE}")
        logger.info(f"  PROTOTYPE_SAMPLES: {PROTOTYPE_SAMPLES}")

        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration loading."""
    logger.info("Testing configuration...")

    try:
        from qsm import CONFIG

        # Check required sections
        required_sections = ["data", "models", "datasets", "durations_ms"]

        for section in required_sections:
            if section not in CONFIG:
                raise ValueError(f"Missing config section: {section}")

        logger.info(f"✓ Configuration valid")
        logger.info(f"  Durations: {CONFIG['durations_ms']}")

        return True

    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False


def test_data_structures():
    """Test basic data structure creation."""
    logger.info("Testing data structures...")

    try:
        import pandas as pd
        from qsm.data import FrameTable

        # Create minimal FrameTable
        data = pd.DataFrame({
            "uri": ["test_file"],
            "start_s": [0.0],
            "end_s": [1.0],
            "label": ["SPEECH"],
            "split": ["train"],
            "dataset": ["smoke_test"]
        })

        ft = FrameTable(data=data)

        assert len(ft.data) == 1
        assert len(ft.speech_segments) == 1
        assert len(ft.nonspeech_segments) == 0

        logger.info(f"✓ Data structures working")

        return True

    except Exception as e:
        logger.error(f"✗ Data structure test failed: {e}")
        return False


def test_slicing():
    """Test segment slicing."""
    logger.info("Testing slicing...")

    try:
        from pyannote.core import Segment
        from qsm.data.slicing import slice_segments_from_interval

        interval = Segment(0.0, 1.0)

        segments = slice_segments_from_interval(
            interval=interval,
            duration_ms=100,
            mode="speech"
        )

        assert len(segments) == 10
        assert segments[0].duration == 0.1

        logger.info(f"✓ Slicing working (created {len(segments)} segments)")

        return True

    except Exception as e:
        logger.error(f"✗ Slicing test failed: {e}")
        return False


def test_directory_structure():
    """Test that required directories exist or can be created."""
    logger.info("Testing directory structure...")

    try:
        from qsm import CONFIG

        required_dirs = [
            CONFIG["data"]["root"],
            CONFIG["data"]["raw"],
            CONFIG["data"]["processed"],
            CONFIG["data"]["segments"],
        ]

        for dir_path in required_dirs:
            path = Path(dir_path)
            path.mkdir(parents=True, exist_ok=True)

            if not path.exists():
                raise FileNotFoundError(f"Could not create: {dir_path}")

        logger.info(f"✓ Directory structure valid")

        return True

    except Exception as e:
        logger.error(f"✗ Directory test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    logger.info("=" * 60)
    logger.info("SMOKE TEST - Qwen Speech Minimum")
    logger.info("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Data Structures", test_data_structures),
        ("Slicing", test_slicing),
        ("Directory Structure", test_directory_structure),
    ]

    results = []

    for name, test_func in tests:
        logger.info(f"\n[{name}]")
        result = test_func()
        results.append((name, result))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {name}")

        if not result:
            all_passed = False

    logger.info("=" * 60)

    if all_passed:
        logger.info("✓ ALL TESTS PASSED")
        return 0
    else:
        logger.error("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
