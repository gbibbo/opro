#!/usr/bin/env python3
"""
Comprehensive test runner that executes all tests and saves logs.

This script runs:
1. Smoke test
2. Unit tests (pytest)
3. Code quality checks (ruff, black)

All output is saved to logs/ directory with timestamps.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
root_dir = Path(__file__).parent.parent
log_dir = root_dir / "logs"
log_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
master_log_file = log_dir / f"test_run_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(master_log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_command(cmd, description, log_file=None):
    """
    Run a shell command and capture output.

    Args:
        cmd: Command to run (list or string)
        description: Human-readable description
        log_file: Optional file to save output to

    Returns:
        tuple: (success: bool, output: str)
    """
    logger.info("=" * 80)
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    logger.info("-" * 80)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=root_dir,
            timeout=300  # 5 minute timeout
        )

        output = result.stdout + result.stderr
        success = result.returncode == 0

        # Log output
        if output:
            logger.info(output)

        # Save to dedicated log file if specified
        if log_file:
            log_path = log_dir / log_file
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"# {description}\n")
                f.write(f"# Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}\n")
                f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"# Exit code: {result.returncode}\n")
                f.write("\n" + output)
            logger.info(f"Output saved to: {log_path}")

        if success:
            logger.info(f"[PASS] {description}")
        else:
            logger.error(f"[FAIL] {description} (exit code: {result.returncode})")

        return success, output

    except subprocess.TimeoutExpired:
        logger.error(f"[FAIL] {description} TIMED OUT")
        return False, "Command timed out after 5 minutes"

    except Exception as e:
        logger.error(f"[FAIL] {description} ERROR: {e}")
        return False, str(e)


def main():
    """Run all tests and collect results."""
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE TEST RUNNER")
    logger.info(f"Master log file: {master_log_file}")
    logger.info("=" * 80)

    results = []

    # 1. Smoke test
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 1: SMOKE TEST")
    logger.info("=" * 80)

    smoke_cmd = [sys.executable, "scripts/smoke_test.py"]
    success, _ = run_command(
        smoke_cmd,
        "Smoke Test",
        f"smoke_test_{timestamp}.log"
    )
    results.append(("Smoke Test", success))

    # 2. Unit tests with pytest
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 2: UNIT TESTS (pytest)")
    logger.info("=" * 80)

    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--log-cli-level=INFO"
    ]
    success, _ = run_command(
        pytest_cmd,
        "Unit Tests (pytest)",
        f"pytest_{timestamp}.log"
    )
    results.append(("Unit Tests", success))

    # 3. Code quality - ruff (if available)
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 3: CODE QUALITY")
    logger.info("=" * 80)

    try:
        ruff_cmd = [sys.executable, "-m", "ruff", "check", "src/", "tests/", "scripts/"]
        success, _ = run_command(
            ruff_cmd,
            "Linting (ruff)",
            f"ruff_{timestamp}.log"
        )
        results.append(("Linting (ruff)", success))
    except Exception as e:
        logger.warning(f"Ruff not available: {e}")
        results.append(("Linting (ruff)", None))

    # 4. Code formatting - black (if available)
    try:
        black_cmd = [sys.executable, "-m", "black", "--check", "src/", "tests/", "scripts/"]
        success, _ = run_command(
            black_cmd,
            "Formatting (black)",
            f"black_{timestamp}.log"
        )
        results.append(("Formatting (black)", success))
    except Exception as e:
        logger.warning(f"Black not available: {e}")
        results.append(("Formatting (black)", None))

    # 5. Import test
    logger.info("\n" + "=" * 80)
    logger.info("SECTION 4: IMPORT TEST")
    logger.info("=" * 80)

    import_cmd = [
        sys.executable, "-c",
        "import qsm; from qsm import CONFIG, PROTOTYPE_MODE; from qsm.data import FrameTable; print('[PASS] All imports successful')"
    ]
    success, _ = run_command(
        import_cmd,
        "Import Test",
        f"import_test_{timestamp}.log"
    )
    results.append(("Import Test", success))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results:
        if result is True:
            logger.info(f"[PASS] {name}")
            passed += 1
        elif result is False:
            logger.error(f"[FAIL] {name}")
            failed += 1
        else:
            logger.warning(f"[SKIP] {name}")
            skipped += 1

    logger.info("-" * 80)
    logger.info(f"Total: {len(results)} tests")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info("=" * 80)

    if failed == 0 and passed > 0:
        logger.info("[PASS] ALL TESTS PASSED")
        logger.info(f"\nMaster log saved to: {master_log_file}")
        logger.info(f"Individual logs saved to: {log_dir}/")
        return 0
    else:
        logger.error("[FAIL] SOME TESTS FAILED")
        logger.info(f"\nMaster log saved to: {master_log_file}")
        logger.info(f"Individual logs saved to: {log_dir}/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
