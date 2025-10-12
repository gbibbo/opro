#!/usr/bin/env python3
"""
Quick validation script - Tests the complete pipeline with minimal samples.

This script:
1. Validates dataset integrity
2. Tests SNR measurement
3. Runs quick evaluation (1 clip per class)
4. Verifies results consistency

Expected runtime: ~2-3 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}\n")


def test_dataset_integrity():
    """Test 1: Validate dataset structure and files."""
    print_section("TEST 1: Dataset Integrity")

    manifest_path = Path("data/processed/conditions_final/conditions_manifest.parquet")

    if not manifest_path.exists():
        print(f"❌ FAIL: Manifest not found at {manifest_path}")
        return False

    df = pd.read_parquet(manifest_path)
    print(f"✓ Manifest loaded: {len(df)} samples")

    # Normalize label column to ground_truth if needed
    if 'label' in df.columns and 'ground_truth' not in df.columns:
        df['ground_truth'] = df['label'].str.replace('-', '').str.replace('_', '').str.upper()

    # Check structure
    required_cols = ['clip_id', 'audio_path', 'ground_truth', 'variant_type']
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        print(f"❌ FAIL: Missing columns: {missing_cols}")
        return False

    print(f"✓ All required columns present")

    # Check variant distribution
    variant_counts = df['variant_type'].value_counts()
    print(f"\nVariant distribution:")
    for vtype, count in variant_counts.items():
        print(f"  {vtype:12s}: {count:4d} samples")

    expected_variants = {'duration', 'snr', 'band', 'rir'}
    if set(variant_counts.keys()) != expected_variants:
        print(f"❌ FAIL: Expected variants {expected_variants}, got {set(variant_counts.keys())}")
        return False

    print(f"✓ All variant types present")

    # Sample 3 random files and check they exist
    sample_files = df.sample(3)['audio_path'].values
    all_exist = True

    for fpath in sample_files:
        # Convert Windows paths to cross-platform
        fpath_str = str(fpath).replace('\\', '/')
        p = Path(fpath_str)

        if not p.exists():
            print(f"❌ FAIL: File not found: {p}")
            all_exist = False
        else:
            print(f"✓ File exists: {p.name}")

    if not all_exist:
        return False

    print(f"\n✅ PASS: Dataset integrity validated")
    return True


def test_snr_measurement():
    """Test 2: Verify SNR is correct in generated files."""
    print_section("TEST 2: SNR Measurement")

    manifest_path = Path("data/processed/conditions_final/conditions_manifest.parquet")
    df = pd.read_parquet(manifest_path)

    # Normalize label to ground_truth if needed
    if 'label' in df.columns and 'ground_truth' not in df.columns:
        df['ground_truth'] = df['label'].str.replace('-', '').str.replace('_', '').str.upper()

    # Get one SNR sample per level
    snr_df = df[df['variant_type'] == 'snr'].copy()

    # Sample 3 SNR levels
    test_snrs = [0.0, -10.0, 10.0]

    print("Testing SNR accuracy on sample files:\n")

    all_passed = True

    for target_snr in test_snrs:
        # Get one sample at this SNR level
        samples = snr_df[snr_df['snr_db'] == target_snr]

        if len(samples) == 0:
            print(f"⚠  SKIP: No samples for SNR={target_snr:+.0f}dB")
            continue

        sample = samples.iloc[0]
        audio_path_str = str(sample['audio_path']).replace('\\', '/')
        audio_path = Path(audio_path_str)

        if not audio_path.exists():
            print(f"❌ FAIL: Audio not found: {audio_path}")
            all_passed = False
            continue

        # Load audio
        audio, sr = sf.read(audio_path)

        # Get corresponding original (padded) file
        clip_id = sample['clip_id']
        gt = sample['ground_truth']

        # Find original padded file
        orig_dir = Path(f"data/processed/padded/{gt.lower()}")
        orig_pattern = f"{clip_id}_padded.wav"
        orig_files = list(orig_dir.glob(orig_pattern))

        if len(orig_files) == 0:
            print(f"⚠  SKIP: Original not found for {clip_id}")
            continue

        orig_audio, _ = sf.read(orig_files[0])

        # Compute RMS of effective segment (500-1500ms)
        start_idx = int(sr * 0.5)
        end_idx = int(sr * 1.5)

        rms_orig = np.sqrt(np.mean(orig_audio[start_idx:end_idx]**2))
        rms_snr = np.sqrt(np.mean(audio[start_idx:end_idx]**2))

        # Expected ratio for this SNR
        # For SNR=-10: RMS_mix / RMS_orig = 3.317
        # For SNR=0:   RMS_mix / RMS_orig = 1.414
        # For SNR=+10: RMS_mix / RMS_orig = 1.005
        expected_ratios = {
            -10.0: 3.317,
            0.0: 1.414,
            10.0: 1.005,
        }

        measured_ratio = rms_snr / rms_orig if rms_orig > 0 else 0
        expected_ratio = expected_ratios.get(target_snr, 1.0)

        error_pct = abs(measured_ratio - expected_ratio) / expected_ratio * 100

        status = "✓" if error_pct < 10 else "❌"
        print(f"{status} SNR={target_snr:+3.0f}dB: ratio={measured_ratio:.3f} (expected={expected_ratio:.3f}, error={error_pct:.1f}%)")

        if error_pct >= 10:
            all_passed = False

    if all_passed:
        print(f"\n✅ PASS: SNR measurements within tolerance")
    else:
        print(f"\n❌ FAIL: Some SNR measurements out of tolerance")

    return all_passed


def test_quick_evaluation():
    """Test 3: Run quick evaluation with 1 clip per class."""
    print_section("TEST 3: Quick Evaluation (1 clip per class)")

    print("This will take ~2-3 minutes...")
    print("Loading model and running evaluation on 40 samples (2 clips × 20 variants)")
    print()

    import subprocess

    cmd = [
        "python", "scripts/debug_evaluate.py",
        "--n_clips", "1",
        "--output_dir", "results/quick_validation",
        "--seed", "42"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )

        # Check if evaluation completed
        output_file = Path("results/quick_validation/debug_results.parquet")

        if not output_file.exists():
            print(f"❌ FAIL: Evaluation output not created")
            print(f"STDOUT:\n{result.stdout[-500:]}")
            print(f"STDERR:\n{result.stderr[-500:]}")
            return False

        # Load results
        df = pd.read_parquet(output_file)

        if len(df) == 0:
            print(f"❌ FAIL: No results in output")
            return False

        # Calculate metrics
        total = len(df)
        correct = df['correct'].sum()
        accuracy = correct / total * 100

        print(f"Results:")
        print(f"  Total samples: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {accuracy:.1f}%")
        print()

        # By variant type
        print("By variant type:")
        for vtype in df['variant_type'].unique():
            subset = df[df['variant_type'] == vtype]
            acc = subset['correct'].mean() * 100
            print(f"  {vtype:12s}: {acc:5.1f}% (n={len(subset)})")

        # Minimum acceptable accuracy: 70% (allowing some failures)
        if accuracy < 70:
            print(f"\n❌ FAIL: Accuracy {accuracy:.1f}% below threshold (70%)")
            return False

        print(f"\n✅ PASS: Evaluation completed successfully")
        return True

    except subprocess.TimeoutExpired:
        print(f"❌ FAIL: Evaluation timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ FAIL: Evaluation error: {e}")
        return False


def main():
    """Run all tests."""
    print_section("OPRO QWEN - QUICK VALIDATION")
    print("Testing complete pipeline with minimal samples")
    print("Expected runtime: ~2-3 minutes")

    results = {}

    # Test 1: Dataset integrity
    try:
        results['dataset'] = test_dataset_integrity()
    except Exception as e:
        print(f"❌ FAIL: Dataset test crashed: {e}")
        results['dataset'] = False

    # Test 2: SNR measurement (skip if dataset failed)
    if results['dataset']:
        try:
            results['snr'] = test_snr_measurement()
        except Exception as e:
            print(f"❌ FAIL: SNR test crashed: {e}")
            results['snr'] = False
    else:
        print("\n⚠  SKIP: SNR test (dataset failed)")
        results['snr'] = False

    # Test 3: Quick evaluation (skip if previous tests failed)
    if results['dataset'] and results['snr']:
        try:
            results['evaluation'] = test_quick_evaluation()
        except Exception as e:
            print(f"❌ FAIL: Evaluation test crashed: {e}")
            results['evaluation'] = False
    else:
        print("\n⚠  SKIP: Evaluation test (previous tests failed)")
        results['evaluation'] = False

    # Summary
    print_section("VALIDATION SUMMARY")

    all_passed = all(results.values())

    print("Test Results:")
    print(f"  1. Dataset Integrity:  {'✅ PASS' if results['dataset'] else '❌ FAIL'}")
    print(f"  2. SNR Measurement:    {'✅ PASS' if results['snr'] else '❌ FAIL'}")
    print(f"  3. Quick Evaluation:   {'✅ PASS' if results['evaluation'] else '❌ FAIL'}")
    print()

    if all_passed:
        print("✅ ALL TESTS PASSED - Pipeline is working correctly!")
        print()
        print("Next steps:")
        print("  - Run full evaluation: python scripts/debug_evaluate.py --n_clips 50")
        print("  - Analyze results in results/quick_validation/")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please check errors above")
        print()
        print("Troubleshooting:")
        if not results['dataset']:
            print("  - Check that dataset exists in data/processed/conditions_final/")
            print("  - Try regenerating: python scripts/build_conditions.py ...")
        if not results['snr']:
            print("  - Check SNR implementation in src/qsm/audio/noise.py")
        if not results['evaluation']:
            print("  - Check model loading")
            print("  - Verify CUDA/GPU availability")
        return 1


if __name__ == "__main__":
    sys.exit(main())
