#!/usr/bin/env python3
"""
Smoke test - Ultra-fast validation (< 30 seconds).

Two modes:
1. If dataset exists: Full validation (data + code)
2. If no dataset: Code-only validation (imports, syntax)

Tests:
1. Dataset files exist (optional if data not available)
2. SNR calculation is correct (optional if data not available)
3. Audio files are readable (optional if data not available)
4. Core module imports work (always)

Does NOT test:
- Model loading/inference (too slow)
- Full evaluation
"""

import sys
from pathlib import Path

# Check if dataset exists
DATA_AVAILABLE = Path("data/processed/conditions_final/conditions_manifest.parquet").exists()


def test_imports():
    """Test that core modules can be imported."""
    print("1. Testing core imports...")

    try:
        # Core modules
        import qsm
        from qsm.audio import (
            extract_segment_center,
            pad_audio_center,
            add_white_noise,
            mix_at_snr,
            apply_bandpass,
            apply_rir,
        )
        from qsm.models import Qwen2AudioClassifier
        from qsm.data import load_dataset

        print("   [OK] All core modules importable")
        return True
    except ImportError as e:
        print(f"   [X] Import failed: {e}")
        return False


def test_dataset():
    """Quick dataset check."""
    print("2. Testing dataset...")

    if not DATA_AVAILABLE:
        print("   [SKIP] No dataset (expected in CI)")
        return True

    import pandas as pd

    manifest = Path("data/processed/conditions_final/conditions_manifest.parquet")
    df = pd.read_parquet(manifest)
    print(f"   [OK] Manifest loaded: {len(df)} samples")

    # Check 3 random files
    for audio_path in df.sample(3)['audio_path']:
        p = Path(str(audio_path).replace(chr(92), '/'))
        if not p.exists():
            print(f"   [X] File not found: {p}")
            return False

    print("   [OK] Sample files exist")
    return True


def test_snr():
    """Quick SNR check on 3 samples."""
    print("3. Testing SNR accuracy...")

    if not DATA_AVAILABLE:
        print("   [SKIP] No dataset (expected in CI)")
        return True

    import pandas as pd
    import numpy as np
    import soundfile as sf

    manifest = Path("data/processed/conditions_final/conditions_manifest.parquet")
    df = pd.read_parquet(manifest)

    snr_df = df[df['variant_type'] == 'snr']

    # Test 3 SNR levels
    for target_snr in [0.0, -10.0, 10.0]:
        samples = snr_df[snr_df['snr_db'] == target_snr]
        if len(samples) == 0:
            continue

        sample = samples.iloc[0]
        audio_path = Path(str(sample['audio_path']).replace(chr(92), '/'))

        if not audio_path.exists():
            print(f"   [X] Audio not found: {audio_path}")
            return False

        audio, sr = sf.read(audio_path)

        # Get original
        clip_id = sample['clip_id']
        gt = sample['ground_truth']
        orig_dir = Path(f"data/processed/padded/{gt.lower()}")
        orig_files = list(orig_dir.glob(f"{clip_id}_padded.wav"))

        if not orig_files:
            continue

        orig_audio, _ = sf.read(orig_files[0])

        # Check effective segment (500-1500ms)
        start = int(sr * 0.5)
        end = int(sr * 1.5)

        rms_orig = np.sqrt(np.mean(orig_audio[start:end]**2))
        rms_snr = np.sqrt(np.mean(audio[start:end]**2))

        ratio = rms_snr / rms_orig if rms_orig > 0 else 0

        # Expected ratios (with 20% tolerance)
        expected = {-10.0: 3.3, 0.0: 1.4, 10.0: 1.0}[target_snr]
        error = abs(ratio - expected) / expected * 100

        if error > 20:
            print(f"   [X] SNR={target_snr:+.0f}dB: ratio={ratio:.2f} (expected ~{expected:.1f})")
            return False

    print("   [OK] SNR measurements correct")
    return True


def test_audio():
    """Quick audio read test."""
    print("4. Testing audio files...")

    if not DATA_AVAILABLE:
        print("   [SKIP] No dataset (expected in CI)")
        return True

    import pandas as pd
    import soundfile as sf

    manifest = Path("data/processed/conditions_final/conditions_manifest.parquet")
    df = pd.read_parquet(manifest)

    # Test one file per variant type
    for vtype in ['duration', 'snr', 'band', 'rir']:
        samples = df[df['variant_type'] == vtype]
        if len(samples) == 0:
            print(f"   [!] No {vtype} samples")
            continue

        audio_path = Path(str(samples.iloc[0]['audio_path']).replace(chr(92), '/'))

        try:
            audio, sr = sf.read(audio_path)
            if len(audio) == 0:
                print(f"   [X] Empty audio: {audio_path.name}")
                return False
        except Exception as e:
            print(f"   [X] Cannot read {audio_path.name}: {e}")
            return False

    print("   [OK] All variant types readable")
    return True


def main():
    print("=" * 60)
    print("SMOKE TEST - Quick Pipeline Validation")
    print("=" * 60)

    if DATA_AVAILABLE:
        print("[INFO] Dataset detected - Full validation mode")
    else:
        print("[INFO] No dataset - Code-only validation mode (CI)")

    print()

    tests = [
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("SNR", test_snr),
        ("Audio", test_audio),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"   [X] CRASH: {e}")
            results[name] = False
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"{status} {name}")

    print()

    if all_passed:
        print("[OK] SMOKE TEST PASSED")
        print()
        if DATA_AVAILABLE:
            print("Next: Run full validation")
            print("  python scripts/quick_validation.py")
        else:
            print("Note: Data-dependent tests skipped (no dataset)")
            print("To run full tests, generate dataset first.")
        return 0
    else:
        print("[FAIL] SMOKE TEST FAILED")
        print()
        print("Fix issues above before continuing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
