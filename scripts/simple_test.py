#!/usr/bin/env python3
"""
Simple test - No dependencies on evaluation, just checks files exist.
Runtime: < 10 seconds
"""

import sys
from pathlib import Path
import pandas as pd

def main():
    print("=" * 60)
    print("SIMPLE TEST - Basic File Validation")
    print("=" * 60)
    print()

    # Test 1: Check manifest
    print("1. Checking manifest...")
    manifest = Path("data/processed/conditions_final/conditions_manifest.parquet")

    if not manifest.exists():
        print("   [FAIL] Manifest not found")
        return 1

    df = pd.read_parquet(manifest)
    print(f"   [OK] Manifest loaded: {len(df)} samples")

    # Normalize labels
    if 'label' in df.columns:
        df['ground_truth'] = df['label'].str.replace('-', '').str.upper()

    # Test 2: Check variant distribution
    print("\n2. Checking variant distribution...")
    variant_counts = df['variant_type'].value_counts()

    for vtype, count in variant_counts.items():
        print(f"   {vtype:12s}: {count:4d} samples")

    expected = {'duration', 'snr', 'band', 'rir'}
    if set(variant_counts.keys()) != expected:
        print(f"   [FAIL] Missing variant types")
        return 1

    print("   [OK] All variant types present")

    # Test 3: Check 5 random files exist
    print("\n3. Checking sample files...")

    sample_paths = df.sample(min(5, len(df)))['audio_path']
    all_exist = True

    for path_str in sample_paths:
        # Handle Windows paths
        p = Path(str(path_str).replace(chr(92), '/'))

        if p.exists():
            print(f"   [OK] {p.name}")
        else:
            print(f"   [FAIL] Not found: {p}")
            all_exist = False

    if not all_exist:
        return 1

    # Test 4: Check clip distribution
    print("\n4. Checking clip distribution...")

    n_clips = df['clip_id'].nunique()
    print(f"   Total unique clips: {n_clips}")

    if 'ground_truth' in df.columns:
        speech_clips = df[df['ground_truth'] == 'SPEECH']['clip_id'].nunique()
        nonspeech_clips = df[df['ground_truth'] == 'NONSPEECH']['clip_id'].nunique()

        print(f"   SPEECH clips: {speech_clips}")
        print(f"   NONSPEECH clips: {nonspeech_clips}")

        max_clips = min(speech_clips, nonspeech_clips)
        print(f"   Max balanced clips for eval: {max_clips}")

    # Summary
    print("\n" + "=" * 60)
    print("[OK] ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  - Quick eval (2-3 min):  python scripts/debug_evaluate.py --n_clips 2")
    print("  - Medium eval (10 min):  python scripts/debug_evaluate.py --n_clips 10")
    print(f"  - Full eval (20 min):    python scripts/debug_evaluate.py --n_clips {max_clips}")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
