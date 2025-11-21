#!/usr/bin/env python3
"""
Automated audit script to verify zero data leakage in train/test split.

This script checks that no clip_id appears in both train and test sets,
which would constitute data leakage and inflate test accuracy.

Usage:
    python scripts/audit_split_leakage.py \
        --train_csv data/processed/grouped_split/train_metadata.csv \
        --test_csv data/processed/grouped_split/test_metadata.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def extract_base_clip_id(clip_id):
    """
    Extract the base clip ID (without time segment and duration suffix).

    Examples:
        '1-68734-A-34_1000ms_075' -> '1-68734-A-34'
        'voxconverse_afjiv_42.120_1000ms' -> 'voxconverse_afjiv'
        '1-51805-C-33_1000ms_039' -> '1-51805-C-33'
    """
    # For voxconverse: remove time segment (e.g., _42.120_1000ms)
    if clip_id.startswith('voxconverse_'):
        # Format: voxconverse_SPEAKER_TIME_DURATION
        # Extract: voxconverse_SPEAKER
        parts = clip_id.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"  # voxconverse_SPEAKER

    # For ESC-50: remove duration suffix (e.g., _1000ms_039)
    # Format: 1-68734-A-34_1000ms_075
    # Extract: 1-68734-A-34
    if '_1000ms_' in clip_id or '_200ms_' in clip_id:
        parts = clip_id.rsplit('_', 2)
        if len(parts) == 3:
            return parts[0]

    # Fallback: return as-is
    return clip_id


def audit_split_leakage(train_csv, test_csv, verbose=True):
    """
    Check for data leakage between train and test sets.

    Returns:
        dict with audit results including overlap count and details
    """
    # Load CSVs
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Extract base clip_ids
    train_df['base_clip_id'] = train_df['clip_id'].apply(extract_base_clip_id)
    test_df['base_clip_id'] = test_df['clip_id'].apply(extract_base_clip_id)

    # Get unique base clip_ids
    train_clips = set(train_df['base_clip_id'].unique())
    test_clips = set(test_df['base_clip_id'].unique())

    # Find overlap
    overlap = train_clips & test_clips

    # Statistics
    results = {
        'train_clips': len(train_clips),
        'test_clips': len(test_clips),
        'overlap_clips': len(overlap),
        'overlap_list': sorted(list(overlap)),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'leakage_detected': len(overlap) > 0
    }

    if verbose:
        print("="*70)
        print("SPLIT LEAKAGE AUDIT")
        print("="*70)

        print(f"\nTrain set:")
        print(f"  Unique clip_ids: {results['train_clips']}")
        print(f"  Total samples: {results['train_samples']}")

        print(f"\nTest set:")
        print(f"  Unique clip_ids: {results['test_clips']}")
        print(f"  Total samples: {results['test_samples']}")

        print(f"\nLeakage check:")
        print(f"  Overlapping clip_ids: {results['overlap_clips']}")

        if results['leakage_detected']:
            print(f"\n  WARNING: DATA LEAKAGE DETECTED!")
            print(f"  The following clip_ids appear in BOTH train and test:")
            for clip_id in results['overlap_list'][:10]:  # Show first 10
                print(f"    - {clip_id}")
            if len(results['overlap_list']) > 10:
                print(f"    ... and {len(results['overlap_list']) - 10} more")

            print(f"\n  IMPACT: Test accuracy is INFLATED and NOT reliable")
            print(f"  ACTION: Re-create split using GroupShuffleSplit")
        else:
            print(f"\n  OK: ZERO LEAKAGE - Split is clean!")
            print(f"  All clip_ids are unique to train or test")

        print()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Audit train/test split for data leakage"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Path to train metadata CSV"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to test metadata CSV"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Run audit
    results = audit_split_leakage(
        args.train_csv,
        args.test_csv,
        verbose=not args.quiet
    )

    # Exit code: 0 if no leakage, 1 if leakage detected
    exit_code = 1 if results['leakage_detected'] else 0
    exit(exit_code)


if __name__ == "__main__":
    main()
