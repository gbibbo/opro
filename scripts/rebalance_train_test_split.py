#!/usr/bin/env python3
"""
Re-balance Train/Test Split for Extended Test Set

Uses existing normalized clips to create a larger test set (100-120 samples)
while maintaining balanced duration/SNR/class distribution.

Strategy:
- Current: 128 train, 32 test (160 total)
- Target: 60-80 train, 80-100 test
- Stratified sampling to maintain balance
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def load_all_normalized_data(normalized_csv: Path) -> pd.DataFrame:
    """Load all normalized clips."""
    df = pd.read_csv(normalized_csv)
    print(f"Loaded {len(df)} total normalized clips")

    # Parse label column (might be 'ground_truth' or 'label')
    if 'ground_truth' in df.columns:
        df['label'] = df['ground_truth']

    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.5,
    random_state: int = 42,
) -> tuple:
    """
    Stratified train/test split based on:
    - Class (SPEECH/NONSPEECH)
    - Duration (200/400/600/800/1000 ms)
    - SNR (0/5/10/15/20 dB)
    """

    # Create stratification key
    df['strata'] = (
        df['label'].astype(str) + "_" +
        df['duration_ms'].astype(str) + "ms_" +
        df['snr_db'].astype(str) + "dB"
    )

    # Count samples per stratum
    strata_counts = df['strata'].value_counts()
    print(f"\nStratification groups: {len(strata_counts)}")
    print(f"Samples per group: min={strata_counts.min()}, max={strata_counts.max()}, "
          f"mean={strata_counts.mean():.1f}")

    # Split
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['strata'],
            random_state=random_state,
        )
    except ValueError as e:
        print(f"\n[WARNING] Stratified split failed: {e}")
        print("Falling back to stratification by label only...")
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label'],
            random_state=random_state,
        )

    # Drop temporary column
    train_df = train_df.drop(columns=['strata'], errors='ignore')
    test_df = test_df.drop(columns=['strata'], errors='ignore')

    return train_df, test_df


def print_split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print summary of train/test split."""

    print(f"\n{'='*80}")
    print("SPLIT SUMMARY")
    print(f"{'='*80}")

    print(f"\nTotal samples: {len(train_df) + len(test_df)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/(len(train_df)+len(test_df))*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/(len(train_df)+len(test_df))*100:.1f}%)")

    # Class balance
    print(f"\nClass distribution:")
    print(f"  Train - SPEECH: {len(train_df[train_df['label']=='SPEECH'])}, "
          f"NONSPEECH: {len(train_df[train_df['label']=='NONSPEECH'])}")
    print(f"  Test  - SPEECH: {len(test_df[test_df['label']=='SPEECH'])}, "
          f"NONSPEECH: {len(test_df[test_df['label']=='NONSPEECH'])}")

    # Duration distribution
    print(f"\nDuration distribution (test set):")
    print(test_df['duration_ms'].value_counts().sort_index())

    # SNR distribution
    print(f"\nSNR distribution (test set):")
    print(test_df['snr_db'].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser(
        description="Re-balance train/test split for extended test set"
    )
    parser.add_argument(
        "--normalized_csv",
        type=str,
        default="data/processed/normalized_clips/normalized_metadata.csv",
        help="Path to normalized metadata CSV",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.6,
        help="Fraction of data to use for test (0.6 = 60%)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/extended_test_split",
        help="Output directory for new splits",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    normalized_csv = project_root / args.normalized_csv
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print("RE-BALANCE TRAIN/TEST SPLIT")
    print(f"{'='*80}")
    print(f"\nInput: {normalized_csv}")
    print(f"Test size: {args.test_size * 100:.0f}%")
    print(f"Random seed: {args.random_state}")
    print(f"Output: {output_dir}")

    # Load data
    df = load_all_normalized_data(normalized_csv)

    # Split
    print(f"\nPerforming stratified split...")
    train_df, test_df = stratified_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    # Summary
    print_split_summary(train_df, test_df)

    # Save
    # Rename 'ground_truth' back to 'label' for consistency with test scripts
    if 'ground_truth' in train_df.columns and 'label' not in train_df.columns:
        train_df = train_df.rename(columns={'ground_truth': 'label'})
        test_df = test_df.rename(columns={'ground_truth': 'label'})

    train_csv = output_dir / "train_metadata.csv"
    test_csv = output_dir / "test_metadata.csv"

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"\n{'='*80}")
    print("SAVED")
    print(f"{'='*80}")
    print(f"Train metadata: {train_csv} ({len(train_df)} samples)")
    print(f"Test metadata:  {test_csv} ({len(test_df)} samples)")

    print(f"\nNext step: Re-evaluate models with extended test set")
    print(f"  python scripts/run_robust_evaluation.py \\")
    print(f"      --models attention_only mlp \\")
    print(f"      --test_csv {test_csv}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
