#!/usr/bin/env python3
"""
Create train/test split with NO DATA LEAKAGE using GroupShuffleSplit.

CRITICAL FIX: Previous split was random by row, causing leakage when the same
clip_id appears with different variants (duration/SNR) in both train and test.

Solution: Group by clip_id and use GroupShuffleSplit to ensure all variants
of the same source clip stay together in either train OR test (never both).

We maintain stratification by label (SPEECH/NONSPEECH) while respecting groups.
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter


def extract_base_clip_id(clip_id: str) -> str:
    """
    Extract the base clip ID (without variant suffix).

    Examples:
        '1-68734-A-34_1000ms_075' -> '1-68734-A-34_1000ms_075'
        'voxconverse_afjiv_42.120_1000ms' -> 'voxconverse_afjiv_42.120_1000ms'

    The base clip_id is already the grouping key - all variants of the same
    source clip will have different original_variant values but same clip_id.
    """
    return clip_id


def stratified_group_split(df, test_size=0.2, random_state=42):
    """
    Split dataset by groups (clip_id) while maintaining class balance.

    Strategy:
    1. Group by clip_id (each group = all variants of same source clip)
    2. For each group, get its majority label (SPEECH or NONSPEECH)
    3. Use GroupShuffleSplit to split groups (not rows)
    4. This ensures no clip_id appears in both train and test
    5. Stratification is approximate (by group label, not exact row counts)

    Args:
        df: DataFrame with columns [clip_id, ground_truth, ...]
        test_size: Fraction for test set (default 0.2 = 80/20 split)
        random_state: Random seed for reproducibility

    Returns:
        train_df, test_df
    """
    # Extract base clip_id for grouping
    df = df.copy()
    df['group_id'] = df['clip_id'].apply(extract_base_clip_id)

    # Use 'ground_truth' column (not 'label')
    label_col = 'ground_truth' if 'ground_truth' in df.columns else 'label'

    # Assign each group to a stratum based on majority label
    group_labels = df.groupby('group_id')[label_col].agg(lambda x: x.mode()[0])
    df['group_label'] = df['group_id'].map(group_labels)

    # Get unique groups per stratum
    groups_speech = df[df['group_label'] == 'SPEECH']['group_id'].unique()
    groups_nonspeech = df[df['group_label'] == 'NONSPEECH']['group_id'].unique()

    print(f"\n=== Group Statistics ===")
    print(f"Total unique groups: {df['group_id'].nunique()}")
    print(f"  SPEECH groups: {len(groups_speech)}")
    print(f"  NONSPEECH groups: {len(groups_nonspeech)}")

    # Split each stratum separately
    train_groups = []
    test_groups = []

    for groups, label in [(groups_speech, 'SPEECH'), (groups_nonspeech, 'NONSPEECH')]:
        if len(groups) == 0:
            continue

        # Create dummy data for GroupShuffleSplit (it needs X and groups)
        n_groups = len(groups)
        dummy_X = list(range(n_groups))

        # Split groups
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(dummy_X, groups=groups))

        train_groups.extend(groups[train_idx])
        test_groups.extend(groups[test_idx])

        print(f"\n{label} split:")
        print(f"  Train groups: {len(train_idx)} ({len(train_idx)/n_groups*100:.1f}%)")
        print(f"  Test groups: {len(test_idx)} ({len(test_idx)/n_groups*100:.1f}%)")

    # Create train/test DataFrames
    train_df = df[df['group_id'].isin(train_groups)].copy()
    test_df = df[df['group_id'].isin(test_groups)].copy()

    # Remove temporary columns
    train_df = train_df.drop(columns=['group_id', 'group_label'])
    test_df = test_df.drop(columns=['group_id', 'group_label'])

    return train_df, test_df


def verify_no_leakage(train_df, test_df):
    """Verify that no clip_id appears in both train and test."""
    train_ids = set(train_df['clip_id'].apply(extract_base_clip_id))
    test_ids = set(test_df['clip_id'].apply(extract_base_clip_id))

    overlap = train_ids & test_ids

    print(f"\n=== Leakage Check ===")
    print(f"Train clip_ids: {len(train_ids)}")
    print(f"Test clip_ids: {len(test_ids)}")
    print(f"Overlap: {len(overlap)}")

    if overlap:
        print(f"WARNING: Found {len(overlap)} overlapping clip_ids!")
        print(f"Examples: {list(overlap)[:5]}")
        return False
    else:
        print("OK: No leakage detected - all clip_ids are unique to train or test")
        return True


def print_split_statistics(train_df, test_df):
    """Print detailed statistics about the split."""
    # Determine label column
    label_col = 'ground_truth' if 'ground_truth' in train_df.columns else 'label'

    print(f"\n=== Split Statistics ===")
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Total: {len(train_df) + len(test_df)} samples")

    print(f"\nClass distribution (Train):")
    train_counts = train_df[label_col].value_counts()
    for label, count in train_counts.items():
        print(f"  {label}: {count} ({count/len(train_df)*100:.1f}%)")

    print(f"\nClass distribution (Test):")
    test_counts = test_df[label_col].value_counts()
    for label, count in test_counts.items():
        print(f"  {label}: {count} ({count/len(test_df)*100:.1f}%)")

    # Duration × SNR distribution
    print(f"\nTrain duration × SNR distribution:")
    train_variant_counts = train_df.groupby(['duration_ms', 'snr_db', label_col]).size()
    print(train_variant_counts.to_string())

    print(f"\nTest duration × SNR distribution:")
    test_variant_counts = test_df.groupby(['duration_ms', 'snr_db', label_col]).size()
    print(test_variant_counts.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Create group-stratified train/test split (no data leakage)"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/processed/clean_clips/clean_metadata.csv",
        help="Input CSV with all samples"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/grouped_split",
        help="Output directory for train/test CSVs"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for test set (default: 0.2 for 80/20 split)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples")

    # Create split
    print(f"\nCreating group-stratified split (test_size={args.test_size})...")
    train_df, test_df = stratified_group_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Verify no leakage
    no_leakage = verify_no_leakage(train_df, test_df)

    # Print statistics
    print_split_statistics(train_df, test_df)

    # Save splits
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_metadata.csv"
    test_path = output_dir / "test_metadata.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n=== Saved ===")
    print(f"Train: {train_path} ({len(train_df)} samples)")
    print(f"Test: {test_path} ({len(test_df)} samples)")

    if not no_leakage:
        print("\nWARNING: Data leakage detected! Check the split logic.")
        exit(1)
    else:
        print("\nSUCCESS: Split created successfully with no data leakage!")


if __name__ == "__main__":
    main()
