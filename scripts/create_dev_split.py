#!/usr/bin/env python3
"""
Create Dev Split from Train Set (for Calibration)

Splits the training set into 80% train / 20% dev using GroupShuffleSplit
to prevent data leakage. The dev set is used for:
1. Temperature calibration
2. Hyperparameter tuning
3. Early stopping

The test set remains completely held-out for final evaluation.

Usage:
    python scripts/create_dev_split.py \
        --train_csv data/processed/grouped_split/train_metadata.csv \
        --output_dir data/processed/grouped_split_with_dev \
        --dev_size 0.2 \
        --random_state 42
"""

import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


def extract_base_clip_id(clip_id: str) -> str:
    """
    Extract base clip ID for grouping (same as in create_group_stratified_split.py).

    For voxconverse: remove time segment and duration suffix
        voxconverse_afjiv_35.680_1000ms → voxconverse_afjiv

    For ESC-50: remove duration suffix
        1-17742-A-12_1000ms_008 → 1-17742-A-12
    """
    # For voxconverse: remove time segment (e.g., _42.120_1000ms)
    if clip_id.startswith('voxconverse_'):
        parts = clip_id.split('_')
        if len(parts) >= 2:
            # Return speaker ID only: voxconverse_SPEAKER
            return f"{parts[0]}_{parts[1]}"

    # For ESC-50: remove duration suffix (e.g., _1000ms_039)
    if '_1000ms_' in clip_id or '_200ms_' in clip_id:
        parts = clip_id.rsplit('_', 2)
        if len(parts) == 3:
            # Return base clip: 1-17742-A-12
            return parts[0]

    # Fallback: return as-is
    return clip_id


def main():
    parser = argparse.ArgumentParser(description='Create dev split from train set')
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to train metadata CSV')
    parser.add_argument('--output_dir', type=str, default='data/processed/grouped_split_with_dev',
                       help='Output directory for train/dev split')
    parser.add_argument('--dev_size', type=float, default=0.2,
                       help='Fraction of train to use for dev (default: 0.2 = 20%)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load train data
    print(f"Loading train data from: {args.train_csv}")
    train_df = pd.read_csv(args.train_csv)
    print(f"Total train samples: {len(train_df)}")

    # Extract base clip IDs for grouping
    print("\nExtracting base clip IDs for grouping...")
    train_df['base_clip_id'] = train_df['clip_id'].apply(extract_base_clip_id)

    # Get unique groups
    unique_groups = train_df['base_clip_id'].unique()
    n_groups = len(unique_groups)
    print(f"Unique groups (speakers/sounds): {n_groups}")

    # Determine label column
    label_col = 'ground_truth' if 'ground_truth' in train_df.columns else 'label'

    # Count per class
    class_counts = train_df[label_col].value_counts()
    print(f"\nClass distribution:")
    for label, count in class_counts.items():
        print(f"  {label}: {count}")

    # Count groups per class
    groups_by_class = train_df.groupby(label_col)['base_clip_id'].nunique()
    print(f"\nGroups per class:")
    for label, count in groups_by_class.items():
        print(f"  {label}: {count} groups")

    # Split using GroupShuffleSplit
    print(f"\nSplitting with dev_size={args.dev_size} (GroupShuffleSplit)...")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.dev_size, random_state=args.random_state)

    # Get indices
    train_idx, dev_idx = next(gss.split(
        X=train_df,
        y=train_df[label_col],
        groups=train_df['base_clip_id']
    ))

    # Create splits
    new_train_df = train_df.iloc[train_idx].copy()
    dev_df = train_df.iloc[dev_idx].copy()

    # Remove temporary column
    new_train_df = new_train_df.drop(columns=['base_clip_id'])
    dev_df = dev_df.drop(columns=['base_clip_id'])

    # Verify no overlap in base_clip_ids
    train_groups = set(train_df.iloc[train_idx]['base_clip_id'])
    dev_groups = set(train_df.iloc[dev_idx]['base_clip_id'])
    overlap = train_groups & dev_groups

    print("\n" + "="*70)
    print("SPLIT SUMMARY")
    print("="*70)

    print(f"\nNew Train Set:")
    print(f"  Total samples: {len(new_train_df)}")
    print(f"  Unique groups: {len(train_groups)}")
    train_class_counts = new_train_df[label_col].value_counts()
    for label, count in train_class_counts.items():
        print(f"  {label}: {count}")

    print(f"\nDev Set:")
    print(f"  Total samples: {len(dev_df)}")
    print(f"  Unique groups: {len(dev_groups)}")
    dev_class_counts = dev_df[label_col].value_counts()
    for label, count in dev_class_counts.items():
        print(f"  {label}: {count}")

    print(f"\nOverlap Check:")
    if len(overlap) == 0:
        print(f"  OK: Zero overlap (0 shared groups)")
    else:
        print(f"  WARNING: {len(overlap)} groups overlap!")
        print(f"  Overlapping groups: {overlap}")

    # Save splits
    train_output = output_dir / 'train_metadata.csv'
    dev_output = output_dir / 'dev_metadata.csv'

    new_train_df.to_csv(train_output, index=False)
    dev_df.to_csv(dev_output, index=False)

    print(f"\n" + "="*70)
    print("FILES SAVED")
    print("="*70)
    print(f"Train: {train_output} ({len(new_train_df)} samples)")
    print(f"Dev:   {dev_output} ({len(dev_df)} samples)")
    print()

    # Usage instructions
    print("="*70)
    print("USAGE")
    print("="*70)
    print("\n1. Train model on new train split:")
    print(f"   python scripts/finetune_qwen_audio.py \\")
    print(f"       --train_csv {train_output} \\")
    print(f"       --seed 42 \\")
    print(f"       --output_dir checkpoints/with_dev/seed_42")

    print("\n2. Get uncalibrated predictions on dev:")
    print(f"   python scripts/evaluate_with_logits.py \\")
    print(f"       --checkpoint checkpoints/with_dev/seed_42/final \\")
    print(f"       --test_csv {dev_output} \\")
    print(f"       --temperature 1.0 \\")
    print(f"       --output_csv results/dev_uncalibrated.csv")

    print("\n3. Find optimal temperature on dev:")
    print(f"   python scripts/calibrate_temperature.py \\")
    print(f"       --predictions_csv results/dev_uncalibrated.csv \\")
    print(f"       --output_temp results/optimal_temperature.txt \\")
    print(f"       --plot results/reliability_diagram_dev.png")

    print("\n4. Evaluate on test with optimal temperature:")
    print(f"   python scripts/evaluate_with_logits.py \\")
    print(f"       --checkpoint checkpoints/with_dev/seed_42/final \\")
    print(f"       --test_csv data/processed/grouped_split/test_metadata.csv \\")
    print(f"       --temperature $(cat results/optimal_temperature.txt) \\")
    print(f"       --output_csv results/test_calibrated.csv")

    print()


if __name__ == "__main__":
    main()
