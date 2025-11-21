#!/usr/bin/env python3
"""
Sprint 6: Stratified split generation for reproducible evaluation.

Creates dev/test splits with stratification by:
- duration_ms
- snr_db
- band_filter
- T60_bin
- dataset (from clip_id)
- label (SPEECH/NONSPEECH)

Ensures same clip_id stays in same split across all 20 variants.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def extract_dataset_from_clip_id(clip_id: str) -> str:
    """Extract dataset name from clip_id."""
    # voxconverse_xxx_xxx -> voxconverse
    # 1-172649-B-40_1000ms_010 -> esc50
    if clip_id.startswith("voxconverse"):
        return "voxconverse"
    else:
        return "esc50"


def create_stratification_key(row: pd.Series) -> str:
    """
    Create stratification key for a variant.

    Format: {dataset}_{label}_{variant_type}_{condition}
    """
    dataset = extract_dataset_from_clip_id(row["clip_id"])
    label = row["ground_truth"]
    variant_type = row["variant_type"]

    # Condition depends on variant type
    if variant_type == "duration":
        # Bin durations: short (≤100ms), medium (≤500ms), long (>500ms)
        dur = row["duration_ms"]
        if dur <= 100:
            condition = "dur_short"
        elif dur <= 500:
            condition = "dur_medium"
        else:
            condition = "dur_long"

    elif variant_type == "snr":
        # Bin SNR: low (≤0), medium (≤10), high (>10)
        snr = row["snr_db"]
        if snr <= 0:
            condition = "snr_low"
        elif snr <= 10:
            condition = "snr_medium"
        else:
            condition = "snr_high"

    elif variant_type == "band":
        condition = f"band_{row['band_filter']}"

    elif variant_type == "rir":
        condition = f"rir_{row['T60_bin']}"

    else:
        condition = "unknown"

    return f"{dataset}_{label}_{variant_type}_{condition}"


def stratified_split_by_clip(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create stratified dev/test split ensuring clips stay together.

    Steps:
    1. Group by clip_id to get one row per clip
    2. Create stratification key based on clip's variants
    3. Split clips into dev/test
    4. Assign all variants of each clip to same split

    Args:
        df: DataFrame with all variants
        test_size: Fraction for test set (default 0.2 = 80/20 split)
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with added 'split' column
    """
    print("Creating stratified dev/test split...")
    print(f"  Random seed: {random_state}")
    print(f"  Test size: {test_size:.1%}")

    # Add ground_truth if not present
    if "ground_truth" not in df.columns:
        df["ground_truth"] = df["label"].str.replace("-", "").str.replace("_", "").str.upper()

    # Get unique clips with their stratification characteristics
    clip_info = []

    for clip_id in df["clip_id"].unique():
        clip_variants = df[df["clip_id"] == clip_id]

        # Get label (same for all variants of this clip)
        label = clip_variants["ground_truth"].iloc[0]
        dataset = extract_dataset_from_clip_id(clip_id)

        # Count variant types present
        variant_counts = clip_variants["variant_type"].value_counts().to_dict()

        # Create stratification key based on dataset and label
        # (we ensure balanced representation across datasets and labels)
        strat_key = f"{dataset}_{label}"

        clip_info.append({
            "clip_id": clip_id,
            "label": label,
            "dataset": dataset,
            "strat_key": strat_key,
            "n_variants": len(clip_variants),
        })

    clip_df = pd.DataFrame(clip_info)

    print(f"\nClip statistics:")
    print(f"  Total clips: {len(clip_df)}")
    print(f"  By dataset and label:")
    print(clip_df.groupby(["dataset", "label"]).size())

    # Manual stratified split (per strat_key to handle imbalanced groups)
    train_clips = []
    test_clips = []

    np.random.seed(random_state)

    for strat_key in sorted(clip_df["strat_key"].unique()):
        subset = clip_df[clip_df["strat_key"] == strat_key]["clip_id"].values
        n_total = len(subset)
        n_test = max(1, int(n_total * test_size))  # At least 1 clip for test

        print(f"  {strat_key}: {n_total} clips -> {n_total - n_test} train / {n_test} test")

        # Shuffle and split
        shuffled = subset.copy()
        np.random.shuffle(shuffled)

        test_clips.extend(shuffled[:n_test])
        train_clips.extend(shuffled[n_test:])

    # Assign split to all variants
    df["split"] = df["clip_id"].map(
        lambda cid: "test" if cid in test_clips else "dev"
    )

    # Verify split
    print(f"\nSplit distribution:")
    print(df.groupby(["split", "ground_truth"]).size())
    print(f"\nVariants by split:")
    print(df["split"].value_counts())

    # Check stratification quality
    print(f"\nStratification check (clips per split):")
    for split in ["dev", "test"]:
        split_clips = df[df["split"] == split]["clip_id"].unique()
        split_clip_df = clip_df[clip_df["clip_id"].isin(split_clips)]
        print(f"\n{split.upper()}:")
        print(split_clip_df.groupby(["dataset", "label"]).size())

    return df


def main():
    parser = argparse.ArgumentParser(description="Create stratified dev/test split")
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=Path("data/processed/conditions_final/conditions_manifest.parquet"),
        help="Input manifest path",
    )
    parser.add_argument(
        "--output_manifest",
        type=Path,
        default=Path("data/processed/conditions_final/conditions_manifest_split.parquet"),
        help="Output manifest with split column",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction for test set (default 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Load manifest
    print(f"Loading manifest from: {args.input_manifest}")
    df = pd.read_parquet(args.input_manifest)
    print(f"Loaded {len(df)} samples from {df['clip_id'].nunique()} clips")

    # Create split
    df_split = stratified_split_by_clip(
        df,
        test_size=args.test_size,
        random_state=args.seed,
    )

    # Save
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    df_split.to_parquet(args.output_manifest, index=False)

    print(f"\nSaved split manifest to: {args.output_manifest}")
    print(f"Added 'split' column with values: {df_split['split'].unique()}")

    # Save metadata
    metadata = {
        "random_seed": args.seed,
        "test_size": args.test_size,
        "total_samples": len(df_split),
        "total_clips": df_split["clip_id"].nunique(),
        "dev_samples": len(df_split[df_split["split"] == "dev"]),
        "test_samples": len(df_split[df_split["split"] == "test"]),
        "dev_clips": df_split[df_split["split"] == "dev"]["clip_id"].nunique(),
        "test_clips": df_split[df_split["split"] == "test"]["clip_id"].nunique(),
    }

    metadata_path = args.output_manifest.with_suffix(".metadata.json")
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to: {metadata_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
