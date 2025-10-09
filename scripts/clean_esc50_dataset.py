#!/usr/bin/env python3
"""
Clean ESC-50 dataset by removing ambiguous sound categories.
Remove human sounds and animal vocalizations that could be confused with speech.
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

# Categories to REMOVE (ambiguous sounds)
AMBIGUOUS_CATEGORIES = {
    # Human sounds
    "breathing",
    "clapping",
    "coughing",
    "crying_baby",
    "drinking_sipping",
    "footsteps",
    "sneezing",
    "snoring",
    # Animal vocalizations
    "cat",
    "chirping_birds",
    "crickets",  # could be confused with high-frequency speech
    "frog",
    "hen",
    "insects",  # similar to crickets
    "pig",
    "rooster",
    "sheep",
}

# Categories to KEEP (clearly non-speech environmental sounds)
CLEAN_CATEGORIES = {
    "airplane",
    "can_opening",
    "car_horn",
    "chainsaw",
    "clock_alarm",
    "clock_tick",
    "crackling_fire",
    "door_wood_creaks",
    "engine",
    "glass_breaking",
    "helicopter",
    "keyboard_typing",
    "pouring_water",
    "rain",
    "sea_waves",
    "siren",
    "thunderstorm",
    "toilet_flush",
    "train",
    "vacuum_cleaner",
    "washing_machine",
    "water_drops",
    "wind",
}


def main():
    segments_dir = Path("data/segments/esc50/nonspeech")
    metadata_path = segments_dir / "segments.parquet"

    # Load metadata
    df = pd.read_parquet(metadata_path)

    print("="*80)
    print("CLEANING ESC-50 DATASET - REMOVING AMBIGUOUS SOUNDS")
    print("="*80)

    print(f"\nOriginal dataset:")
    print(f"  Total segments: {len(df)}")
    print(f"  Total categories: {df['condition'].nunique()}")

    # Identify segments to remove
    to_remove = df[df["condition"].isin(AMBIGUOUS_CATEGORIES)]
    to_keep = df[df["condition"].isin(CLEAN_CATEGORIES)]

    print(f"\nSegments to REMOVE (ambiguous):")
    print(f"  Total: {len(to_remove)}")
    print(f"  Categories: {len(to_remove['condition'].unique())}")
    print("\n  Categories being removed:")
    for cat in sorted(AMBIGUOUS_CATEGORIES):
        count = len(to_remove[to_remove["condition"] == cat])
        if count > 0:
            print(f"    - {cat:<20} ({count} segments)")

    print(f"\nSegments to KEEP (clean):")
    print(f"  Total: {len(to_keep)}")
    print(f"  Categories: {len(to_keep['condition'].unique())}")
    print("\n  Categories being kept:")
    for cat in sorted(CLEAN_CATEGORIES):
        count = len(to_keep[to_keep["condition"] == cat])
        if count > 0:
            print(f"    - {cat:<20} ({count} segments)")

    # Verify no overlap
    overlap = AMBIGUOUS_CATEGORIES & CLEAN_CATEGORIES
    if overlap:
        print(f"\n[WARNING] Overlap detected: {overlap}")

    # Check for uncategorized
    all_categories = set(df["condition"].unique())
    uncategorized = all_categories - AMBIGUOUS_CATEGORIES - CLEAN_CATEGORIES
    if uncategorized:
        print(f"\n[WARNING] Uncategorized conditions: {uncategorized}")
        print("These will be KEPT by default")
        to_keep = df[~df["condition"].isin(AMBIGUOUS_CATEGORIES)]

    # Create backup
    backup_dir = segments_dir.parent / "esc50_backup_with_ambiguous"
    if not backup_dir.exists():
        print(f"\nCreating backup at: {backup_dir}")
        shutil.copytree(segments_dir, backup_dir)
        print("  [DONE] Backup created")

    # Delete audio files for removed segments
    print(f"\nDeleting {len(to_remove)} audio files...")
    deleted_count = 0
    for idx, row in to_remove.iterrows():
        audio_file = segments_dir / Path(row["audio_path"]).name
        if audio_file.exists():
            audio_file.unlink()
            deleted_count += 1

    print(f"  [DONE] Deleted {deleted_count} files")

    # Save cleaned metadata
    cleaned_metadata_path = segments_dir / "segments.parquet"
    to_keep.to_parquet(cleaned_metadata_path, index=False)

    print(f"\nSaved cleaned metadata:")
    print(f"  Path: {cleaned_metadata_path}")
    print(f"  Segments: {len(to_keep)}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Original:  {len(df)} segments, {df['condition'].nunique()} categories")
    print(f"Cleaned:   {len(to_keep)} segments, {to_keep['condition'].nunique()} categories")
    print(f"Removed:   {len(to_remove)} segments ({len(to_remove)/len(df)*100:.1f}%)")

    # Distribution by duration
    print("\nDistribution by duration (cleaned dataset):")
    duration_counts = to_keep.groupby("duration_ms").size().sort_index()
    for duration, count in duration_counts.items():
        print(f"  {duration}ms: {count} segments")


if __name__ == "__main__":
    main()
