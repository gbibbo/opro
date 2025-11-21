#!/usr/bin/env python3
"""
Create augmented dataset with improved NONSPEECH diversity.

Combines:
- Existing SPEECH samples (from VoxConverse, LibriSpeech, etc.)
- Existing NONSPEECH (ESC-50)
- NEW: Music clips
- NEW: Silence/noise clips

Target distribution for NONSPEECH:
- 40% ESC-50 (environmental sounds)
- 40% Music
- 20% Silence/noise

Usage:
    python scripts/create_augmented_dataset.py \
        --original_train data/processed/experimental_variants/train_metadata.csv \
        --music_dir data/raw/music \
        --noise_dir data/raw/silence_noise \
        --output_dir data/processed/augmented_dataset
"""

import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np


def load_original_dataset(csv_path):
    """Load original training dataset."""
    df = pd.read_csv(csv_path)
    print(f"Original dataset: {len(df)} samples")
    print(f"  SPEECH: {len(df[df['ground_truth'] == 'SPEECH'])}")
    print(f"  NONSPEECH: {len(df[df['ground_truth'] == 'NONSPEECH'])}")
    return df


def create_metadata_for_new_audio(audio_dir, label, source_type):
    """Create metadata entries for new audio files."""
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        print(f"WARNING: {audio_dir} does not exist")
        return []

    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} {source_type} files in {audio_dir}")

    entries = []
    for audio_file in audio_files:
        entries.append({
            'clip_id': audio_file.stem,
            'audio_path': str(audio_file.relative_to(Path.cwd())),
            'ground_truth': label,
            'dataset': source_type,
            'duration_ms': 2000,  # All augmented clips are 2s
            'snr_db': None,  # Original, no added noise
        })

    return entries


def balance_nonspeech(esc50_df, music_entries, noise_entries, target_total):
    """Balance NONSPEECH according to target distribution."""
    # Target: 40% ESC-50, 40% Music, 20% Noise
    n_esc50 = int(target_total * 0.40)
    n_music = int(target_total * 0.40)
    n_noise = target_total - n_esc50 - n_music  # Remaining

    print(f"\nBalancing NONSPEECH (target={target_total}):")
    print(f"  ESC-50: {n_esc50} (40%)")
    print(f"  Music: {n_music} (40%)")
    print(f"  Noise: {n_noise} (20%)")

    # Sample from each source
    if len(esc50_df) > n_esc50:
        esc50_sampled = esc50_df.sample(n=n_esc50, random_state=42)
    else:
        print(f"  WARNING: Not enough ESC-50 samples ({len(esc50_df)} < {n_esc50})")
        esc50_sampled = esc50_df

    if len(music_entries) > n_music:
        music_sampled = pd.DataFrame(music_entries).sample(n=n_music, random_state=42)
    else:
        print(f"  WARNING: Not enough music samples ({len(music_entries)} < {n_music})")
        music_sampled = pd.DataFrame(music_entries)

    if len(noise_entries) > n_noise:
        noise_sampled = pd.DataFrame(noise_entries).sample(n=n_noise, random_state=42)
    else:
        print(f"  WARNING: Not enough noise samples ({len(noise_entries)} < {n_noise})")
        noise_sampled = pd.DataFrame(noise_entries)

    # Combine
    nonspeech_df = pd.concat([esc50_sampled, music_sampled, noise_sampled], ignore_index=True)

    print(f"\nFinal NONSPEECH: {len(nonspeech_df)} samples")
    print(nonspeech_df['dataset'].value_counts())

    return nonspeech_df


def main():
    parser = argparse.ArgumentParser(description="Create augmented dataset")
    parser.add_argument("--original_train", type=str, required=True,
                        help="Original training CSV")
    parser.add_argument("--music_dir", type=str, default="data/raw/music",
                        help="Directory with music clips")
    parser.add_argument("--noise_dir", type=str, default="data/raw/silence_noise",
                        help="Directory with noise clips")
    parser.add_argument("--output_dir", type=str, default="data/processed/augmented_dataset",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CREATING AUGMENTED DATASET")
    print("=" * 60)

    # Load original dataset
    original_df = load_original_dataset(args.original_train)

    # Split SPEECH and NONSPEECH
    speech_df = original_df[original_df['ground_truth'] == 'SPEECH'].copy()
    esc50_df = original_df[original_df['ground_truth'] == 'NONSPEECH'].copy()

    print(f"\nOriginal SPEECH: {len(speech_df)}")
    print(f"Original NONSPEECH (ESC-50): {len(esc50_df)}")

    # Load new audio metadata
    print("\n" + "=" * 60)
    print("LOADING NEW NONSPEECH SOURCES")
    print("=" * 60)

    music_entries = create_metadata_for_new_audio(args.music_dir, 'NONSPEECH', 'music')
    noise_entries = create_metadata_for_new_audio(args.noise_dir, 'NONSPEECH', 'noise')

    # Balance NONSPEECH to match SPEECH count
    target_nonspeech = len(speech_df)

    nonspeech_df = balance_nonspeech(esc50_df, music_entries, noise_entries, target_nonspeech)

    # Combine SPEECH + NONSPEECH
    augmented_df = pd.concat([speech_df, nonspeech_df], ignore_index=True)

    # Shuffle
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("FINAL AUGMENTED DATASET")
    print("=" * 60)
    print(f"Total samples: {len(augmented_df)}")
    print(f"  SPEECH: {len(augmented_df[augmented_df['ground_truth'] == 'SPEECH'])}")
    print(f"  NONSPEECH: {len(augmented_df[augmented_df['ground_truth'] == 'NONSPEECH'])}")
    print(f"\nNONSPEECH breakdown:")
    print(augmented_df[augmented_df['ground_truth'] == 'NONSPEECH']['dataset'].value_counts())

    # Save
    output_csv = output_dir / "train_metadata_augmented.csv"
    augmented_df.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")

    # Copy test/dev sets unchanged
    original_dir = Path(args.original_train).parent

    for split in ['test', 'dev']:
        original_split = original_dir / f"{split}_metadata.csv"
        if original_split.exists():
            output_split = output_dir / f"{split}_metadata.csv"
            shutil.copy(original_split, output_split)
            print(f"Copied {split} set to: {output_split}")

    print("\n" + "=" * 60)
    print("DONE! Next steps:")
    print("=" * 60)
    print(f"1. Review: {output_csv}")
    print(f"2. Re-train model:")
    print(f"   python scripts/finetune_qwen_audio.py \\")
    print(f"     --train_csv {output_csv} \\")
    print(f"     --output_dir checkpoints/qwen_augmented")


if __name__ == "__main__":
    main()
