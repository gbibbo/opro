#!/usr/bin/env python3
"""
Generate factorial SNR×Duration dataset for proper psychometric curve fitting.

Design:
- 4 durations: {20, 80, 200, 1000} ms (short, medium-short, medium-long, long)
- 8 SNR levels: {-20, -15, -10, -5, 0, +5, +10, +20} dB
- Total: 4 × 8 = 32 conditions per clip

This allows fitting:
1. SNR curves stratified by duration (4 separate curves)
2. GLMM with SNR×Duration interaction

Note: This is a mini-dataset for validation. If successful, extend to full pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

# Import audio processing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.audio.duration import apply_duration_truncation
from qsm.audio.noise import add_noise_at_snr


def generate_snr_duration_crossed(
    input_file: Path,
    output_dir: Path,
    durations_ms: List[int] = [20, 80, 200, 1000],
    snr_db_levels: List[int] = [-20, -15, -10, -5, 0, 5, 10, 20],
    sample_rate: int = 16000,
    seed: int = 42,
):
    """
    Generate factorial SNR×Duration dataset.

    Args:
        input_file: Path to parquet or CSV with clip metadata
        output_dir: Output directory for crossed conditions
        durations_ms: List of duration levels
        snr_db_levels: List of SNR levels
        sample_rate: Audio sample rate
        seed: Random seed for noise generation
    """
    # Load metadata
    print(f"Loading metadata from: {input_file}")
    if input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} clips")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Storage for metadata
    all_metadata = []

    np.random.seed(seed)

    print(f"\nGenerating {len(durations_ms)} durations × {len(snr_db_levels)} SNR levels = {len(durations_ms) * len(snr_db_levels)} conditions per clip")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing clips"):
        audio_path_str = str(row["audio_path"]).replace("\\", "/")
        audio_path = Path(audio_path_str)
        clip_id = row["clip_id"]
        label = row["ground_truth"]

        # Load 1000ms audio
        if not audio_path.exists():
            print(f"Warning: File not found: {audio_path}")
            continue

        audio, sr = sf.read(audio_path)

        # Verify sample rate
        if sr != sample_rate:
            print(f"Warning: Expected {sample_rate}Hz, got {sr}Hz for {clip_id}")
            continue

        # Audio should be 1000ms (may vary slightly)
        expected_samples = sample_rate  # 1000ms = 16000 samples at 16kHz
        if abs(len(audio) - expected_samples) > 100:  # Allow 100 samples tolerance
            print(f"Warning: Expected ~1000ms ({expected_samples} samples), got {len(audio)} samples for {clip_id}")
            # Trim or pad to exactly 1000ms
            if len(audio) > expected_samples:
                audio = audio[:expected_samples]
            else:
                audio = np.pad(audio, (0, expected_samples - len(audio)), mode='constant')

        # Generate all SNR×Duration combinations
        for duration_ms in durations_ms:
            for snr_db in snr_db_levels:
                # 1. Apply duration truncation
                audio_truncated = apply_duration_truncation(
                    audio,
                    target_duration_ms=duration_ms,
                    sample_rate=sample_rate,
                    padding_duration_ms=1000,  # Keep 1000ms container
                )

                # 2. Add noise at target SNR
                audio_noisy = add_noise_at_snr(
                    audio_truncated,
                    target_snr_db=snr_db,
                    sample_rate=sample_rate,
                    noise_type="white",
                    seed=np.random.randint(0, 2**31),
                )

                # 3. Save
                variant_name = f"dur{duration_ms}ms_snr{snr_db:+d}dB"
                output_path = output_dir / f"{clip_id}_{variant_name}.wav"
                sf.write(output_path, audio_noisy, sample_rate)

                # 4. Store metadata
                all_metadata.append({
                    "clip_id": clip_id,
                    "variant_name": variant_name,
                    "variant_type": "snr_duration_crossed",
                    "duration_ms": duration_ms,
                    "snr_db": snr_db,
                    "audio_path": str(output_path.as_posix()),
                    "ground_truth": label,
                    "dataset": row.get("dataset", "unknown"),
                })

    # Save metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\nGenerated {len(metadata_df)} audio files ({len(durations_ms)}×{len(snr_db_levels)} conditions × {len(df)} clips)")
    print(f"Saved to: {output_dir}")
    print(f"Metadata: {metadata_path}")

    # Summary statistics
    print("\nDataset summary:")
    print(f"  Clips: {metadata_df['clip_id'].nunique()}")
    print(f"  Conditions: {len(durations_ms)} durations × {len(snr_db_levels)} SNR = {len(durations_ms) * len(snr_db_levels)}")
    print(f"  Total samples: {len(metadata_df)}")
    print(f"  Labels: {metadata_df['ground_truth'].value_counts().to_dict()}")

    return metadata_df


def main():
    parser = argparse.ArgumentParser(description="Generate factorial SNR×Duration dataset")
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("data/processed/subset_20clips_for_crossed.csv"),
        help="Path to CSV/parquet with clip metadata (must have: clip_id, audio_path, ground_truth)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/snr_duration_crossed"),
        help="Output directory",
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[20, 80, 200, 1000],
        help="Duration levels in ms",
    )
    parser.add_argument(
        "--snr_levels",
        type=int,
        nargs="+",
        default=[-20, -15, -10, -5, 0, 5, 10, 20],
        help="SNR levels in dB",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("="*60)
    print("GENERATE FACTORIAL SNR×DURATION DATASET")
    print("="*60)
    print(f"\nDurations: {args.durations} ms")
    print(f"SNR levels: {args.snr_levels} dB")
    print(f"Total conditions: {len(args.durations)} × {len(args.snr_levels)} = {len(args.durations) * len(args.snr_levels)}")

    # Generate dataset
    metadata_df = generate_snr_duration_crossed(
        input_file=args.input_file,
        output_dir=args.output_dir,
        durations_ms=args.durations,
        snr_db_levels=args.snr_levels,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )

    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
