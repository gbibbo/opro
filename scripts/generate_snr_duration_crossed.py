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

from qsm.audio.slicing import extract_from_padded_1000ms, pad_audio_center
from qsm.audio.noise import mix_at_snr


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
            print(f"\nWarning: File not found: {audio_path}")
            continue

        audio_1000ms, sr = sf.read(audio_path)

        # Verify sample rate
        if sr != sample_rate:
            print(f"\nWarning: Expected {sample_rate}Hz, got {sr}Hz for {clip_id}")
            continue

        # Verify length (should be ~1000ms)
        expected_samples = sample_rate  # 1000ms = 16000 samples at 16kHz
        if abs(len(audio_1000ms) - expected_samples) > 100:  # Allow 100 samples tolerance
            # Trim or pad to exactly 1000ms
            if len(audio_1000ms) > expected_samples:
                audio_1000ms = audio_1000ms[:expected_samples]
            else:
                audio_1000ms = np.pad(audio_1000ms, (0, expected_samples - len(audio_1000ms)), mode='constant')

        # Pad to 2000ms (required by extract_from_padded_1000ms)
        audio_2000ms = pad_audio_center(audio_1000ms, target_ms=2000, sr=sample_rate)

        # Generate all SNR×Duration combinations
        for duration_ms in durations_ms:
            for snr_db in snr_db_levels:
                # 1. Extract duration segment (from center of 2000ms container)
                audio_segment = extract_from_padded_1000ms(
                    audio_2000ms,
                    duration_ms=duration_ms,
                    sr=sample_rate
                )

                # 2. Add noise at target SNR
                # mix_at_snr returns (audio_mixed, metadata_dict)
                audio_noisy, snr_metadata = mix_at_snr(
                    audio_segment,
                    snr_db=snr_db,
                    sr=sample_rate,
                    padding_ms=2000,
                    effective_dur_ms=duration_ms,
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
                    "rms_signal": snr_metadata.get("rms_signal"),
                    "rms_noise": snr_metadata.get("rms_noise"),
                    "measured_snr_db": snr_metadata.get("measured_snr_db"),
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

    # SNR accuracy check
    print("\nSNR accuracy (measured vs target):")
    for target_snr in sorted(snr_db_levels):
        subset = metadata_df[metadata_df["snr_db"] == target_snr]
        if len(subset) > 0:
            measured_mean = subset["measured_snr_db"].mean()
            measured_std = subset["measured_snr_db"].std()
            error = abs(measured_mean - target_snr)
            print(f"  Target {target_snr:+3d} dB → Measured {measured_mean:+6.2f} ± {measured_std:.2f} dB (error: {error:.2f} dB)")

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
