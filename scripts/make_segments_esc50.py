#!/usr/bin/env python3
"""
Generate NONSPEECH segments from ESC-50 dataset.

ESC-50 contains 2,000 environmental audio clips (5 seconds each) across 50 categories,
all guaranteed to be free of human speech.
"""

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import soundfile as sf
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.data.slicing import SegmentMetadata


def extract_segments_from_esc50(
    esc50_audio_dir: Path,
    esc50_meta_csv: Path,
    output_dir: Path,
    durations_ms: list[int],
    max_segments_per_duration: int,
    seed: int = 42,
) -> list[SegmentMetadata]:
    """Extract NONSPEECH segments from ESC-50 audio files.

    Args:
        esc50_audio_dir: Directory containing ESC-50 audio files
        esc50_meta_csv: Path to esc50.csv metadata file
        output_dir: Output directory for segments
        durations_ms: List of durations to generate (in milliseconds)
        max_segments_per_duration: Maximum segments per duration
        seed: Random seed for reproducibility

    Returns:
        List of segment metadata
    """
    random.seed(seed)

    # Load metadata
    df = pd.read_csv(esc50_meta_csv)
    logger.info(f"Loaded ESC-50 metadata: {len(df)} files across {df['category'].nunique()} categories")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    all_segments = []

    for duration_ms in durations_ms:
        logger.info(f"Processing duration: {duration_ms}ms")

        duration_s = duration_ms / 1000.0
        segments_created = 0

        # Shuffle files to get random distribution across categories
        shuffled_files = df.sample(frac=1, random_state=seed).to_dict('records')

        for file_info in shuffled_files:
            if segments_created >= max_segments_per_duration:
                break

            audio_path = esc50_audio_dir / file_info['filename']

            if not audio_path.exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue

            # Get audio info
            try:
                info = sf.info(audio_path)
            except Exception as e:
                logger.warning(f"Failed to read {audio_path}: {e}")
                continue

            # ESC-50 files are 5 seconds - we can extract from random position
            max_start = info.duration - duration_s

            if max_start <= 0:
                # Duration too long for this file
                continue

            # Random start position within the file
            start_s = random.uniform(0, max_start)
            end_s = start_s + duration_s

            # Read segment
            try:
                start_frame = int(start_s * info.samplerate)
                num_frames = int(duration_s * info.samplerate)

                audio_data, sr = sf.read(
                    audio_path,
                    start=start_frame,
                    frames=num_frames,
                    dtype='float32'
                )

                # Create output filename
                segment_id = f"{file_info['filename'].replace('.wav', '')}_{duration_ms}ms_{segments_created:03d}"
                output_path = output_dir / f"{segment_id}.wav"

                # Write segment
                sf.write(output_path, audio_data, sr)

                # Create metadata
                metadata = SegmentMetadata(
                    uri=file_info['filename'],
                    start_s=start_s,
                    end_s=end_s,
                    duration_ms=duration_ms,
                    label="NONSPEECH",
                    dataset="esc50",
                    split="nonspeech",
                    condition=file_info['category'],  # ESC-50 category (e.g., "rain", "dog", "train")
                    audio_path=str(output_path),
                )

                all_segments.append(metadata)
                segments_created += 1

            except Exception as e:
                logger.warning(f"Failed to extract segment from {audio_path}: {e}")
                continue

        logger.info(f"Created {segments_created} segments for {duration_ms}ms")

    return all_segments


def main():
    parser = argparse.ArgumentParser(
        description="Generate NONSPEECH segments from ESC-50 dataset"
    )
    parser.add_argument(
        "--esc50-dir",
        type=Path,
        default=Path("data/ESC-50"),
        help="ESC-50 dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/segments/esc50/nonspeech"),
        help="Output directory for segments",
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 200, 500, 1000],
        help="Durations in milliseconds",
    )
    parser.add_argument(
        "--max-per-duration",
        type=int,
        default=40,
        help="Maximum segments per duration",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Paths
    esc50_audio_dir = args.esc50_dir / "audio"
    esc50_meta_csv = args.esc50_dir / "meta" / "esc50.csv"

    # Validate paths
    if not esc50_audio_dir.exists():
        raise FileNotFoundError(f"ESC-50 audio directory not found: {esc50_audio_dir}")
    if not esc50_meta_csv.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {esc50_meta_csv}")

    logger.info(f"ESC-50 audio directory: {esc50_audio_dir}")
    logger.info(f"ESC-50 metadata: {esc50_meta_csv}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Durations: {args.durations}")
    logger.info(f"Max per duration: {args.max_per_duration}")

    # Generate segments
    segments = extract_segments_from_esc50(
        esc50_audio_dir=esc50_audio_dir,
        esc50_meta_csv=esc50_meta_csv,
        output_dir=args.output_dir,
        durations_ms=args.durations,
        max_segments_per_duration=args.max_per_duration,
        seed=args.seed,
    )

    # Save metadata
    metadata_path = args.output_dir / "segments.parquet"
    df = pd.DataFrame([asdict(s) for s in segments])
    df.to_parquet(metadata_path, index=False)

    logger.info(f"Total segments created: {len(segments)}")
    logger.info(f"Metadata saved to: {metadata_path}")

    # Print statistics
    logger.info("\nSegments by duration:")
    duration_counts = df.groupby("duration_ms").size()
    for duration_ms, count in duration_counts.items():
        logger.info(f"  {duration_ms}ms: {count} segments")

    logger.info("\nTop 10 categories used:")
    category_counts = df['condition'].value_counts().head(10)
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} segments")


if __name__ == "__main__":
    main()
