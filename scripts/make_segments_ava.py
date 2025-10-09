#!/usr/bin/env python3
"""
Generate sliced audio segments from AVA-Speech with safety buffers.

This script creates fixed-duration audio segments (20ms to 1000ms) from
AVA-Speech dataset, ensuring 100% label confidence by adding 1-second
safety buffers at interval boundaries.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import CONFIG
from qsm.data.loaders import load_ava_speech
from qsm.data.slicing import create_segments

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Create sliced audio segments with 1s safety buffers"
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100, 150, 200, 300, 500, 1000],
        help="Target durations in milliseconds",
    )
    parser.add_argument(
        "--max-per-duration",
        type=int,
        default=50,
        help="Maximum segments per duration/label combination",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for segments",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("AVA-SPEECH SEGMENT SLICING WITH SAFETY BUFFERS")
    logger.info("=" * 80)
    logger.info(f"Split: {args.split}")
    logger.info(f"Durations: {args.durations} ms")
    logger.info(f"Max per duration: {args.max_per_duration}")
    logger.info("Safety buffer: 1.0 seconds at start/end of each interval")
    logger.info("")
    logger.info("IMPORTANT: Safety buffers ensure 100% label confidence")
    logger.info("  - NO_SPEECH regions: exclude 1s before transition to SPEECH")
    logger.info("  - SPEECH regions: exclude 1s after start and 1s before end")
    logger.info("")

    # Load dataset
    data_root = Path(CONFIG["data"]["root"])

    logger.info(f"Loading AVA-Speech annotations...")
    annotations_path = (
        data_root / "raw" / "ava-speech" / "annotations" / "ava_speech_labels_v1.csv"
    )
    frame_table = load_ava_speech(annotations_path, split=args.split)
    audio_root = data_root / "raw" / "ava-speech" / "audio"

    # Filter to only URIs that have audio files
    available_audios = [f.stem for f in audio_root.glob("*.wav")]
    logger.info(f"Available audio files: {available_audios}")

    frame_table.data = frame_table.data[frame_table.data["uri"].isin(available_audios)]

    logger.info(f"✅ Loaded {len(frame_table.data)} intervals (filtered to available audio)")
    logger.info(f"   Speech: {len(frame_table.speech_segments)}")
    logger.info(f"   Non-speech: {len(frame_table.nonspeech_segments)}")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = data_root / "segments" / "ava_speech" / args.split

    logger.info(f"\nOutput directory: {args.output_dir}")

    # Create segments
    logger.info("\n" + "=" * 80)
    logger.info("CREATING SEGMENTS WITH SAFETY BUFFERS")
    logger.info("=" * 80)

    metadata_df = create_segments(
        frame_table=frame_table,
        audio_root=audio_root,
        output_dir=args.output_dir,
        durations_ms=args.durations,
        max_segments_per_config=args.max_per_duration,
    )

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total segments created: {len(metadata_df)}")

    if len(metadata_df) == 0:
        logger.error("❌ No segments were created!")
        logger.error("   Check that audio files exist and intervals are long enough (>2s after buffers)")
        sys.exit(1)

    logger.info("\nSegments by duration:")
    duration_counts = metadata_df["duration_ms"].value_counts().sort_index()
    for duration_ms, count in duration_counts.items():
        logger.info(f"  {duration_ms:4d}ms: {count:4d} segments")

    logger.info("\nSegments by label:")
    label_counts = metadata_df["label"].value_counts()
    for label, count in label_counts.items():
        logger.info(f"  {label:12s}: {count:4d} segments")

    if "condition" in metadata_df.columns:
        logger.info("\nSegments by condition:")
        condition_counts = metadata_df["condition"].value_counts()
        for condition, count in condition_counts.items():
            if pd.notna(condition):
                logger.info(f"  {str(condition):12s}: {count:4d} segments")

    logger.info("\n" + "=" * 80)
    logger.info("✅ SEGMENT CREATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Segments saved to: {args.output_dir}")
    logger.info(f"Metadata: {args.output_dir / 'segments_metadata.parquet'}")
    logger.info(f"JSONL: {args.output_dir / 'segments_metadata.jsonl'}")

    # Show some examples
    logger.info("\nExample segments (first 5 of each label):")
    for label in ["SPEECH", "NONSPEECH"]:
        label_samples = metadata_df[metadata_df["label"] == label].head(5)
        logger.info(f"\n{label}:")
        for idx, row in label_samples.iterrows():
            filename = Path(row["audio_path"]).name
            logger.info(
                f"  {filename} | {row['start_s']:.2f}-{row['end_s']:.2f}s | "
                f"{row['duration_ms']}ms | condition={row.get('condition', 'N/A')}"
            )

    logger.info("\nTo verify samples, run:")
    logger.info(f"  python scripts/verify_segments.py --segments-dir {args.output_dir}")


if __name__ == "__main__":
    main()
