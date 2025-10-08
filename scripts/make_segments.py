#!/usr/bin/env python3
"""
Create segmented dataset at target durations.

Usage:
    python scripts/make_segments.py --dataset voxconverse --split dev --durations 40 100 200
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import CONFIG
from qsm.data import create_segments, load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Create segmented dataset")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["voxconverse", "dihard", "ava_speech", "ami", "ava_activespeaker"],
        help="Dataset to process",
    )

    parser.add_argument(
        "--split", type=str, default="dev", choices=["train", "dev", "test"], help="Data split"
    )

    parser.add_argument(
        "--durations",
        nargs="+",
        type=int,
        default=CONFIG["durations_ms"],
        help="Target durations in milliseconds",
    )

    parser.add_argument(
        "--max-segments",
        type=int,
        default=1000,
        help="Maximum segments per (duration, condition) combination",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/segments/{dataset}_{split})",
    )

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading {args.dataset} ({args.split} split)...")
    frame_table = load_dataset(args.dataset, split=args.split)

    logger.info(f"Loaded {len(frame_table.data)} annotations")

    # Set output directory
    if args.output_dir is None:
        args.output_dir = Path(CONFIG["data"]["segments"]) / f"{args.dataset}_{args.split}"

    # Find audio root
    config_path = Path(__file__).parent.parent / "configs" / "datasets" / f"{args.dataset}.yaml"

    import yaml

    with open(config_path) as f:
        dataset_config = yaml.safe_load(f)

    audio_root = Path(dataset_config["audio_path"][args.split])

    logger.info(f"Audio root: {audio_root}")
    logger.info(f"Output dir: {args.output_dir}")

    # Create segments
    metadata_df = create_segments(
        frame_table=frame_table,
        audio_root=audio_root,
        output_dir=args.output_dir,
        durations_ms=args.durations,
        max_segments_per_config=args.max_segments,
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SEGMENTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total segments created: {len(metadata_df)}")
    logger.info(f"Output directory: {args.output_dir}")

    # Stats by duration
    logger.info("\nSegments by duration:")
    for duration in sorted(metadata_df["duration_ms"].unique()):
        count = len(metadata_df[metadata_df["duration_ms"] == duration])
        logger.info(f"  {duration}ms: {count} segments")

    # Stats by label
    logger.info("\nSegments by label:")
    for label in metadata_df["label"].unique():
        count = len(metadata_df[metadata_df["label"] == label])
        logger.info(f"  {label}: {count} segments")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
