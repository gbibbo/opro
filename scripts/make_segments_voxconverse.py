#!/usr/bin/env python3
"""
Generate sliced audio segments from VoxConverse dataset with safety buffers.

VoxConverse provides RTTM annotations with speaker diarization timestamps.
We convert these to SPEECH/NONSPEECH segments and apply 1s safety buffers.

Usage:
    python scripts/make_segments_voxconverse.py
    python scripts/make_segments_voxconverse.py --durations 100 200 500 1000
    python scripts/make_segments_voxconverse.py --max-per-duration 30
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import CONFIG
from qsm.data.loaders import FrameTable, load_rttm_dataset
from qsm.data.slicing import create_segments

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_voxconverse_multi_rttm(
    rttm_dir: Path,
    audio_dir: Path,
    dataset_name: str = "voxconverse",
    split: str = "dev",
) -> FrameTable:
    """
    Load VoxConverse RTTM annotations from multiple files.

    VoxConverse has one RTTM file per recording.

    Args:
        rttm_dir: Directory containing .rttm files
        audio_dir: Directory containing .wav files (to filter available recordings)
        dataset_name: Name of dataset
        split: Data split

    Returns:
        Combined FrameTable with all recordings
    """
    logger.info(f"Loading VoxConverse RTTMs from {rttm_dir}")

    # Get available audio files
    available_audios = {f.stem for f in audio_dir.glob("*.wav")}
    logger.info(f"Found {len(available_audios)} audio files: {sorted(available_audios)}")

    # Find matching RTTM files
    rttm_files = [f for f in rttm_dir.glob("*.rttm") if f.stem in available_audios]
    logger.info(f"Found {len(rttm_files)} matching RTTM files")

    if not rttm_files:
        raise FileNotFoundError(f"No RTTM files found matching audio files in {rttm_dir}")

    # Load each RTTM file and combine
    all_frames = []

    for rttm_file in sorted(rttm_files):
        logger.info(f"Loading {rttm_file.name}...")

        # Load this RTTM (with nonspeech generation)
        frame_table = load_rttm_dataset(
            rttm_path=rttm_file,
            audio_dir=audio_dir,  # Provide audio directory for full duration
            dataset_name=dataset_name,
            split=split,
            include_nonspeech=True,  # Generate NONSPEECH from gaps
        )

        all_frames.append(frame_table.data)

        # Show stats for this file
        speech_count = (frame_table.data["label"] == "SPEECH").sum()
        nonspeech_count = (frame_table.data["label"] == "NONSPEECH").sum()
        logger.info(
            f"  {rttm_file.stem}: {speech_count} SPEECH intervals, "
            f"{nonspeech_count} NONSPEECH intervals"
        )

    # Combine all dataframes
    combined_df = pd.concat(all_frames, ignore_index=True)

    logger.info(f"Total intervals loaded: {len(combined_df)}")
    logger.info(f"  SPEECH: {(combined_df['label'] == 'SPEECH').sum()}")
    logger.info(f"  NONSPEECH: {(combined_df['label'] == 'NONSPEECH').sum()}")

    return FrameTable(data=combined_df)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sliced segments from VoxConverse with safety buffers"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--durations",
        nargs="+",
        type=int,
        default=[100, 200, 500, 1000],
        help="Segment durations in milliseconds (default: 100 200 500 1000)",
    )
    parser.add_argument(
        "--max-per-duration",
        type=int,
        default=20,
        help="Max segments per duration per label (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/segments/voxconverse/{split})",
    )

    args = parser.parse_args()

    # Get paths from config
    data_root = Path(CONFIG["data"]["root"])
    rttm_dir = data_root / "raw" / "voxconverse" / args.split
    audio_root = data_root / "raw" / "voxconverse" / "audio" / args.split

    if args.output_dir is None:
        args.output_dir = data_root / "segments" / "voxconverse" / args.split

    logger.info("=" * 80)
    logger.info("VOXCONVERSE SEGMENTATION WITH SAFETY BUFFERS")
    logger.info("=" * 80)
    logger.info(f"Split: {args.split}")
    logger.info(f"RTTM dir: {rttm_dir}")
    logger.info(f"Audio root: {audio_root}")
    logger.info(f"Durations: {args.durations} ms")
    logger.info(f"Max per duration per label: {args.max_per_duration}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")

    # Check paths
    if not rttm_dir.exists():
        logger.error(f"RTTM directory not found: {rttm_dir}")
        sys.exit(1)

    if not audio_root.exists():
        logger.error(f"Audio directory not found: {audio_root}")
        sys.exit(1)

    # Load annotations
    logger.info("Loading VoxConverse RTTM annotations...")
    frame_table = load_voxconverse_multi_rttm(
        rttm_dir=rttm_dir,
        audio_dir=audio_root,
        dataset_name="voxconverse",
        split=args.split,
    )

    logger.info(
        f"Loaded {len(frame_table.data)} intervals from {frame_table.data['uri'].nunique()} recordings"
    )
    logger.info("")

    # Create segments with safety buffers (hardcoded in slicing.py)
    logger.info("Creating segments with 1-second safety buffers (hardcoded in slicing logic)...")
    metadata_df = create_segments(
        frame_table=frame_table,
        audio_root=audio_root,
        output_dir=args.output_dir,
        durations_ms=args.durations,
        max_segments_per_config=args.max_per_duration,
    )

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ VOXCONVERSE SEGMENTATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total segments created: {len(metadata_df)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")

    # Per-duration breakdown
    for duration_ms in sorted(args.durations):
        subset = metadata_df[metadata_df["duration_ms"] == duration_ms]
        speech = (subset["label"] == "SPEECH").sum()
        nonspeech = (subset["label"] == "NONSPEECH").sum()
        logger.info(
            f"{duration_ms:4d}ms: {len(subset):3d} total ({speech} SPEECH + {nonspeech} NONSPEECH)"
        )

    logger.info("")
    logger.info("Verify with:")
    logger.info(f"  python scripts/verify_segments.py --segments-dir {args.output_dir} --list")


if __name__ == "__main__":
    main()
