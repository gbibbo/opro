#!/usr/bin/env python3
"""
Download AVA-ActiveSpeaker dataset.

NOTE: This requires manual download from Google due to license requirements.
This script provides instructions and structure but doesn't auto-download.

Official: https://github.com/cvdfoundation/ava-dataset
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import PROTOTYPE_MODE, PROTOTYPE_SAMPLES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_ava_activespeaker(data_root: Path, prototype: bool = True):
    """
    Set up AVA-ActiveSpeaker dataset structure.

    NOTE: Audio/video requires manual download from Google.

    Args:
        data_root: Root directory for data storage
        prototype: If True, create only PROTOTYPE_SAMPLES placeholders
    """
    logger.info("=" * 80)
    logger.info("📥 AVA-ACTIVESPEAKER SETUP")
    logger.info("=" * 80)

    ava_dir = data_root / "ava_activespeaker"
    ava_dir.mkdir(parents=True, exist_ok=True)

    annotations_dir = ava_dir / "annotations"
    audio_dir = ava_dir / "audio"
    video_dir = ava_dir / "video"

    annotations_dir.mkdir(exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    video_dir.mkdir(exist_ok=True)

    logger.warning("⚠️  AVA-ActiveSpeaker requires manual download:")
    logger.warning("   1. Visit: https://github.com/cvdfoundation/ava-dataset")
    logger.warning("   2. Accept license terms")
    logger.warning("   3. Download annotations CSV")
    logger.warning("   4. Download video clips")
    logger.warning(f"   5. Extract audio to: {audio_dir}")

    # Create placeholder annotation structure
    splits = ["train", "val", "test"]
    num_samples = PROTOTYPE_SAMPLES if prototype else 100

    for split in splits:
        csv_path = annotations_dir / f"{split}.csv"

        if csv_path.exists():
            logger.info(f"✅ {split}.csv already exists")
            continue

        # Create placeholder CSV with proper structure
        logger.info(f"Creating placeholder {split}.csv ({num_samples} samples)...")

        data = []
        for i in range(num_samples):
            # AVA-ActiveSpeaker format:
            # video_id, frame_timestamp, entity_box_x1, entity_box_y1,
            # entity_box_x2, entity_box_y2, label, entity_id
            data.append(
                {
                    "video_id": f"mock_video_{i % 5}",  # 5 unique videos
                    "frame_timestamp": i,
                    "entity_box_x1": 0.1,
                    "entity_box_y1": 0.1,
                    "entity_box_x2": 0.5,
                    "entity_box_y2": 0.5,
                    "label": "SPEAKING_AUDIBLE" if i % 3 == 0 else "NOT_SPEAKING",
                    "entity_id": f"person_{i % 3}",
                }
            )

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ Created {csv_path}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ AVA-ACTIVESPEAKER STRUCTURE READY")
    logger.info("=" * 80)
    logger.info(f"Annotations: {annotations_dir}")
    logger.info(f"Audio: {audio_dir} (EMPTY - requires manual download)")
    logger.info(f"Video: {video_dir} (EMPTY - requires manual download)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Download real annotations from GitHub")
    logger.info("  2. Download video clips (requires Google approval)")
    logger.info("  3. Extract audio using ffmpeg")


def main():
    parser = argparse.ArgumentParser(
        description="Setup AVA-ActiveSpeaker dataset structure (manual download required)"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw"), help="Data root directory"
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Create full placeholder structure (not prototype)",
    )

    args = parser.parse_args()

    # Determine mode
    prototype = PROTOTYPE_MODE and not args.force_full

    if prototype:
        logger.info(f"PROTOTYPE_MODE: Will create {PROTOTYPE_SAMPLES} sample placeholders")
    else:
        logger.info("FULL MODE: Will create complete placeholder structure")

    download_ava_activespeaker(args.data_root, prototype=prototype)


if __name__ == "__main__":
    main()
