#!/usr/bin/env python3
"""
Test dataset loaders with the downloaded data.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import CONFIG
from qsm.data.loaders import load_ava_speech

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("TESTING DATASET LOADERS")
    logger.info("=" * 80)

    data_root = Path(CONFIG["data"]["root"])

    # Test AVA-Speech
    logger.info("\n1. Testing AVA-Speech loader...")
    ava_annotations = data_root / "raw" / "ava-speech" / "annotations" / "ava_speech_labels_v1.csv"

    if ava_annotations.exists():
        try:
            frame_table = load_ava_speech(ava_annotations, split="train")
            logger.info(f"✅ Loaded {len(frame_table.data)} intervals from AVA-Speech")
            logger.info(f"   URIs: {frame_table.data['uri'].nunique()}")
            logger.info(f"   Speech segments: {len(frame_table.speech_segments)}")
            logger.info(f"   Non-speech segments: {len(frame_table.nonspeech_segments)}")

            # Show sample
            logger.info("\n   Sample data:")
            print(frame_table.data.head(10))

            # Show condition distribution
            if "condition" in frame_table.data.columns:
                logger.info("\n   Condition distribution:")
                print(frame_table.data["condition"].value_counts())

        except Exception as e:
            logger.error(f"❌ Failed to load AVA-Speech: {e}")
            import traceback

            traceback.print_exc()
    else:
        logger.warning(f"⚠️  AVA-Speech annotations not found at {ava_annotations}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ LOADER TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
