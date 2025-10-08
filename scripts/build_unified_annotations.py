#!/usr/bin/env python3
"""
Build unified annotation table from all datasets.

Converts different annotation formats to a single FrameTable:
- RTTM (VoxConverse, DIHARD) → segment-level
- CSV (AVA-Speech) → frame-level
- Word alignment (AMI) → word-level

Output: Unified parquet file with consistent schema.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from qsm.data import FrameTable, load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_unified_table(
    datasets: list[str],
    output_path: Path,
    include_nonspeech: bool = False,
) -> FrameTable:
    """
    Build unified annotation table from multiple datasets.

    Args:
        datasets: List of dataset names to include
        output_path: Where to save the unified parquet file
        include_nonspeech: Whether to include nonspeech segments

    Returns:
        Unified FrameTable
    """
    logger.info("=" * 80)
    logger.info("BUILDING UNIFIED ANNOTATION TABLE")
    logger.info("=" * 80)

    all_frames = []

    for dataset_name in datasets:
        logger.info(f"\n📊 Processing {dataset_name}...")

        try:
            # Load dataset using our loaders
            config_path = (
                Path(__file__).parent.parent / "configs" / "datasets" / f"{dataset_name}.yaml"
            )

            if not config_path.exists():
                logger.warning(f"Config not found for {dataset_name}, skipping...")
                continue

            # Load dataset (this will use appropriate loader based on dataset type)
            ft = load_dataset(dataset_name, split="train", config_path=config_path)

            logger.info(f"  Loaded {len(ft.data)} annotations")
            logger.info(f"  URIs: {ft.data['uri'].nunique()}")
            logger.info(f"  Labels: {ft.data['label'].value_counts().to_dict()}")

            # Add to collection
            all_frames.append(ft.data)

            # Show sample
            logger.info("  Sample row:")
            logger.info(f"    {ft.data.iloc[0].to_dict()}")

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            continue

    if not all_frames:
        logger.error("No datasets loaded successfully!")
        return None

    # Combine all datasets
    logger.info("\n" + "=" * 80)
    logger.info("COMBINING ALL DATASETS")
    logger.info("=" * 80)

    combined_df = pd.concat(all_frames, ignore_index=True)
    unified_ft = FrameTable(data=combined_df)

    # Summary statistics
    logger.info("\n📈 UNIFIED TABLE STATISTICS")
    logger.info(f"  Total annotations: {len(unified_ft.data)}")
    logger.info(f"  Unique URIs: {unified_ft.data['uri'].nunique()}")
    logger.info(f"  Datasets: {unified_ft.data['dataset'].value_counts().to_dict()}")
    logger.info(f"  Labels: {unified_ft.data['label'].value_counts().to_dict()}")
    logger.info(f"  Splits: {unified_ft.data['split'].value_counts().to_dict()}")

    # Duration statistics
    unified_ft.data["duration_s"] = unified_ft.data["end_s"] - unified_ft.data["start_s"]
    total_duration = unified_ft.data["duration_s"].sum()
    speech_duration = unified_ft.data[unified_ft.data["label"] == "SPEECH"]["duration_s"].sum()
    logger.info("\n⏱️  DURATION STATISTICS")
    logger.info(f"  Total duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    logger.info(f"  Speech duration: {speech_duration:.2f}s ({speech_duration/60:.2f} min)")
    logger.info(
        f"  Speech percentage: {(speech_duration/total_duration*100) if total_duration > 0 else 0:.1f}%"
    )

    # Per-dataset breakdown
    logger.info("\n📊 PER-DATASET BREAKDOWN")
    for dataset in unified_ft.data["dataset"].unique():
        dataset_data = unified_ft.data[unified_ft.data["dataset"] == dataset]
        dataset_duration = dataset_data["duration_s"].sum()
        logger.info(f"  {dataset}:")
        logger.info(f"    Annotations: {len(dataset_data)}")
        logger.info(f"    URIs: {dataset_data['uri'].nunique()}")
        logger.info(f"    Duration: {dataset_duration:.2f}s")

    # Save to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified_ft.save(output_path)
    logger.info(f"\n💾 Saved unified table to: {output_path}")

    return unified_ft


def inspect_unified_table(parquet_path: Path):
    """
    Inspect a previously built unified table.

    Args:
        parquet_path: Path to the unified parquet file
    """
    logger.info("=" * 80)
    logger.info("INSPECTING UNIFIED TABLE")
    logger.info("=" * 80)

    ft = FrameTable.load(parquet_path)

    logger.info("\n📋 SCHEMA")
    logger.info(f"  Columns: {list(ft.data.columns)}")
    logger.info(f"  Shape: {ft.data.shape}")
    logger.info(f"  Memory usage: {ft.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    logger.info("\n📊 STATISTICS")
    logger.info(ft.data.describe())

    logger.info("\n🔍 SAMPLE ROWS")
    logger.info(ft.data.head(10).to_string())

    # Check for missing values
    logger.info("\n❓ MISSING VALUES")
    missing = ft.data.isnull().sum()
    if missing.sum() > 0:
        logger.info(missing[missing > 0])
    else:
        logger.info("  No missing values!")

    return ft


def main():
    parser = argparse.ArgumentParser(
        description="Build unified annotation table from multiple datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["voxconverse", "dihard", "ava_speech"],
        help="Datasets to include",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/unified_annotations.parquet"),
        help="Output path for unified table",
    )
    parser.add_argument(
        "--include-nonspeech", action="store_true", help="Include nonspeech segments"
    )
    parser.add_argument(
        "--inspect",
        type=Path,
        help="Inspect an existing unified table (skips building)",
    )

    args = parser.parse_args()

    if args.inspect:
        # Just inspect existing table
        inspect_unified_table(args.inspect)
    else:
        # Build new unified table
        unified_ft = build_unified_table(
            datasets=args.datasets,
            output_path=args.output,
            include_nonspeech=args.include_nonspeech,
        )

        if unified_ft is not None:
            logger.info("\n✅ SUCCESS! Unified table built successfully.")
            logger.info(
                f"\nTo inspect the table later, run:\n  python {__file__} --inspect {args.output}"
            )


if __name__ == "__main__":
    main()
