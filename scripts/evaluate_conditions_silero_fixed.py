#!/usr/bin/env python3
"""
Evaluate Silero-VAD on psychoacoustic conditions using ORIGINAL segments.

Key difference from evaluate_conditions_silero.py:
- Uses original segments (without 2000ms padding) to avoid confusing Silero
- Applies manipulations on-the-fly from clip_id to find original segment
"""

import sys
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm
import argparse
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from qsm.vad import SileroVAD


def find_original_segment(clip_id: str, segments_root: Path = Path("data/segments")) -> Path:
    """
    Find original segment file from clip_id.

    Examples:
        voxconverse_abjxc_1.400_20ms -> data/segments/voxconverse/dev/voxconverse_abjxc_1.400_20ms.wav
        1-172649-B-40_20ms_010 -> data/segments/esc50/nonspeech/1-172649-B-40_20ms_010.wav
    """
    # Try voxconverse first
    vox_path = segments_root / "voxconverse" / "dev" / f"{clip_id}.wav"
    if vox_path.exists():
        return vox_path

    # Try ESC-50
    esc_path = segments_root / "esc50" / "nonspeech" / f"{clip_id}.wav"
    if esc_path.exists():
        return esc_path

    raise FileNotFoundError(f"Could not find original segment for clip_id: {clip_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Silero-VAD on psychoacoustic conditions using original segments"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/processed/conditions/conditions_manifest_subset.parquet",
        help="Path to conditions manifest (Parquet)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/subset",
        help="Directory to save results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Speech detection threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--window_samples",
        type=int,
        default=512,
        help="Window size in samples (512=32ms at 16kHz)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run model on (cuda/cpu)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SILERO-VAD EVALUATION: Psychoacoustic Conditions (FIXED)")
    print("=" * 80)

    # Load manifest
    print(f"\nLoading manifest: {args.manifest}")
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        sys.exit(1)

    df = pd.read_parquet(manifest_path)

    print(f"Total variants to evaluate: {len(df):,}")
    print(f"\nBreakdown by variant type:")
    print(df["variant_type"].value_counts().to_string())
    print(f"\nBreakdown by label:")
    print(df["label"].value_counts().to_string())

    # Load model
    print(f"\nLoading Silero-VAD model...")
    print(f"  Threshold: {args.threshold}")
    print(f"  Window size: {args.window_samples} samples ({args.window_samples/16:.1f}ms at 16kHz)")
    print(f"  Device: {args.device}")

    vad = SileroVAD(
        threshold=args.threshold,
        window_size_samples=args.window_samples,
        device=args.device,
    )

    print(f"\nModel loaded: {vad.name}")
    print("\nNOTE: Using ORIGINAL segments (without padding) to avoid false negatives")

    # Initialize results tracking
    results = []
    start_time = time.time()
    errors = []

    # Process all samples
    print(f"\nStarting evaluation...")
    print("=" * 80)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating", unit="sample"):
        clip_id = row["clip_id"]

        try:
            # Find original segment (without padding)
            audio_path = find_original_segment(clip_id)

            # Run prediction on original segment
            pred = vad.predict(audio_path)

            # Determine correctness (normalize NONSPEECH variants)
            ground_truth = row["label"]
            pred_normalized = pred.label.replace("-", "").replace("_", "")
            gt_normalized = ground_truth.replace("-", "").replace("_", "")
            is_correct = (pred_normalized == gt_normalized)

            results.append({
                # Identification
                "clip_id": clip_id,
                "audio_path": str(audio_path),
                "manipulated_path": row["audio_path"],  # For reference

                # Ground truth metadata
                "ground_truth": ground_truth,
                "duration_ms": row["duration_ms"],

                # Manipulation parameters (metadata only, not applied)
                "variant_type": row["variant_type"],
                "snr_db": row.get("snr_db", None),
                "band_filter": row.get("band_filter", None),

                # Model output
                "predicted": pred.label,
                "confidence": pred.confidence,
                "correct": is_correct,
                "latency_ms": pred.latency_ms,
            })

        except Exception as e:
            error_msg = f"ERROR processing {clip_id}: {e}"
            errors.append(error_msg)
            print(f"\n{error_msg}")

            results.append({
                "clip_id": clip_id,
                "audio_path": None,
                "manipulated_path": row["audio_path"],
                "ground_truth": row["label"],
                "duration_ms": row["duration_ms"],
                "variant_type": row["variant_type"],
                "snr_db": row.get("snr_db", None),
                "band_filter": row.get("band_filter", None),
                "predicted": None,
                "confidence": None,
                "correct": False,
                "latency_ms": None,
            })

    # Calculate overall statistics
    total_time = time.time() - start_time
    results_df = pd.DataFrame(results)

    # Filter out errors for accuracy calculation
    valid_results = results_df[results_df["predicted"].notna()]

    if len(valid_results) == 0:
        print("\nERROR: No valid predictions")
        return

    overall_accuracy = valid_results["correct"].mean() * 100
    total_samples = len(valid_results)
    total_correct = valid_results["correct"].sum()
    avg_latency = valid_results["latency_ms"].mean()
    avg_time_per_sample = total_time / len(df)

    # Print summary
    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 80}\n")

    if errors:
        print(f"WARNINGS: {len(errors)} errors occurred")
        print(f"Valid samples: {len(valid_results)}/{len(df)}\n")

    print(f"Overall Statistics:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Correct: {total_correct:,}")
    print(f"  Accuracy: {overall_accuracy:.2f}%")
    print(f"  Avg latency: {avg_latency:.1f}ms")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"  Time per sample: {avg_time_per_sample:.2f}s")
    print(f"  Samples per minute: {60 / avg_time_per_sample:.1f}")

    # Breakdown by variant type
    print(f"\nAccuracy by Variant Type:")
    print("-" * 50)
    for variant_type in valid_results["variant_type"].unique():
        subset = valid_results[valid_results["variant_type"] == variant_type]
        acc = subset["correct"].mean() * 100
        count = len(subset)
        print(f"  {variant_type:12s}: {acc:6.2f}% (n={count:,})")

    # Breakdown by duration
    print(f"\nAccuracy by Duration:")
    print("-" * 50)
    duration_summary = valid_results.groupby("duration_ms")["correct"].agg(
        accuracy=lambda x: x.mean() * 100,
        count="count"
    ).sort_index()
    for dur, row in duration_summary.iterrows():
        print(f"  {int(dur):4d} ms: {row['accuracy']:6.2f}% (n={int(row['count']):,})")

    # Breakdown by ground truth
    print(f"\nAccuracy by Ground Truth:")
    print("-" * 50)
    for gt in valid_results["ground_truth"].unique():
        subset = valid_results[valid_results["ground_truth"] == gt]
        acc = subset["correct"].mean() * 100
        count = len(subset)
        print(f"  {gt:12s}: {acc:6.2f}% (n={count:,})")

    # Save detailed results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_path = output_dir / "silero_conditions.parquet"
    results_df.to_parquet(detailed_path, index=False)
    print(f"\nDetailed results saved to: {detailed_path}")

    print(f"\n{'=' * 80}")
    print("Done!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
