#!/usr/bin/env python3
"""
Run Qwen2-Audio inference on segmented audio.

Usage:
    python scripts/run_qwen_inference.py --segments-dir data/segments/ava_speech/train
    python scripts/run_qwen_inference.py --segments-dir data/segments/esc50/nonspeech --limit 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.models import Qwen2AudioClassifier


def evaluate_model_on_segments(
    model,
    segments_metadata_path: Path,
    segments_dir: Path,
    output_path: Path | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate Qwen2-Audio model on a set of segments.

    Args:
        model: Qwen2AudioClassifier instance
        segments_metadata_path: Path to segments metadata parquet
        segments_dir: Directory containing segment audio files
        output_path: Optional path to save predictions
        limit: Optional limit on number of segments to process

    Returns:
        DataFrame with predictions and metrics
    """
    # Load segment metadata
    df = pd.read_parquet(segments_metadata_path)

    if limit:
        print(f"Limiting to first {limit} segments")
        df = df.head(limit)

    print(f"\n{'='*80}")
    print(f"Evaluating {model.name}")
    print(f"{'='*80}")
    print(f"Total segments: {len(df)}")
    print(f"SPEECH: {(df['label'] == 'SPEECH').sum()}")
    print(f"NONSPEECH: {(df['label'] == 'NONSPEECH').sum()}")
    print(f"{'='*80}\n")

    # Run predictions
    predictions = []
    latencies = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing segments"):
        audio_path = segments_dir / Path(row["audio_path"]).name

        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        try:
            prediction = model.predict(audio_path)

            predictions.append(
                {
                    "segment_id": idx,
                    "true_label": row["label"],
                    "pred_label": prediction.label,
                    "confidence": prediction.confidence,
                    "latency_ms": prediction.latency_ms,
                    "duration_ms": row["duration_ms"],
                    "dataset": row.get("dataset", "unknown"),
                    "condition": row.get("condition", "unknown"),
                    "raw_output": prediction.raw_output,
                }
            )

            latencies.append(prediction.latency_ms)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(predictions)

    if len(results_df) == 0:
        print("No predictions generated!")
        return results_df

    # Calculate metrics
    print(f"\n{'='*80}")
    print(f"Results for {model.name}")
    print(f"{'='*80}")

    # Overall metrics
    y_true = results_df["true_label"]
    y_pred = results_df["pred_label"]

    # Filter out UNKNOWN predictions for metrics
    valid_mask = y_pred != "UNKNOWN"
    if valid_mask.sum() < len(y_pred):
        print(f"\nWarning: {(~valid_mask).sum()} predictions were UNKNOWN and excluded from metrics")

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_true_valid) > 0:
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        precision = precision_score(
            y_true_valid, y_pred_valid, pos_label="SPEECH", zero_division=0
        )
        recall = recall_score(y_true_valid, y_pred_valid, pos_label="SPEECH", zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, pos_label="SPEECH", zero_division=0)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Avg Latency: {sum(latencies)/len(latencies):.2f}ms")
        print(f"  Max Latency: {max(latencies):.2f}ms")

        # Metrics by duration
        print(f"\nMetrics by Duration:")
        print(f"{'Duration (ms)':<15} {'Accuracy':<10} {'F1':<10} {'N':<10}")
        print(f"{'-'*60}")

        for duration in sorted(results_df["duration_ms"].unique()):
            mask = (results_df["duration_ms"] == duration) & valid_mask
            if mask.sum() == 0:
                continue

            y_true_dur = results_df.loc[mask, "true_label"]
            y_pred_dur = results_df.loc[mask, "pred_label"]

            acc_dur = accuracy_score(y_true_dur, y_pred_dur)
            f1_dur = f1_score(y_true_dur, y_pred_dur, pos_label="SPEECH", zero_division=0)

            print(f"{duration:<15} {acc_dur:<10.4f} {f1_dur:<10.4f} {mask.sum():<10}")

        # Metrics by condition (if available)
        if "condition" in results_df.columns and results_df["condition"].notna().any():
            print(f"\nMetrics by Condition (top 10):")
            print(f"{'Condition':<20} {'Accuracy':<10} {'F1':<10} {'N':<10}")
            print(f"{'-'*60}")

            # Get top 10 most common conditions
            top_conditions = results_df["condition"].value_counts().head(10).index

            for condition in top_conditions:
                if pd.isna(condition) or condition == "unknown":
                    continue

                mask = (results_df["condition"] == condition) & valid_mask
                if mask.sum() == 0:
                    continue

                y_true_cond = results_df.loc[mask, "true_label"]
                y_pred_cond = results_df.loc[mask, "pred_label"]

                acc_cond = accuracy_score(y_true_cond, y_pred_cond)
                f1_cond = f1_score(
                    y_true_cond, y_pred_cond, pos_label="SPEECH", zero_division=0
                )

                print(f"{condition:<20} {acc_cond:<10.4f} {f1_cond:<10.4f} {mask.sum():<10}")

    print(f"{'='*80}\n")

    # Save results if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_path, index=False)
        print(f"Results saved to: {output_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen2-Audio inference on segmented audio"
    )

    parser.add_argument(
        "--segments-dir",
        type=Path,
        required=True,
        help="Directory containing segments and metadata",
    )

    parser.add_argument(
        "--metadata-file",
        type=str,
        default="segments_metadata.parquet",
        help="Name of metadata file (default: segments_metadata.parquet)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/qwen_inference"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2-Audio-7B-Instruct)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="Model dtype (default: auto)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of segments to process (for testing)",
    )

    args = parser.parse_args()

    # Check if segments directory exists
    if not args.segments_dir.exists():
        print(f"Error: Segments directory not found: {args.segments_dir}")
        return

    # Try both possible metadata filenames
    metadata_path = args.segments_dir / args.metadata_file
    if not metadata_path.exists():
        # Try alternative name
        metadata_path = args.segments_dir / "segments.parquet"
        if not metadata_path.exists():
            print(f"Error: Metadata file not found in {args.segments_dir}")
            return

    # Initialize Qwen2-Audio
    print("Initializing Qwen2-Audio model...")
    model = Qwen2AudioClassifier(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=args.dtype,
    )

    # Run evaluation
    dataset_name = args.segments_dir.parent.name
    output_path = args.output_dir / dataset_name / f"{model.name}.parquet"

    evaluate_model_on_segments(
        model, metadata_path, args.segments_dir, output_path, args.limit
    )

    print("\nInference complete!")


if __name__ == "__main__":
    main()
