#!/usr/bin/env python3
"""
Run VAD baselines (WebRTC, Silero) on segmented audio and evaluate.

Usage:
    python scripts/run_vad_baselines.py --model webrtc --segments-dir data/segments/ava_speech/train
    python scripts/run_vad_baselines.py --model silero --segments-dir data/segments/voxconverse/dev
    python scripts/run_vad_baselines.py --all  # Run all configurations
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.vad import WebRTCVAD, SileroVAD


def evaluate_vad_on_segments(
    vad_model,
    segments_metadata_path: Path,
    segments_dir: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Evaluate VAD model on a set of segments.

    Args:
        vad_model: VAD model instance (WebRTCVAD or SileroVAD)
        segments_metadata_path: Path to segments metadata parquet
        segments_dir: Directory containing segment audio files
        output_path: Optional path to save predictions

    Returns:
        DataFrame with predictions and metrics
    """
    # Load segment metadata
    df = pd.read_parquet(segments_metadata_path)

    print(f"\n{'='*80}")
    print(f"Evaluating {vad_model.name}")
    print(f"{'='*80}")
    print(f"Total segments: {len(df)}")
    print(f"SPEECH: {(df['label'] == 'SPEECH').sum()}")
    print(f"NONSPEECH: {(df['label'] == 'NONSPEECH').sum()}")
    print(f"Frame duration: {vad_model.frame_duration_ms}ms")
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
            prediction = vad_model.predict(audio_path)

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
                }
            )

            latencies.append(prediction.latency_ms)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    # Convert to DataFrame
    results_df = pd.DataFrame(predictions)

    # Calculate metrics
    print(f"\n{'='*80}")
    print(f"Results for {vad_model.name}")
    print(f"{'='*80}")

    # Overall metrics
    y_true = (results_df["true_label"] == "SPEECH").astype(int)
    y_pred = (results_df["pred_label"] == "SPEECH").astype(int)
    y_score = results_df["confidence"]

    # Adjust scores for NONSPEECH predictions (invert confidence)
    y_score_adjusted = y_score.copy()
    y_score_adjusted[y_pred == 0] = 1.0 - y_score_adjusted[y_pred == 0]

    overall_f1 = f1_score(y_true, y_pred)
    overall_auroc = roc_auc_score(y_true, y_score_adjusted)
    overall_auprc = average_precision_score(y_true, y_score_adjusted)

    print(f"\nOverall Metrics:")
    print(f"  F1 Score:  {overall_f1:.4f}")
    print(f"  AUROC:     {overall_auroc:.4f}")
    print(f"  AUPRC:     {overall_auprc:.4f}")
    print(f"  Avg Latency: {sum(latencies)/len(latencies):.2f}ms")
    print(f"  Max Latency: {max(latencies):.2f}ms")

    # Metrics by duration
    print(f"\nMetrics by Duration:")
    print(f"{'Duration (ms)':<15} {'F1':<10} {'AUROC':<10} {'AUPRC':<10} {'N':<10}")
    print(f"{'-'*60}")

    for duration in sorted(results_df["duration_ms"].unique()):
        mask = results_df["duration_ms"] == duration
        if mask.sum() == 0:
            continue

        y_true_dur = (results_df.loc[mask, "true_label"] == "SPEECH").astype(int)
        y_pred_dur = (results_df.loc[mask, "pred_label"] == "SPEECH").astype(int)
        y_score_dur = results_df.loc[mask, "confidence"].copy()

        # Adjust scores
        y_score_dur[y_pred_dur == 0] = 1.0 - y_score_dur[y_pred_dur == 0]

        f1_dur = f1_score(y_true_dur, y_pred_dur)
        auroc_dur = roc_auc_score(y_true_dur, y_score_dur) if len(y_true_dur.unique()) > 1 else 0.0
        auprc_dur = (
            average_precision_score(y_true_dur, y_score_dur)
            if len(y_true_dur.unique()) > 1
            else 0.0
        )

        print(
            f"{duration:<15} {f1_dur:<10.4f} {auroc_dur:<10.4f} {auprc_dur:<10.4f} {mask.sum():<10}"
        )

    # Metrics by condition (if available)
    if "condition" in results_df.columns and results_df["condition"].notna().any():
        print(f"\nMetrics by Condition:")
        print(f"{'Condition':<15} {'F1':<10} {'AUROC':<10} {'AUPRC':<10} {'N':<10}")
        print(f"{'-'*60}")

        for condition in sorted(results_df["condition"].unique()):
            if pd.isna(condition) or condition == "unknown":
                continue

            mask = results_df["condition"] == condition
            if mask.sum() == 0:
                continue

            y_true_cond = (results_df.loc[mask, "true_label"] == "SPEECH").astype(int)
            y_pred_cond = (results_df.loc[mask, "pred_label"] == "SPEECH").astype(int)
            y_score_cond = results_df.loc[mask, "confidence"].copy()

            # Adjust scores
            y_score_cond[y_pred_cond == 0] = 1.0 - y_score_cond[y_pred_cond == 0]

            f1_cond = f1_score(y_true_cond, y_pred_cond)
            auroc_cond = (
                roc_auc_score(y_true_cond, y_score_cond) if len(y_true_cond.unique()) > 1 else 0.0
            )
            auprc_cond = (
                average_precision_score(y_true_cond, y_score_cond)
                if len(y_true_cond.unique()) > 1
                else 0.0
            )

            print(
                f"{condition:<15} {f1_cond:<10.4f} {auroc_cond:<10.4f} {auprc_cond:<10.4f} {mask.sum():<10}"
            )

    print(f"{'='*80}\n")

    # Save results if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_path, index=False)
        print(f"Results saved to: {output_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run VAD baselines on segmented audio")

    parser.add_argument(
        "--model",
        type=str,
        choices=["webrtc", "silero"],
        help="VAD model to use",
    )

    parser.add_argument(
        "--segments-dir",
        type=Path,
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
        default=Path("results/vad_baselines"),
        help="Output directory for results",
    )

    # WebRTC-specific options
    parser.add_argument(
        "--webrtc-frame-ms",
        type=int,
        default=30,
        choices=[10, 20, 30],
        help="WebRTC frame duration in ms (default: 30)",
    )

    parser.add_argument(
        "--webrtc-aggressiveness",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="WebRTC aggressiveness level (default: 1)",
    )

    # Silero-specific options
    parser.add_argument(
        "--silero-threshold",
        type=float,
        default=0.5,
        help="Silero confidence threshold (default: 0.5)",
    )

    parser.add_argument(
        "--silero-window-samples",
        type=int,
        default=512,
        help="Silero window size in samples (512=32ms at 16kHz, default: 512)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all VAD configurations on all segment directories",
    )

    args = parser.parse_args()

    # Run all configurations
    if args.all:
        print("Running all VAD configurations on all segment directories...")

        segment_dirs = [
            Path("data/segments/ava_speech/train"),
            Path("data/segments/voxconverse/dev"),
            Path("data/segments/esc50/nonspeech"),
        ]

        # WebRTC configurations
        webrtc_configs = [
            {"frame_duration_ms": 10, "aggressiveness": 1},
            {"frame_duration_ms": 20, "aggressiveness": 1},
            {"frame_duration_ms": 30, "aggressiveness": 1},
            {"frame_duration_ms": 30, "aggressiveness": 0},
            {"frame_duration_ms": 30, "aggressiveness": 2},
            {"frame_duration_ms": 30, "aggressiveness": 3},
        ]

        # Silero configurations
        silero_configs = [
            {"threshold": 0.3, "window_size_samples": 512},
            {"threshold": 0.5, "window_size_samples": 512},
            {"threshold": 0.7, "window_size_samples": 512},
            {"threshold": 0.5, "window_size_samples": 1536},  # 96ms at 16kHz
        ]

        for segments_dir in segment_dirs:
            if not segments_dir.exists():
                print(f"Skipping {segments_dir} (not found)")
                continue

            metadata_path = segments_dir / "segments_metadata.parquet"
            if not metadata_path.exists():
                metadata_path = segments_dir / "segments.parquet"

            if not metadata_path.exists():
                print(f"Skipping {segments_dir} (no metadata found)")
                continue

            dataset_name = segments_dir.parent.name

            # WebRTC
            for config in webrtc_configs:
                vad = WebRTCVAD(**config)
                output_path = args.output_dir / "webrtc" / dataset_name / f"{vad.name}.parquet"
                evaluate_vad_on_segments(vad, metadata_path, segments_dir, output_path)

            # Silero
            for config in silero_configs:
                vad = SileroVAD(**config)
                output_path = args.output_dir / "silero" / dataset_name / f"{vad.name}.parquet"
                evaluate_vad_on_segments(vad, metadata_path, segments_dir, output_path)

        print("\n✅ All configurations complete!")
        return

    # Single model run
    if not args.model or not args.segments_dir:
        parser.error("--model and --segments-dir are required (or use --all)")

    # Find metadata file
    metadata_path = args.segments_dir / args.metadata_file
    if not metadata_path.exists():
        # Try alternative name
        metadata_path = args.segments_dir / "segments.parquet"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found in {args.segments_dir}")

    # Initialize VAD model
    if args.model == "webrtc":
        vad = WebRTCVAD(
            frame_duration_ms=args.webrtc_frame_ms,
            aggressiveness=args.webrtc_aggressiveness,
        )
    elif args.model == "silero":
        vad = SileroVAD(
            threshold=args.silero_threshold,
            window_size_samples=args.silero_window_samples,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Run evaluation
    dataset_name = args.segments_dir.parent.name
    output_path = args.output_dir / args.model / dataset_name / f"{vad.name}.parquet"

    evaluate_vad_on_segments(vad, metadata_path, args.segments_dir, output_path)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
