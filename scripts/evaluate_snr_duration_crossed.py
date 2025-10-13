#!/usr/bin/env python3
"""
Sprint 8: Evaluate factorial SNR×Duration dataset.

Evaluates 640 samples (20 clips × 4 durations × 8 SNR levels) to generate
stratified SNR curves by duration level.

Expected output:
- predictions.parquet with [clip_id, duration_ms, snr_db, y_true, y_pred]
- Accuracy by (duration, SNR) condition
- Ready for stratified psychometric curve fitting
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.models import Qwen2AudioClassifier


def evaluate_factorial_dataset(
    model: Qwen2AudioClassifier,
    metadata_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Evaluate factorial SNR×Duration dataset.

    Args:
        model: Qwen2AudioClassifier instance
        metadata_path: Path to metadata.csv
        output_dir: Output directory

    Returns:
        predictions_df with columns [clip_id, duration_ms, snr_db, y_true, y_pred, ...]
    """
    print(f"\n{'='*60}")
    print("SPRINT 8: EVALUATE FACTORIAL SNR×DURATION DATASET")
    print(f"{'='*60}\n")

    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    print(f"Loaded {len(metadata_df)} samples")
    print(f"  Clips: {metadata_df['clip_id'].nunique()}")
    print(f"  Durations: {sorted(metadata_df['duration_ms'].unique())}")
    print(f"  SNR levels: {sorted(metadata_df['snr_db'].unique())}")
    print(f"  Labels: {metadata_df['ground_truth'].value_counts().to_dict()}")

    # Normalize ground truth labels
    metadata_df["ground_truth"] = metadata_df["ground_truth"].str.replace("-", "").str.replace("_", "").str.upper()

    # Run predictions
    predictions = []

    print("\nRunning predictions...")
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Evaluating"):
        audio_path = Path(str(row["audio_path"]).replace(chr(92), "/"))

        if not audio_path.exists():
            print(f"Warning: Audio not found: {audio_path}")
            continue

        # Predict
        start_time = time.time()
        result = model.predict(audio_path)
        latency_ms = (time.time() - start_time) * 1000

        predictions.append({
            "clip_id": row["clip_id"],
            "variant_name": row["variant_name"],
            "duration_ms": row["duration_ms"],
            "snr_db": row["snr_db"],
            "y_true": row["ground_truth"],
            "y_pred": result.label,
            "confidence": result.confidence,
            "raw_output": result.raw_output,
            "latency_ms": latency_ms,
            "audio_path": str(audio_path),
            "rms_signal": row.get("rms_signal"),
            "rms_noise": row.get("rms_noise"),
            "measured_snr_db": row.get("measured_snr_db"),
        })

    predictions_df = pd.DataFrame(predictions)

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.parquet"
    predictions_df.to_parquet(pred_path, index=False)
    print(f"\nSaved predictions to: {pred_path}")

    # Compute overall metrics
    overall_acc = accuracy_score(predictions_df["y_true"], predictions_df["y_pred"])
    overall_bal_acc = balanced_accuracy_score(predictions_df["y_true"], predictions_df["y_pred"])

    print(f"\nOverall metrics:")
    print(f"  Accuracy: {overall_acc:.3f}")
    print(f"  Balanced Accuracy: {overall_bal_acc:.3f}")

    # Compute metrics by duration
    print(f"\nMetrics by duration:")
    duration_metrics = {}
    for dur in sorted(predictions_df["duration_ms"].unique()):
        dur_df = predictions_df[predictions_df["duration_ms"] == dur]
        acc = accuracy_score(dur_df["y_true"], dur_df["y_pred"])
        bal_acc = balanced_accuracy_score(dur_df["y_true"], dur_df["y_pred"])
        duration_metrics[f"duration_{int(dur)}ms"] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "n_samples": len(dur_df),
        }
        print(f"  {int(dur):4d} ms: Acc={acc:.3f}, Bal.Acc={bal_acc:.3f} (n={len(dur_df)})")

    # Compute metrics by SNR
    print(f"\nMetrics by SNR:")
    snr_metrics = {}
    for snr in sorted(predictions_df["snr_db"].unique()):
        snr_df = predictions_df[predictions_df["snr_db"] == snr]
        acc = accuracy_score(snr_df["y_true"], snr_df["y_pred"])
        bal_acc = balanced_accuracy_score(snr_df["y_true"], snr_df["y_pred"])
        snr_metrics[f"snr_{int(snr):+d}dB"] = {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "n_samples": len(snr_df),
        }
        print(f"  {int(snr):+3d} dB: Acc={acc:.3f}, Bal.Acc={bal_acc:.3f} (n={len(snr_df)})")

    # Compute metrics by (duration, SNR) condition
    print(f"\nMetrics by (duration, SNR) condition:")
    condition_metrics = {}
    for dur in sorted(predictions_df["duration_ms"].unique()):
        for snr in sorted(predictions_df["snr_db"].unique()):
            cond_df = predictions_df[
                (predictions_df["duration_ms"] == dur) & (predictions_df["snr_db"] == snr)
            ]
            if len(cond_df) == 0:
                continue

            acc = accuracy_score(cond_df["y_true"], cond_df["y_pred"])
            bal_acc = balanced_accuracy_score(cond_df["y_true"], cond_df["y_pred"])
            condition_metrics[f"dur{int(dur)}ms_snr{int(snr):+d}dB"] = {
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "n_samples": len(cond_df),
            }
            print(f"  {int(dur):4d}ms × {int(snr):+3d}dB: Acc={acc:.3f}, Bal.Acc={bal_acc:.3f} (n={len(cond_df)})")

    # Save metrics
    metrics = {
        "overall": {
            "accuracy": overall_acc,
            "balanced_accuracy": overall_bal_acc,
            "n_samples": len(predictions_df),
        },
        "by_duration": duration_metrics,
        "by_snr": snr_metrics,
        "by_condition": condition_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}\n")

    return predictions_df


def main():
    parser = argparse.ArgumentParser(description="Sprint 8: Evaluate factorial SNR×Duration dataset")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/snr_duration_crossed/metadata.csv"),
        help="Path to metadata.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/sprint8_factorial"),
        help="Output directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Load model
    print(f"Loading model: {args.model_name}")
    print(f"Device: {args.device}")
    model = Qwen2AudioClassifier(
        model_name=args.model_name,
        device=args.device,
        load_in_4bit=True,
    )

    # Evaluate
    predictions_df = evaluate_factorial_dataset(
        model=model,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
