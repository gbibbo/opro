#!/usr/bin/env python3
"""
Re-analyze existing Sprint 6 results with corrected metrics logic.

This script loads saved predictions and recomputes metrics correctly WITHOUT
re-running the expensive model inference.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)


def aggregate_by_clip(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate variant-level predictions to clip-level (majority vote)."""
    clip_results = []

    for clip_id in predictions_df["clip_id"].unique():
        clip_preds = predictions_df[predictions_df["clip_id"] == clip_id]

        y_true = clip_preds["y_true"].iloc[0]
        pred_counts = clip_preds["y_pred"].value_counts()
        y_pred = pred_counts.idxmax()
        confidence = pred_counts.max() / len(clip_preds)

        clip_results.append({
            "clip_id": clip_id,
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidence,
            "n_variants": len(clip_preds),
            "correct": (y_true == y_pred),
        })

    return pd.DataFrame(clip_results)


def compute_robust_metrics(y_true, y_pred, prefix=""):
    """Compute Balanced Accuracy and Macro-F1."""
    metrics = {}

    metrics[f"{prefix}balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics[f"{prefix}macro_f1"] = f1_score(y_true, y_pred, average="macro")
    metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    for label in ["SPEECH", "NONSPEECH"]:
        if label in report:
            metrics[f"{prefix}{label.lower()}_f1"] = report[label]["f1-score"]
            metrics[f"{prefix}{label.lower()}_precision"] = report[label]["precision"]
            metrics[f"{prefix}{label.lower()}_recall"] = report[label]["recall"]

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Re-analyze Sprint 6 results with corrected logic")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/sprint6_robust/dev_predictions.parquet"),
        help="Predictions parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/sprint6_robust_corrected"),
        help="Output directory for corrected metrics",
    )

    args = parser.parse_args()

    print("="*60)
    print("RE-ANALYZING SPRINT 6 RESULTS (CORRECTED LOGIC)")
    print("="*60)

    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    pred_df = pd.read_parquet(args.predictions)

    print(f"Loaded {len(pred_df)} variant predictions from {pred_df['clip_id'].nunique()} clips")

    # 1. Variant-level metrics
    print("\n" + "="*60)
    print("VARIANT-LEVEL METRICS")
    print("="*60)

    variant_metrics = compute_robust_metrics(
        pred_df["y_true"].values,
        pred_df["y_pred"].values,
        prefix="variant_",
    )

    print(f"Accuracy: {variant_metrics['variant_accuracy']:.3f}")
    print(f"Balanced Accuracy: {variant_metrics['variant_balanced_accuracy']:.3f}")
    print(f"Macro-F1: {variant_metrics['variant_macro_f1']:.3f}")

    # 2. Clip-level metrics (CORRECT: majority vote)
    print("\n" + "="*60)
    print("CLIP-LEVEL METRICS (MAJORITY VOTE - ANTI-INFLATION)")
    print("="*60)

    clip_agg = aggregate_by_clip(pred_df)

    clip_metrics = compute_robust_metrics(
        clip_agg["y_true"].values,
        clip_agg["y_pred"].values,
        prefix="clip_",
    )

    print(f"Accuracy: {clip_metrics['clip_accuracy']:.3f}")
    print(f"Balanced Accuracy: {clip_metrics['clip_balanced_accuracy']:.3f}")
    print(f"Macro-F1: {clip_metrics['clip_macro_f1']:.3f}")

    # Sanity check
    if clip_metrics['clip_accuracy'] >= variant_metrics['variant_accuracy']:
        print("[OK] Clip-level >= Variant-level (expected)")
    else:
        print("[ERROR] Clip-level < Variant-level (impossible!)")

    # 3. Condition-specific metrics (VARIANT-LEVEL)
    print("\n" + "="*60)
    print("CONDITION-SPECIFIC METRICS (VARIANT-LEVEL)")
    print("="*60)

    condition_metrics = {}

    # Duration
    print("\nDURATION:")
    duration_df = pred_df[pred_df["variant_type"] == "duration"]
    for dur in sorted(duration_df["duration_ms"].unique()):
        subset = duration_df[duration_df["duration_ms"] == dur]
        metrics = compute_robust_metrics(subset["y_true"].values, subset["y_pred"].values)
        condition_metrics[f"duration_{int(dur)}ms"] = metrics
        print(f"  {int(dur):4d}ms: Acc={metrics['accuracy']:.3f}  Bal.Acc={metrics['balanced_accuracy']:.3f}  F1={metrics['macro_f1']:.3f}  (n={len(subset)})")

    # SNR
    print("\nSNR:")
    snr_df = pred_df[pred_df["variant_type"] == "snr"]
    for snr in sorted(snr_df["snr_db"].dropna().unique()):
        subset = snr_df[snr_df["snr_db"] == snr]
        metrics = compute_robust_metrics(subset["y_true"].values, subset["y_pred"].values)
        condition_metrics[f"snr_{int(snr):+d}dB"] = metrics
        print(f"  {int(snr):+3d}dB: Acc={metrics['accuracy']:.3f}  Bal.Acc={metrics['balanced_accuracy']:.3f}  F1={metrics['macro_f1']:.3f}  (n={len(subset)})")

    # Band
    print("\nBAND:")
    band_df = pred_df[pred_df["variant_type"] == "band"]
    for band in sorted(band_df["band_filter"].unique()):
        subset = band_df[band_df["band_filter"] == band]
        metrics = compute_robust_metrics(subset["y_true"].values, subset["y_pred"].values)
        condition_metrics[f"band_{band}"] = metrics
        print(f"  {band:10s}: Acc={metrics['accuracy']:.3f}  Bal.Acc={metrics['balanced_accuracy']:.3f}  F1={metrics['macro_f1']:.3f}  (n={len(subset)})")

    # RIR
    print("\nRIR:")
    rir_df = pred_df[pred_df["variant_type"] == "rir"]
    for t60 in sorted(rir_df["T60_bin"].unique()):
        subset = rir_df[rir_df["T60_bin"] == t60]
        metrics = compute_robust_metrics(subset["y_true"].values, subset["y_pred"].values)
        condition_metrics[f"rir_{t60}"] = metrics
        print(f"  {t60:15s}: Acc={metrics['accuracy']:.3f}  Bal.Acc={metrics['balanced_accuracy']:.3f}  F1={metrics['macro_f1']:.3f}  (n={len(subset)})")

    # 4. Macro across conditions (OBJECTIVE METRIC)
    cond_bal_accs = [m["balanced_accuracy"] for m in condition_metrics.values()]
    cond_macro_f1s = [m["macro_f1"] for m in condition_metrics.values()]

    macro_metrics = {
        "macro_balanced_accuracy": np.mean(cond_bal_accs),
        "macro_macro_f1": np.mean(cond_macro_f1s),
    }

    print("\n" + "="*60)
    print("MACRO ACROSS CONDITIONS (OBJECTIVE METRIC)")
    print("="*60)
    print(f"Macro Balanced Accuracy: {macro_metrics['macro_balanced_accuracy']:.3f}")
    print(f"Macro Macro-F1: {macro_metrics['macro_macro_f1']:.3f}")

    # Save corrected metrics
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "n_variants": len(pred_df),
        "n_clips": len(clip_agg),
        **variant_metrics,
        **clip_metrics,
        **macro_metrics,
        "by_condition": condition_metrics,
    }

    metrics_path = args.output_dir / "corrected_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved corrected metrics to: {metrics_path}")

    # Save clip aggregation
    clip_path = args.output_dir / "clips.parquet"
    clip_agg.to_parquet(clip_path, index=False)
    print(f"Saved clip aggregation to: {clip_path}")

    print("\n" + "="*60)
    print("RE-ANALYSIS COMPLETE")
    print("="*60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
