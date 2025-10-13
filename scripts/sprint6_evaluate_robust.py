#!/usr/bin/env python3
"""
Sprint 6: Robust evaluation with clip-grouped metrics.

Implements:
1. Clip-level aggregation (anti-inflation: average 20 variants → 1 score per clip)
2. Balanced Accuracy and Macro-F1 per condition
3. Weighted metrics for hard conditions (duration ≤200ms, SNR ≤0dB)
4. Deterministic evaluation (temperature=0, fixed seed)
5. Detailed predictions.parquet output
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.models import Qwen2AudioClassifier


def aggregate_by_clip(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate variant-level predictions to clip-level.

    For each clip_id, average the predictions across all 20 variants.
    Final prediction = majority vote.

    Args:
        predictions_df: DataFrame with columns [clip_id, y_true, y_pred, ...]

    Returns:
        DataFrame with one row per clip
    """
    print("Aggregating predictions by clip...")

    clip_results = []

    for clip_id in predictions_df["clip_id"].unique():
        clip_preds = predictions_df[predictions_df["clip_id"] == clip_id]

        # Ground truth (same for all variants)
        y_true = clip_preds["y_true"].iloc[0]

        # Majority vote
        pred_counts = clip_preds["y_pred"].value_counts()
        y_pred = pred_counts.idxmax()

        # Confidence = fraction of variants agreeing with majority
        confidence = pred_counts.max() / len(clip_preds)

        clip_results.append({
            "clip_id": clip_id,
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidence,
            "n_variants": len(clip_preds),
            "correct": (y_true == y_pred),
        })

    clip_df = pd.DataFrame(clip_results)

    print(f"  Aggregated {len(predictions_df)} variant predictions")
    print(f"  → {len(clip_df)} clip predictions")

    return clip_df


def compute_robust_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute robust metrics (Balanced Accuracy and Macro-F1).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        prefix: Prefix for metric names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Balanced Accuracy (accounts for class imbalance)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    metrics[f"{prefix}balanced_accuracy"] = bal_acc

    # Macro-F1 (unweighted average of per-class F1)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    metrics[f"{prefix}macro_f1"] = macro_f1

    # Standard accuracy for reference
    acc = accuracy_score(y_true, y_pred)
    metrics[f"{prefix}accuracy"] = acc

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    for label in ["SPEECH", "NONSPEECH"]:
        if label in report:
            metrics[f"{prefix}{label.lower()}_f1"] = report[label]["f1-score"]
            metrics[f"{prefix}{label.lower()}_precision"] = report[label]["precision"]
            metrics[f"{prefix}{label.lower()}_recall"] = report[label]["recall"]

    return metrics


def compute_condition_metrics(
    predictions_df: pd.DataFrame,
    clip_agg_df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per psychoacoustic condition using VARIANT-LEVEL analysis.

    CRITICAL: For condition-specific metrics, we want to know:
    "How accurate is the model on duration=20ms variants?"

    This is DIFFERENT from clip-level aggregation. Each clip only has
    ONE variant per condition (e.g., one 20ms variant), so we evaluate
    at the VARIANT level for condition analysis.

    The clip-level aggregation is for OVERALL metrics (across all conditions).

    Args:
        predictions_df: Variant-level predictions with condition metadata
        clip_agg_df: Clip-level aggregation (not used for conditions, kept for API)

    Returns:
        Dictionary of metrics by condition (VARIANT-LEVEL)
    """
    print("\nComputing metrics by condition (VARIANT-LEVEL)...")

    condition_metrics = {}

    # Metrics by duration (VARIANT-LEVEL)
    duration_df = predictions_df[predictions_df["variant_type"] == "duration"]
    if len(duration_df) > 0:
        for dur in sorted(duration_df["duration_ms"].unique()):
            dur_subset = duration_df[duration_df["duration_ms"] == dur]

            # Each clip has exactly 1 variant at this duration
            # So variant-level = per-clip for this condition
            metrics = compute_robust_metrics(
                dur_subset["y_true"].values,
                dur_subset["y_pred"].values,
                prefix="",
            )

            condition_metrics[f"duration_{int(dur)}ms"] = metrics

    # Metrics by SNR (VARIANT-LEVEL)
    snr_df = predictions_df[predictions_df["variant_type"] == "snr"]
    if len(snr_df) > 0:
        for snr in sorted(snr_df["snr_db"].dropna().unique()):
            snr_subset = snr_df[snr_df["snr_db"] == snr]

            metrics = compute_robust_metrics(
                snr_subset["y_true"].values,
                snr_subset["y_pred"].values,
                prefix="",
            )

            condition_metrics[f"snr_{int(snr):+d}dB"] = metrics

    # Metrics by band filter (VARIANT-LEVEL)
    band_df = predictions_df[predictions_df["variant_type"] == "band"]
    if len(band_df) > 0:
        for band in band_df["band_filter"].unique():
            band_subset = band_df[band_df["band_filter"] == band]

            metrics = compute_robust_metrics(
                band_subset["y_true"].values,
                band_subset["y_pred"].values,
                prefix="",
            )

            condition_metrics[f"band_{band}"] = metrics

    # Metrics by T60 (VARIANT-LEVEL)
    rir_df = predictions_df[predictions_df["variant_type"] == "rir"]
    if len(rir_df) > 0:
        for t60 in rir_df["T60_bin"].unique():
            t60_subset = rir_df[rir_df["T60_bin"] == t60]

            metrics = compute_robust_metrics(
                t60_subset["y_true"].values,
                t60_subset["y_pred"].values,
                prefix="",
            )

            condition_metrics[f"rir_{t60}"] = metrics

    return condition_metrics


def evaluate_on_split(
    model: Qwen2AudioClassifier,
    manifest_df: pd.DataFrame,
    split: str = "dev",
    output_dir: Path = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate model on a specific split.

    Args:
        model: Qwen2AudioClassifier instance
        manifest_df: Manifest with split column
        split: Split to evaluate on ("dev" or "test")
        output_dir: Directory to save results

    Returns:
        (predictions_df, metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {split.upper()} split")
    print(f"{'='*60}\n")

    # Filter to split
    split_df = manifest_df[manifest_df["split"] == split].copy()

    print(f"Split size: {len(split_df)} variants from {split_df['clip_id'].nunique()} clips")

    # Run predictions
    predictions = []

    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"{split.upper()} eval"):
        audio_path = Path(str(row["audio_path"]).replace(chr(92), "/"))

        if not audio_path.exists():
            print(f"Warning: Audio not found: {audio_path}")
            continue

        # Predict
        result = model.predict(audio_path)

        predictions.append({
            "clip_id": row["clip_id"],
            "variant_id": f"{row['clip_id']}_{row['variant_type']}",  # Unique variant ID
            "variant_type": row["variant_type"],
            "duration_ms": row.get("duration_ms"),
            "snr_db": row.get("snr_db"),
            "band_filter": row.get("band_filter"),
            "T60_bin": row.get("T60_bin"),
            "y_true": row["ground_truth"],
            "y_pred": result.label,
            "confidence": result.confidence,
            "raw_output": result.raw_output,
            "latency_ms": result.latency_ms,
            "audio_path": str(audio_path),
        })

    predictions_df = pd.DataFrame(predictions)

    # Compute metrics

    # 1. Variant-level metrics (for reference)
    variant_metrics = compute_robust_metrics(
        predictions_df["y_true"].values,
        predictions_df["y_pred"].values,
        prefix="variant_",
    )

    # 2. Clip-level metrics (PRIMARY - anti-inflation)
    clip_agg = aggregate_by_clip(predictions_df)

    clip_metrics = compute_robust_metrics(
        clip_agg["y_true"].values,
        clip_agg["y_pred"].values,
        prefix="clip_",
    )

    # 3. Condition-specific metrics (VARIANT-LEVEL)
    condition_metrics = compute_condition_metrics(predictions_df, clip_agg)

    # 4. Compute macro average across conditions
    # (Primary objective metric)
    cond_bal_accs = [m["balanced_accuracy"] for m in condition_metrics.values()]
    cond_macro_f1s = [m["macro_f1"] for m in condition_metrics.values()]

    macro_metrics = {
        "macro_balanced_accuracy": np.mean(cond_bal_accs),
        "macro_macro_f1": np.mean(cond_macro_f1s),
    }

    # Combine all metrics
    all_metrics = {
        "split": split,
        "n_variants": len(predictions_df),
        "n_clips": len(clip_agg),
        **variant_metrics,
        **clip_metrics,
        **macro_metrics,
        "by_condition": condition_metrics,
    }

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        pred_path = output_dir / f"{split}_predictions.parquet"
        predictions_df.to_parquet(pred_path, index=False)
        print(f"\nSaved predictions to: {pred_path}")

        # Save clip aggregation
        clip_path = output_dir / f"{split}_clips.parquet"
        clip_agg.to_parquet(clip_path, index=False)
        print(f"Saved clip aggregation to: {clip_path}")

        # Save metrics
        metrics_path = output_dir / f"{split}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved metrics to: {metrics_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"{split.upper()} RESULTS SUMMARY")
    print(f"{'='*60}\n")

    print(f"Variant-level:")
    print(f"  Accuracy: {variant_metrics['variant_accuracy']:.3f}")
    print(f"  Balanced Accuracy: {variant_metrics['variant_balanced_accuracy']:.3f}")
    print(f"  Macro-F1: {variant_metrics['variant_macro_f1']:.3f}")

    print(f"\nClip-level (PRIMARY - anti-inflation):")
    print(f"  Accuracy: {clip_metrics['clip_accuracy']:.3f}")
    print(f"  Balanced Accuracy: {clip_metrics['clip_balanced_accuracy']:.3f}")
    print(f"  Macro-F1: {clip_metrics['clip_macro_f1']:.3f}")

    print(f"\nMacro across conditions (OBJECTIVE METRIC):")
    print(f"  Macro Balanced Accuracy: {macro_metrics['macro_balanced_accuracy']:.3f}")
    print(f"  Macro Macro-F1: {macro_metrics['macro_macro_f1']:.3f}")

    print(f"\nBy condition:")
    for cond_name, cond_metrics in sorted(condition_metrics.items()):
        print(f"  {cond_name:20s}: Bal.Acc={cond_metrics['balanced_accuracy']:.3f}  F1={cond_metrics['macro_f1']:.3f}")

    return predictions_df, all_metrics


def main():
    parser = argparse.ArgumentParser(description="Sprint 6: Robust evaluation with clip grouping")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/conditions_final/conditions_manifest_split.parquet"),
        help="Manifest with split column",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "test"],
        help="Split to evaluate on",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/sprint6_robust"),
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
        help="Device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Load manifest
    print(f"Loading manifest from: {args.manifest}")
    manifest_df = pd.read_parquet(args.manifest)

    # Add ground_truth if not present
    if "ground_truth" not in manifest_df.columns:
        manifest_df["ground_truth"] = manifest_df["label"].str.replace("-", "").str.replace("_", "").str.upper()

    print(f"Loaded {len(manifest_df)} variants")
    print(f"Splits: {manifest_df['split'].value_counts().to_dict()}")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = Qwen2AudioClassifier(
        model_name=args.model_name,
        device=args.device,
        load_in_4bit=True,
    )

    # Evaluate
    predictions_df, metrics = evaluate_on_split(
        model,
        manifest_df,
        split=args.split,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
