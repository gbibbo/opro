#!/usr/bin/env python3
"""
Sprint 9: Single-prompt evaluator for OPRO optimization.

Evaluates a given prompt on the dev set and computes:
1. Clip-level balanced accuracy (BA_clip) - PRIMARY
2. Condition-averaged balanced accuracy (BA_conditions)
3. Full metrics by condition

Reuses Sprint 6 evaluation infrastructure.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.models import Qwen2AudioClassifier


def aggregate_by_clip(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate variant-level predictions to clip-level using majority vote.

    Args:
        predictions_df: DataFrame with columns [clip_id, y_true, y_pred, ...]

    Returns:
        DataFrame with one row per clip
    """
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

        clip_results.append(
            {
                "clip_id": clip_id,
                "y_true": y_true,
                "y_pred": y_pred,
                "confidence": confidence,
                "n_variants": len(clip_preds),
                "correct": (y_true == y_pred),
            }
        )

    return pd.DataFrame(clip_results)


def compute_condition_ba(predictions_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute balanced accuracy per psychoacoustic condition.

    Args:
        predictions_df: Variant-level predictions with condition metadata

    Returns:
        Dictionary of {condition_name: balanced_accuracy}
    """
    condition_bas = {}

    # Duration conditions
    duration_df = predictions_df[predictions_df["variant_type"] == "duration"]
    if len(duration_df) > 0:
        for dur in sorted(duration_df["duration_ms"].unique()):
            dur_subset = duration_df[duration_df["duration_ms"] == dur]
            ba = balanced_accuracy_score(dur_subset["y_true"], dur_subset["y_pred"])
            condition_bas[f"duration_{int(dur)}ms"] = ba

    # SNR conditions
    snr_df = predictions_df[predictions_df["variant_type"] == "snr"]
    if len(snr_df) > 0:
        for snr in sorted(snr_df["snr_db"].dropna().unique()):
            snr_subset = snr_df[snr_df["snr_db"] == snr]
            ba = balanced_accuracy_score(snr_subset["y_true"], snr_subset["y_pred"])
            condition_bas[f"snr_{int(snr):+d}dB"] = ba

    # Band filter conditions
    band_df = predictions_df[predictions_df["variant_type"] == "band"]
    if len(band_df) > 0:
        for band in band_df["band_filter"].unique():
            band_subset = band_df[band_df["band_filter"] == band]
            ba = balanced_accuracy_score(band_subset["y_true"], band_subset["y_pred"])
            condition_bas[f"band_{band}"] = ba

    # RIR conditions
    rir_df = predictions_df[predictions_df["variant_type"] == "rir"]
    if len(rir_df) > 0:
        for t60 in rir_df["T60_bin"].unique():
            t60_subset = rir_df[rir_df["T60_bin"] == t60]
            ba = balanced_accuracy_score(t60_subset["y_true"], t60_subset["y_pred"])
            condition_bas[f"rir_{t60}"] = ba

    return condition_bas


def evaluate_prompt(
    prompt: str,
    manifest_path: Path,
    model: Qwen2AudioClassifier = None,
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    device: str = "cuda",
    split: str = "dev",
    seed: int = 42,
    save_predictions: bool = False,
    output_dir: Path = None,
) -> Tuple[float, float, Dict]:
    """
    Evaluate a prompt on the dev set.

    Args:
        prompt: Prompt to evaluate
        manifest_path: Path to manifest with split column
        model: Optional pre-loaded model (if None, loads new instance)
        model_name: Model name (if model is None)
        device: Device to use
        split: Split to evaluate on ("dev" or "test")
        seed: Random seed
        save_predictions: Whether to save predictions to disk
        output_dir: Output directory (if save_predictions=True)

    Returns:
        Tuple of (ba_clip, ba_conditions_avg, full_metrics)
    """
    # Set seed
    np.random.seed(seed)

    # Load manifest
    manifest_df = pd.read_parquet(manifest_path)

    # Filter to split
    split_df = manifest_df[manifest_df["split"] == split].copy()

    # Add ground_truth if not present
    if "ground_truth" not in split_df.columns:
        split_df["ground_truth"] = (
            split_df["label"].str.replace("-", "").str.replace("_", "").str.upper()
        )

    # Load model if not provided
    model_loaded = model is not None
    if not model_loaded:
        print(f"Loading model: {model_name}...")
        model = Qwen2AudioClassifier(
            model_name=model_name,
            device=device,
            load_in_4bit=True,
        )

    # Set prompt
    # Parse prompt to extract system and user components
    # Baseline format: "<|audio_bos|><|AUDIO|><|audio_eos|>User text"
    # We need to extract just the user text part
    if "<|audio_eos|>" in prompt:
        # Extract user text after audio markers
        user_text = prompt.split("<|audio_eos|>")[1].strip()
    else:
        # Assume entire prompt is user text
        user_text = prompt.strip()

    model.set_prompt(user_prompt=user_text)

    # Run predictions
    predictions = []

    print(f"Evaluating on {split} set ({len(split_df)} variants)...")
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Evaluating"):
        audio_path = Path(str(row["audio_path"]).replace(chr(92), "/"))

        if not audio_path.exists():
            print(f"Warning: Audio not found: {audio_path}")
            continue

        # Predict
        result = model.predict(audio_path)

        predictions.append(
            {
                "clip_id": row["clip_id"],
                "variant_type": row["variant_type"],
                "duration_ms": row.get("duration_ms"),
                "snr_db": row.get("snr_db"),
                "band_filter": row.get("band_filter"),
                "T60_bin": row.get("T60_bin"),
                "y_true": row["ground_truth"],
                "y_pred": result.label,
                "confidence": result.confidence,
                "raw_output": result.raw_output,
            }
        )

    predictions_df = pd.DataFrame(predictions)

    # Compute clip-level metrics (PRIMARY)
    clip_agg = aggregate_by_clip(predictions_df)
    ba_clip = balanced_accuracy_score(clip_agg["y_true"], clip_agg["y_pred"])

    # Compute condition-level metrics
    condition_bas = compute_condition_ba(predictions_df)
    ba_conditions_avg = np.mean(list(condition_bas.values()))

    # Build full metrics dictionary
    metrics = {
        "prompt": prompt,
        "split": split,
        "n_variants": len(predictions_df),
        "n_clips": len(clip_agg),
        "ba_clip": ba_clip,
        "ba_conditions_avg": ba_conditions_avg,
        "ba_by_condition": condition_bas,
        "clip_accuracy": (clip_agg["y_true"] == clip_agg["y_pred"]).mean(),
        "variant_accuracy": (predictions_df["y_true"] == predictions_df["y_pred"]).mean(),
    }

    # Save results if requested
    if save_predictions and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_path = output_dir / f"{split}_predictions.parquet"
        predictions_df.to_parquet(pred_path, index=False)

        clip_path = output_dir / f"{split}_clips.parquet"
        clip_agg.to_parquet(clip_path, index=False)

        metrics_path = output_dir / f"{split}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved results to: {output_dir}")

    # Print summary
    print(f"\nResults ({split} set):")
    print(f"  BA_clip (PRIMARY): {ba_clip:.3f}")
    print(f"  BA_conditions (avg): {ba_conditions_avg:.3f}")
    print(f"  Clip accuracy: {metrics['clip_accuracy']:.3f}")
    print(f"  Variant accuracy: {metrics['variant_accuracy']:.3f}")

    # Cleanup model if we loaded it
    if not model_loaded:
        del model
        import torch

        torch.cuda.empty_cache()

    return ba_clip, ba_conditions_avg, metrics


def main():
    """Main entry point for single-prompt evaluation."""
    parser = argparse.ArgumentParser(description="Sprint 9: Evaluate single prompt")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt to evaluate (full prompt or just user text)",
    )
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
        default=None,
        help="Output directory (if None, results not saved)",
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

    # Evaluate
    ba_clip, ba_cond, metrics = evaluate_prompt(
        prompt=args.prompt,
        manifest_path=args.manifest,
        model_name=args.model_name,
        device=args.device,
        split=args.split,
        seed=args.seed,
        save_predictions=(args.output_dir is not None),
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"BA_clip: {ba_clip:.4f}")
    print(f"BA_conditions: {ba_cond:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
