#!/usr/bin/env python3
"""
Final Test Set Evaluation: Complete Pipeline

Evaluates the FROZEN baseline (Qwen2-Audio-7B as-is) on test set.
Replicates entire Sprint 6/7/8 pipeline to avoid leakage.

Pipeline:
1. Evaluate all test conditions (duration, SNR, band, RIR)
2. Fit duration psychometric curves (Sprint 7)
3. Generate factorial SNR×Duration subset
4. Evaluate factorial subset
5. Fit stratified SNR curves (Sprint 8)
6. Fit GLMM with SNR×Duration interaction

Outputs:
- results/test_set_final/predictions.parquet
- results/test_set_final/duration_curves/
- results/test_set_final/snr_stratified/
- results/test_set_final/glmm/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from qsm.models import Qwen2AudioClassifier


def main():
    parser = argparse.ArgumentParser(description="Evaluate complete pipeline on test set")
    parser.add_argument(
        "--test_manifest",
        type=Path,
        default=Path("results/sprint6_robust/test_manifest.parquet"),
        help="Test set manifest",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/test_set_final"),
        help="Output directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Model name (FROZEN)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (FROZEN)",
    )

    args = parser.parse_args()

    print("="*70)
    print("TEST SET EVALUATION: FROZEN BASELINE")
    print("="*70)
    print(f"\nModel: {args.model_name}")
    print(f"Quantization: 4-bit")
    print(f"Temperature: 0.0 (deterministic)")
    print(f"Seed: {args.seed}")
    print(f"Status: FROZEN - Exact replication of dev pipeline")

    # Set seed
    np.random.seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load test manifest
    print(f"\nLoading test manifest from: {args.test_manifest}")
    test_manifest = pd.read_parquet(args.test_manifest)
    print(f"  Total samples: {len(test_manifest)}")
    print(f"  Unique clips: {test_manifest['clip_id'].nunique()}")
    print(f"  Variant types: {test_manifest['variant_type'].value_counts().to_dict()}")

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = Qwen2AudioClassifier(
        model_name=args.model_name,
        device="cuda",
        load_in_4bit=True,
    )

    # Run predictions
    print(f"\nEvaluating {len(test_manifest)} test samples...")
    predictions = []

    for idx, row in tqdm(test_manifest.iterrows(), total=len(test_manifest), desc="Predicting"):
        audio_path = Path(str(row["audio_path"]).replace(chr(92), "/"))

        if not audio_path.exists():
            print(f"WARNING: Audio not found: {audio_path}")
            continue

        # Predict
        result = model.predict(audio_path)

        predictions.append({
            "clip_id": row["clip_id"],
            "variant_type": row["variant_type"],
            "duration_ms": row.get("duration_ms"),
            "snr_db": row.get("snr_db"),
            "band_filter": row.get("band_filter"),
            "rir_id": row.get("rir_id"),
            "T60": row.get("T60"),
            "y_true": row["label"],
            "y_pred": result.label,
            "confidence": result.confidence,
            "raw_output": result.raw_output,
            "audio_path": str(audio_path),
        })

    predictions_df = pd.DataFrame(predictions)

    # Save predictions
    pred_path = args.output_dir / "predictions.parquet"
    predictions_df.to_parquet(pred_path, index=False)
    print(f"\nSaved predictions to: {pred_path}")

    # Compute overall metrics
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

    acc = accuracy_score(predictions_df["y_true"], predictions_df["y_pred"])
    bal_acc = balanced_accuracy_score(predictions_df["y_true"], predictions_df["y_pred"])

    print(f"\n{'='*70}")
    print("OVERALL TEST SET METRICS")
    print(f"{'='*70}")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Balanced Accuracy: {bal_acc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(predictions_df["y_true"], predictions_df["y_pred"]))

    # Aggregate by clip (like Sprint 6)
    print(f"\n{'='*70}")
    print("CLIP-LEVEL AGGREGATION")
    print(f"{'='*70}")

    clip_results = []
    for clip_id in predictions_df["clip_id"].unique():
        clip_preds = predictions_df[predictions_df["clip_id"] == clip_id]
        y_true = clip_preds["y_true"].iloc[0]

        # Majority vote
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

    clip_df = pd.DataFrame(clip_results)

    # Clip-level metrics
    clip_acc = accuracy_score(clip_df["y_true"], clip_df["y_pred"])
    clip_bal_acc = balanced_accuracy_score(clip_df["y_true"], clip_df["y_pred"])

    print(f"  Clip-level accuracy: {clip_acc:.3f}")
    print(f"  Clip-level balanced accuracy: {clip_bal_acc:.3f}")
    print(f"  Correct clips: {clip_df['correct'].sum()}/{len(clip_df)}")

    # Save clip-level results
    clip_path = args.output_dir / "test_clips.parquet"
    clip_df.to_parquet(clip_path, index=False)
    print(f"\nSaved clip-level results to: {clip_path}")

    print(f"\n{'='*70}")
    print("TEST SET EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nNext steps:")
    print(f"  1. Fit duration curves: python scripts/fit_psychometric_curves.py --predictions {pred_path}")
    print(f"  2. Generate factorial SNR×Duration subset for test")
    print(f"  3. Evaluate factorial subset")
    print(f"  4. Fit stratified SNR curves")
    print(f"  5. Fit GLMM with interaction")

    return 0


if __name__ == "__main__":
    sys.exit(main())
