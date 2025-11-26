#!/usr/bin/env python3
"""
Optimize Classification Threshold

Analyzes predictions to find optimal threshold for:
1. Global optimal threshold
2. Per-condition optimal thresholds (duration, SNR, band, rir)

Usage:
    python scripts/optimize_threshold.py --predictions results/eval_full_conditions_base_thresh0.50/predictions.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score


def find_optimal_threshold(df: pd.DataFrame, thresholds: np.ndarray = None) -> dict:
    """Find optimal threshold that maximizes balanced accuracy."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.01)

    # Convert ground truth to binary
    y_true = (df["ground_truth"] == "SPEECH").astype(int)
    p_scores = df["p_first_token"].values

    best_thresh = 0.5
    best_ba = 0.0
    results = []

    for thresh in thresholds:
        y_pred = (p_scores > thresh).astype(int)
        ba = balanced_accuracy_score(y_true, y_pred)
        results.append({"threshold": thresh, "ba": ba})

        if ba > best_ba:
            best_ba = ba
            best_thresh = thresh

    # Calculate per-class accuracy at optimal threshold
    y_pred_opt = (p_scores > best_thresh).astype(int)
    speech_mask = y_true == 1
    nonspeech_mask = y_true == 0

    speech_acc = y_pred_opt[speech_mask].mean() if speech_mask.sum() > 0 else 0
    nonspeech_acc = (1 - y_pred_opt[nonspeech_mask]).mean() if nonspeech_mask.sum() > 0 else 0

    return {
        "optimal_threshold": float(best_thresh),
        "optimal_ba": float(best_ba),
        "speech_acc": float(speech_acc),
        "nonspeech_acc": float(nonspeech_acc),
        "n_samples": len(df),
        "n_speech": int(speech_mask.sum()),
        "n_nonspeech": int(nonspeech_mask.sum()),
        "threshold_curve": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Path to predictions.csv from evaluation")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: same dir as predictions)")
    args = parser.parse_args()

    print("="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)

    # Load predictions
    df = pd.read_csv(args.predictions)
    print(f"\nLoaded {len(df)} predictions from {args.predictions}")

    # Ensure required columns
    required = ["ground_truth", "p_first_token", "variant_type"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: Missing column '{col}'")
            return 1

    results = {}

    # Global optimization
    print("\n--- Global Threshold Optimization ---")
    global_result = find_optimal_threshold(df)
    results["global"] = global_result
    print(f"  Optimal threshold: {global_result['optimal_threshold']:.2f}")
    print(f"  Optimal BA: {global_result['optimal_ba']:.4f} ({global_result['optimal_ba']*100:.2f}%)")
    print(f"  Speech acc: {global_result['speech_acc']:.4f}")
    print(f"  Nonspeech acc: {global_result['nonspeech_acc']:.4f}")

    # Per variant type
    print("\n--- Per Variant Type Optimization ---")
    results["by_variant_type"] = {}
    for vtype in df["variant_type"].unique():
        vtype_df = df[df["variant_type"] == vtype]
        vtype_result = find_optimal_threshold(vtype_df)
        results["by_variant_type"][vtype] = vtype_result
        print(f"\n  {vtype.upper()} (n={len(vtype_df)}):")
        print(f"    Optimal threshold: {vtype_result['optimal_threshold']:.2f}")
        print(f"    Optimal BA: {vtype_result['optimal_ba']:.4f} ({vtype_result['optimal_ba']*100:.2f}%)")

    # Per condition within each variant type
    print("\n--- Per Condition Optimization ---")
    results["by_condition"] = {}

    # Duration
    if "duration_ms" in df.columns:
        results["by_condition"]["duration"] = {}
        dur_df = df[df["variant_type"] == "duration"]
        for dur in sorted(dur_df["duration_ms"].dropna().unique()):
            cond_df = dur_df[dur_df["duration_ms"] == dur]
            if len(cond_df) >= 10:  # Minimum samples
                cond_result = find_optimal_threshold(cond_df)
                results["by_condition"]["duration"][f"{int(dur)}ms"] = cond_result

    # SNR
    if "snr_db" in df.columns:
        results["by_condition"]["snr"] = {}
        snr_df = df[df["variant_type"] == "snr"]
        for snr in sorted(snr_df["snr_db"].dropna().unique()):
            cond_df = snr_df[snr_df["snr_db"] == snr]
            if len(cond_df) >= 10:
                cond_result = find_optimal_threshold(cond_df)
                results["by_condition"]["snr"][f"{int(snr)}dB"] = cond_result

    # Band
    if "band_filter" in df.columns:
        results["by_condition"]["band"] = {}
        band_df = df[df["variant_type"] == "band"]
        for band in band_df["band_filter"].dropna().unique():
            cond_df = band_df[band_df["band_filter"] == band]
            if len(cond_df) >= 10:
                cond_result = find_optimal_threshold(cond_df)
                results["by_condition"]["band"][band] = cond_result

    # RIR
    if "T60_bin" in df.columns:
        results["by_condition"]["rir"] = {}
        rir_df = df[df["variant_type"] == "rir"]
        for t60 in rir_df["T60_bin"].dropna().unique():
            cond_df = rir_df[rir_df["T60_bin"] == t60]
            if len(cond_df) >= 10:
                cond_result = find_optimal_threshold(cond_df)
                results["by_condition"]["rir"][t60] = cond_result

    # Print condition-specific results
    for vtype, conditions in results["by_condition"].items():
        print(f"\n  {vtype.upper()}:")
        for cond, cond_result in sorted(conditions.items()):
            print(f"    {cond:15s}: thresh={cond_result['optimal_threshold']:.2f}, BA={cond_result['optimal_ba']*100:.1f}%")

    # Save results
    output_path = args.output or args.predictions.parent / "threshold_optimization.json"
    with open(output_path, "w") as f:
        # Remove threshold_curve for cleaner output
        clean_results = json.loads(json.dumps(results))
        for key in clean_results:
            if isinstance(clean_results[key], dict) and "threshold_curve" in clean_results[key]:
                del clean_results[key]["threshold_curve"]
            if key == "by_variant_type":
                for vt in clean_results[key]:
                    if "threshold_curve" in clean_results[key][vt]:
                        del clean_results[key][vt]["threshold_curve"]
            if key == "by_condition":
                for vt in clean_results[key]:
                    for cond in clean_results[key][vt]:
                        if "threshold_curve" in clean_results[key][vt][cond]:
                            del clean_results[key][vt][cond]["threshold_curve"]

        json.dump(clean_results, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Summary recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print(f"\n1. Global optimal threshold: {results['global']['optimal_threshold']:.2f}")
    print(f"   (improves BA from current to {results['global']['optimal_ba']*100:.2f}%)")

    # Check if per-condition thresholds help significantly
    print("\n2. Per-condition thresholds worth considering:")
    for vtype, vresult in results["by_variant_type"].items():
        improvement = vresult["optimal_ba"] - results["global"]["optimal_ba"]
        if abs(vresult["optimal_threshold"] - results["global"]["optimal_threshold"]) > 0.1:
            print(f"   - {vtype}: thresh={vresult['optimal_threshold']:.2f} (vs global {results['global']['optimal_threshold']:.2f})")

    return 0


if __name__ == "__main__":
    exit(main())
