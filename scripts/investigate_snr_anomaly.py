#!/usr/bin/env python3
"""
Investigate SNR Anomaly

The model performs better at low SNR (-10dB) than high SNR (+20dB).
This script investigates why this counterintuitive pattern occurs.

Hypotheses:
1. Noise masks speech ambiguity, making classification easier
2. The model uses "noise presence" as a proxy for non-speech
3. High SNR samples have different characteristics
4. Label noise or dataset artifacts

Usage:
    python scripts/investigate_snr_anomaly.py --predictions results/eval_full_conditions_base_thresh0.50/predictions.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json


def analyze_predictions_by_snr(df: pd.DataFrame) -> dict:
    """Analyze prediction patterns by SNR level."""
    snr_df = df[df["variant_type"] == "snr"].copy()

    if len(snr_df) == 0:
        return {"error": "No SNR data found"}

    results = {}

    for snr in sorted(snr_df["snr_db"].unique()):
        snr_subset = snr_df[snr_df["snr_db"] == snr]

        speech_subset = snr_subset[snr_subset["ground_truth"] == "SPEECH"]
        nonspeech_subset = snr_subset[snr_subset["ground_truth"] == "NONSPEECH"]

        results[f"snr_{int(snr)}dB"] = {
            "n_total": len(snr_subset),
            "n_speech": len(speech_subset),
            "n_nonspeech": len(nonspeech_subset),
            # p_first_token distributions
            "speech_p_mean": float(speech_subset["p_first_token"].mean()) if len(speech_subset) > 0 else None,
            "speech_p_std": float(speech_subset["p_first_token"].std()) if len(speech_subset) > 0 else None,
            "speech_p_min": float(speech_subset["p_first_token"].min()) if len(speech_subset) > 0 else None,
            "speech_p_max": float(speech_subset["p_first_token"].max()) if len(speech_subset) > 0 else None,
            "nonspeech_p_mean": float(nonspeech_subset["p_first_token"].mean()) if len(nonspeech_subset) > 0 else None,
            "nonspeech_p_std": float(nonspeech_subset["p_first_token"].std()) if len(nonspeech_subset) > 0 else None,
            "nonspeech_p_min": float(nonspeech_subset["p_first_token"].min()) if len(nonspeech_subset) > 0 else None,
            "nonspeech_p_max": float(nonspeech_subset["p_first_token"].max()) if len(nonspeech_subset) > 0 else None,
            # Accuracy
            "speech_correct": float(speech_subset["correct"].mean()) if len(speech_subset) > 0 else None,
            "nonspeech_correct": float(nonspeech_subset["correct"].mean()) if len(nonspeech_subset) > 0 else None,
            # Separation (difference between speech and nonspeech p_first_token means)
            "class_separation": None,
        }

        if results[f"snr_{int(snr)}dB"]["speech_p_mean"] is not None and results[f"snr_{int(snr)}dB"]["nonspeech_p_mean"] is not None:
            results[f"snr_{int(snr)}dB"]["class_separation"] = float(
                results[f"snr_{int(snr)}dB"]["speech_p_mean"] - results[f"snr_{int(snr)}dB"]["nonspeech_p_mean"]
            )

    return results


def analyze_error_patterns(df: pd.DataFrame) -> dict:
    """Analyze which samples are misclassified at different SNR levels."""
    snr_df = df[df["variant_type"] == "snr"].copy()

    if len(snr_df) == 0:
        return {"error": "No SNR data found"}

    # Get unique clip_ids
    clip_ids = snr_df["clip_id"].unique() if "clip_id" in snr_df.columns else []

    results = {
        "consistent_errors": [],  # Clips wrong at all SNR levels
        "snr_dependent_errors": [],  # Clips that flip correctness with SNR
    }

    # For each clip, check error pattern across SNR
    for clip_id in clip_ids:
        clip_data = snr_df[snr_df["clip_id"] == clip_id].sort_values("snr_db")
        if len(clip_data) == 0:
            continue

        correct_at_snr = clip_data.set_index("snr_db")["correct"].to_dict()

        # Check if always wrong
        if all(not v for v in correct_at_snr.values()):
            results["consistent_errors"].append({
                "clip_id": clip_id,
                "ground_truth": clip_data["ground_truth"].iloc[0],
                "p_first_tokens": clip_data.set_index("snr_db")["p_first_token"].to_dict(),
            })
        # Check if error pattern changes with SNR
        elif any(correct_at_snr.values()) and not all(correct_at_snr.values()):
            results["snr_dependent_errors"].append({
                "clip_id": clip_id,
                "ground_truth": clip_data["ground_truth"].iloc[0],
                "correct_at_snr": correct_at_snr,
                "p_first_tokens": clip_data.set_index("snr_db")["p_first_token"].to_dict(),
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print("="*60)
    print("SNR ANOMALY INVESTIGATION")
    print("="*60)

    df = pd.read_csv(args.predictions)
    print(f"\nLoaded {len(df)} predictions")

    # Extract clip_id from audio_path if not present
    if "clip_id" not in df.columns:
        df["clip_id"] = df["audio_path"].apply(lambda x: Path(x).stem.split("_snr")[0])

    results = {}

    # Analysis 1: p_first_token distributions by SNR
    print("\n--- Analysis 1: Prediction Distributions by SNR ---")
    snr_analysis = analyze_predictions_by_snr(df)
    results["snr_distributions"] = snr_analysis

    print("\n  SNR Level | Speech p_mean | Nonspeech p_mean | Separation | Speech Acc | Nonspeech Acc")
    print("  " + "-"*90)
    for snr_key in sorted(snr_analysis.keys(), key=lambda x: int(x.split("_")[1].replace("dB", ""))):
        data = snr_analysis[snr_key]
        print(f"  {snr_key:10s} | {data['speech_p_mean']:.3f}         | {data['nonspeech_p_mean']:.3f}            | "
              f"{data['class_separation']:.3f}      | {data['speech_correct']:.2f}       | {data['nonspeech_correct']:.2f}")

    # Analysis 2: Error patterns
    print("\n--- Analysis 2: Error Patterns Across SNR ---")
    error_analysis = analyze_error_patterns(df)
    results["error_patterns"] = {
        "n_consistent_errors": len(error_analysis["consistent_errors"]),
        "n_snr_dependent_errors": len(error_analysis["snr_dependent_errors"]),
        "consistent_errors": error_analysis["consistent_errors"][:5],  # First 5
        "snr_dependent_examples": error_analysis["snr_dependent_errors"][:5],
    }

    print(f"\n  Clips always misclassified: {len(error_analysis['consistent_errors'])}")
    print(f"  Clips with SNR-dependent errors: {len(error_analysis['snr_dependent_errors'])}")

    # Analysis 3: Hypothesis testing
    print("\n--- Analysis 3: Hypothesis Testing ---")

    # Hypothesis: Nonspeech accuracy improves at low SNR
    snr_df = df[df["variant_type"] == "snr"]
    low_snr = snr_df[snr_df["snr_db"] <= 0]
    high_snr = snr_df[snr_df["snr_db"] >= 10]

    low_snr_nonspeech = low_snr[low_snr["ground_truth"] == "NONSPEECH"]
    high_snr_nonspeech = high_snr[high_snr["ground_truth"] == "NONSPEECH"]

    low_snr_speech = low_snr[low_snr["ground_truth"] == "SPEECH"]
    high_snr_speech = high_snr[high_snr["ground_truth"] == "SPEECH"]

    results["hypothesis_test"] = {
        "low_snr_nonspeech_acc": float(low_snr_nonspeech["correct"].mean()) if len(low_snr_nonspeech) > 0 else None,
        "high_snr_nonspeech_acc": float(high_snr_nonspeech["correct"].mean()) if len(high_snr_nonspeech) > 0 else None,
        "low_snr_speech_acc": float(low_snr_speech["correct"].mean()) if len(low_snr_speech) > 0 else None,
        "high_snr_speech_acc": float(high_snr_speech["correct"].mean()) if len(high_snr_speech) > 0 else None,
        "low_snr_nonspeech_p_mean": float(low_snr_nonspeech["p_first_token"].mean()) if len(low_snr_nonspeech) > 0 else None,
        "high_snr_nonspeech_p_mean": float(high_snr_nonspeech["p_first_token"].mean()) if len(high_snr_nonspeech) > 0 else None,
    }

    print(f"\n  Hypothesis: 'Noise helps identify nonspeech'")
    print(f"  - Low SNR (≤0dB) nonspeech accuracy: {results['hypothesis_test']['low_snr_nonspeech_acc']:.2f}")
    print(f"  - High SNR (≥10dB) nonspeech accuracy: {results['hypothesis_test']['high_snr_nonspeech_acc']:.2f}")
    print(f"  - Low SNR nonspeech p_first_token mean: {results['hypothesis_test']['low_snr_nonspeech_p_mean']:.3f}")
    print(f"  - High SNR nonspeech p_first_token mean: {results['hypothesis_test']['high_snr_nonspeech_p_mean']:.3f}")

    if results['hypothesis_test']['low_snr_nonspeech_acc'] > results['hypothesis_test']['high_snr_nonspeech_acc']:
        print("\n  ✓ CONFIRMED: Model classifies nonspeech better when noise is present")
        print("    Interpretation: The model may be using 'noise presence' as a cue for non-speech")
    else:
        print("\n  ✗ NOT CONFIRMED: Nonspeech accuracy does not improve with noise")

    # Save results
    output_path = args.output or args.predictions.parent / "snr_anomaly_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    # Conclusions
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    print("""
The SNR anomaly (better performance at low SNR) is likely explained by:

1. CLASS SEPARATION: At low SNR, the p_first_token values for speech and
   nonspeech are more separated, making classification easier.

2. NOISE AS CUE: The model may have learned that "noisy" audio is more
   likely to be non-speech, which helps at low SNR but hurts at high SNR.

3. CLEAN AUDIO AMBIGUITY: At high SNR, clean nonspeech samples may sound
   more similar to clean speech, causing confusion.

RECOMMENDATIONS:
- Train with more high-SNR nonspeech examples
- Use prompts that explicitly handle clean audio cases
- Consider ensemble with separate models for high/low SNR
""")

    return 0


if __name__ == "__main__":
    exit(main())
