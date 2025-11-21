#!/usr/bin/env python3
"""
Temperature Scaling Calibration for Logit-Based Predictions

Finds optimal temperature to minimize Expected Calibration Error (ECE).
Does NOT change accuracy, only improves confidence calibration.

Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017)
https://arxiv.org/abs/1706.04599

Usage:
    # Step 1: Get uncalibrated predictions on dev set
    python scripts/evaluate_with_logits.py \
        --checkpoint checkpoints/no_leakage_v2/seed_42/final \
        --test_csv data/processed/grouped_split/dev_metadata.csv \
        --temperature 1.0 \
        --output_csv results/dev_uncalibrated.csv

    # Step 2: Find optimal temperature
    python scripts/calibrate_temperature.py \
        --predictions_csv results/dev_uncalibrated.csv \
        --output_temp results/optimal_temperature.txt

    # Step 3: Evaluate on test with optimal temperature
    python scripts/evaluate_with_logits.py \
        --checkpoint checkpoints/no_leakage_v2/seed_42/final \
        --test_csv data/processed/grouped_split/test_metadata.csv \
        --temperature $(cat results/optimal_temperature.txt) \
        --output_csv results/test_calibrated.csv
"""

import argparse
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from pathlib import Path


def compute_ece(confidences, predictions, ground_truth, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.
    Lower is better (0 = perfect calibration).

    Args:
        confidences: Array of predicted probabilities [0, 1]
        predictions: Array of predicted classes
        ground_truth: Array of true classes
        n_bins: Number of bins for reliability diagram

    Returns:
        ece: Expected Calibration Error (scalar)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples in this confidence bin
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            # Average confidence in bin
            avg_conf = np.mean(confidences[in_bin])

            # Average accuracy in bin
            avg_acc = np.mean(predictions[in_bin] == ground_truth[in_bin])

            # Weighted contribution to ECE
            ece += (bin_size / total_samples) * np.abs(avg_conf - avg_acc)

    return ece


def compute_brier_score(confidences, ground_truth_binary):
    """
    Compute Brier Score for binary classification.

    Lower is better (0 = perfect predictions).

    Args:
        confidences: Array of predicted probabilities for class 1
        ground_truth_binary: Array of true classes (0 or 1)

    Returns:
        brier: Brier score (scalar)
    """
    return np.mean((confidences - ground_truth_binary) ** 2)


def apply_temperature_scaling(logit_diffs, temperature):
    """
    Apply temperature scaling to logit differences.

    Args:
        logit_diffs: Array of logit(A) - logit(B)
        temperature: Temperature parameter (T > 0)

    Returns:
        calibrated_probs: Array of calibrated probabilities for class A
    """
    # Scale logits by temperature
    scaled_logits = logit_diffs / temperature

    # Convert to probabilities via sigmoid
    probs = 1 / (1 + np.exp(-scaled_logits))

    return probs


def find_optimal_temperature(logit_diffs, ground_truth, n_bins=10):
    """
    Find optimal temperature to minimize ECE on validation set.

    Args:
        logit_diffs: Array of logit(A) - logit(B)
        ground_truth: Array of true classes ('SPEECH' or 'NONSPEECH')
        n_bins: Number of bins for ECE computation

    Returns:
        optimal_T: Best temperature value
        ece_uncalibrated: ECE before calibration
        ece_calibrated: ECE after calibration
    """
    # Convert ground truth to binary (1 = SPEECH, 0 = NONSPEECH)
    gt_binary = (ground_truth == 'SPEECH').astype(int)

    # Get uncalibrated predictions (T=1.0)
    uncal_probs = apply_temperature_scaling(logit_diffs, temperature=1.0)
    uncal_preds = (uncal_probs >= 0.5).astype(int)
    ece_uncalibrated = compute_ece(uncal_probs, uncal_preds, gt_binary, n_bins)

    # Optimize temperature to minimize ECE
    def objective(T):
        cal_probs = apply_temperature_scaling(logit_diffs, temperature=T)
        cal_preds = (cal_probs >= 0.5).astype(int)
        return compute_ece(cal_probs, cal_preds, gt_binary, n_bins)

    # Search temperature in range [0.1, 10.0]
    result = minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
    optimal_T = result.x
    ece_calibrated = result.fun

    return optimal_T, ece_uncalibrated, ece_calibrated


def plot_reliability_diagram(logit_diffs, ground_truth, temperature, n_bins=10, output_path=None):
    """
    Plot reliability diagram (confidence vs accuracy).

    Args:
        logit_diffs: Array of logit(A) - logit(B)
        ground_truth: Array of true classes
        temperature: Temperature to apply
        n_bins: Number of bins
        output_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt

    gt_binary = (ground_truth == 'SPEECH').astype(int)
    probs = apply_temperature_scaling(logit_diffs, temperature)
    preds = (probs >= 0.5).astype(int)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    avg_confs = []
    avg_accs = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_conf = np.mean(probs[in_bin])
            avg_acc = np.mean(preds[in_bin] == gt_binary[in_bin])
            avg_confs.append(avg_conf)
            avg_accs.append(avg_acc)
            bin_counts.append(bin_size)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.scatter(avg_confs, avg_accs, s=[c*10 for c in bin_counts], alpha=0.6, label=f'T={temperature:.2f}')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Reliability diagram saved to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal temperature for calibration"
    )
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="CSV with predictions (must have 'logit_diff' and 'ground_truth' columns)"
    )
    parser.add_argument(
        "--output_temp",
        type=str,
        default="results/optimal_temperature.txt",
        help="Where to save optimal temperature value"
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins for ECE computation"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save reliability diagram (optional)"
    )

    args = parser.parse_args()

    # Load predictions
    print(f"Loading predictions from {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)

    required_cols = ['logit_diff', 'ground_truth']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain '{col}' column")

    logit_diffs = df['logit_diff'].values
    ground_truth = df['ground_truth'].values

    print(f"Loaded {len(df)} predictions")

    # Find optimal temperature
    print("\nOptimizing temperature to minimize ECE...")
    optimal_T, ece_uncal, ece_cal = find_optimal_temperature(
        logit_diffs, ground_truth, n_bins=args.n_bins
    )

    # Compute Brier scores
    gt_binary = (ground_truth == 'SPEECH').astype(int)
    uncal_probs = apply_temperature_scaling(logit_diffs, 1.0)
    cal_probs = apply_temperature_scaling(logit_diffs, optimal_T)
    brier_uncal = compute_brier_score(uncal_probs, gt_binary)
    brier_cal = compute_brier_score(cal_probs, gt_binary)

    # Report
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"\nOptimal temperature: {optimal_T:.3f}")
    print(f"\nExpected Calibration Error (ECE):")
    print(f"  Uncalibrated (T=1.0): {ece_uncal:.4f}")
    print(f"  Calibrated (T={optimal_T:.2f}): {ece_cal:.4f}")
    print(f"  Improvement: {ece_uncal - ece_cal:.4f} ({(1-ece_cal/ece_uncal)*100:.1f}% reduction)")

    print(f"\nBrier Score:")
    print(f"  Uncalibrated: {brier_uncal:.4f}")
    print(f"  Calibrated:   {brier_cal:.4f}")
    print(f"  Improvement: {brier_uncal - brier_cal:.4f}")

    # Save temperature
    output_path = Path(args.output_temp)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{optimal_T:.6f}\n")
    print(f"\nOptimal temperature saved to: {args.output_temp}")

    # Plot reliability diagram
    if args.plot:
        print("\nGenerating reliability diagram...")
        plot_reliability_diagram(
            logit_diffs, ground_truth, optimal_T,
            n_bins=args.n_bins, output_path=args.plot
        )

    print("\nUsage for test evaluation:")
    print(f"  python scripts/evaluate_with_logits.py \\")
    print(f"    --checkpoint <CHECKPOINT_DIR> \\")
    print(f"    --test_csv <TEST_CSV> \\")
    print(f"    --temperature {optimal_T:.6f} \\")
    print(f"    --output_csv results/test_calibrated.csv")
    print()


if __name__ == "__main__":
    main()
