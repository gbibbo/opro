#!/usr/bin/env python3
"""
Statistical comparison of two classifiers with McNemar test and Bootstrap CI.

Implements:
1. McNemar's test: For paired binary classifiers on same test set
2. Bootstrap confidence intervals: Resampling-based robust CI for accuracy
3. Contingency table analysis

References:
- McNemar (1947): "Note on the sampling error of the difference between correlated proportions"
- Dietterich (1998): "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"
- Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def mcnemar_test(predictions_A, predictions_B, ground_truth):
    """
    Perform McNemar's test to compare two classifiers.

    McNemar's test is appropriate for:
    - Paired binary classifiers (same test set)
    - Nominal categorical data
    - Tests the null hypothesis that marginal probabilities are equal

    Args:
        predictions_A: Predictions from model A (array of binary values)
        predictions_B: Predictions from model B (array of binary values)
        ground_truth: True labels (array of binary values)

    Returns:
        dict with McNemar statistics, p-value, and interpretation
    """
    # Create binary correct/wrong arrays
    correct_A = (predictions_A == ground_truth)
    correct_B = (predictions_B == ground_truth)

    # Contingency table:
    #           B correct | B wrong
    # A correct |   n00   |   n01   |
    # A wrong   |   n10   |   n11   |

    n00 = np.sum(correct_A & correct_B)      # Both correct
    n01 = np.sum(correct_A & ~correct_B)     # A correct, B wrong
    n10 = np.sum(~correct_A & correct_B)     # A wrong, B correct
    n11 = np.sum(~correct_A & ~correct_B)    # Both wrong

    # McNemar test statistic (with continuity correction)
    # H0: n01 = n10 (models equally accurate)
    # H1: n01 ≠ n10 (models differ in accuracy)

    if n01 + n10 == 0:
        # Both models make identical predictions
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(n01 - n10) - 1)**2 / (n01 + n10)  # With continuity correction
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # Effect size: Difference in disagreement
    disagreements = n01 + n10
    effect_size = (n01 - n10) / disagreements if disagreements > 0 else 0.0

    return {
        'n00_both_correct': n00,
        'n01_A_correct_B_wrong': n01,
        'n10_A_wrong_B_correct': n10,
        'n11_both_wrong': n11,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'significant_at_0.05': p_value < 0.05,
        'significant_at_0.01': p_value < 0.01,
        'effect_size': effect_size,
        'disagreements': disagreements
    }


def bootstrap_confidence_interval(correct_array, n_bootstrap=10000, confidence_level=0.95, random_state=42):
    """
    Compute bootstrap confidence interval for accuracy.

    Bootstrap resampling:
    1. Resample n times WITH replacement from the data
    2. Compute accuracy for each resample
    3. Take percentiles to get CI

    Args:
        correct_array: Binary array (1=correct, 0=wrong)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (0.95 = 95% CI)
        random_state: Random seed

    Returns:
        dict with mean, CI bounds, and bootstrap distribution
    """
    np.random.seed(random_state)

    n = len(correct_array)
    bootstrap_accuracies = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        resampled = correct_array[indices]
        accuracy = resampled.mean() * 100
        bootstrap_accuracies.append(accuracy)

    bootstrap_accuracies = np.array(bootstrap_accuracies)

    # Percentiles for CI
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_accuracies, lower_percentile)
    ci_upper = np.percentile(bootstrap_accuracies, upper_percentile)
    mean_acc = bootstrap_accuracies.mean()
    std_acc = bootstrap_accuracies.std()

    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'bootstrap_distribution': bootstrap_accuracies
    }


def compare_two_models(predictions_A, predictions_B, ground_truth, labels, n_bootstrap=10000):
    """
    Complete statistical comparison of two models.

    Returns comprehensive statistics including:
    - Accuracy for each model
    - McNemar test results
    - Bootstrap confidence intervals
    - Per-class breakdown
    """
    # Convert to numpy arrays
    predictions_A = np.array(predictions_A)
    predictions_B = np.array(predictions_B)
    ground_truth = np.array(ground_truth)

    # Overall accuracy
    correct_A = (predictions_A == ground_truth)
    correct_B = (predictions_B == ground_truth)

    acc_A = correct_A.mean() * 100
    acc_B = correct_B.mean() * 100

    # McNemar test
    mcnemar_results = mcnemar_test(predictions_A, predictions_B, ground_truth)

    # Bootstrap CI
    bootstrap_A = bootstrap_confidence_interval(correct_A, n_bootstrap=n_bootstrap)
    bootstrap_B = bootstrap_confidence_interval(correct_B, n_bootstrap=n_bootstrap)

    # Per-class accuracy
    unique_labels = np.unique(ground_truth)
    per_class_A = {}
    per_class_B = {}

    for label in unique_labels:
        mask = (ground_truth == label)
        if mask.sum() > 0:
            per_class_A[label] = (predictions_A[mask] == ground_truth[mask]).mean() * 100
            per_class_B[label] = (predictions_B[mask] == ground_truth[mask]).mean() * 100

    return {
        'model_A': {
            'accuracy': acc_A,
            'n_correct': correct_A.sum(),
            'n_total': len(correct_A),
            'bootstrap_ci': bootstrap_A,
            'per_class': per_class_A
        },
        'model_B': {
            'accuracy': acc_B,
            'n_correct': correct_B.sum(),
            'n_total': len(correct_B),
            'bootstrap_ci': bootstrap_B,
            'per_class': per_class_B
        },
        'mcnemar': mcnemar_results,
        'accuracy_difference': acc_A - acc_B
    }


def main():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of two models (McNemar + Bootstrap)"
    )
    parser.add_argument(
        "--predictions_A",
        type=str,
        required=True,
        help="CSV with predictions from model A (columns: clip_id, ground_truth, prediction)"
    )
    parser.add_argument(
        "--predictions_B",
        type=str,
        required=True,
        help="CSV with predictions from model B (same format)"
    )
    parser.add_argument(
        "--model_A_name",
        type=str,
        default="Model A",
        help="Name for model A (for display)"
    )
    parser.add_argument(
        "--model_B_name",
        type=str,
        default="Model B",
        help="Name for model B (for display)"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples (default: 10000)"
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default=None,
        help="Path to save detailed report (optional)"
    )

    args = parser.parse_args()

    print("="*70)
    print("STATISTICAL MODEL COMPARISON")
    print("="*70)

    # Load predictions
    print(f"\nLoading predictions...")
    df_A = pd.read_csv(args.predictions_A)
    df_B = pd.read_csv(args.predictions_B)

    print(f"  {args.model_A_name}: {len(df_A)} predictions")
    print(f"  {args.model_B_name}: {len(df_B)} predictions")

    # Ensure same order (merge on clip_id AND audio_path to handle variants)
    merged = df_A.merge(df_B, on=['clip_id', 'audio_path'], suffixes=('_A', '_B'))

    if len(merged) != len(df_A) or len(merged) != len(df_B):
        print(f"\nWARNING: Predictions don't match perfectly!")
        print(f"  Model A: {len(df_A)} samples")
        print(f"  Model B: {len(df_B)} samples")
        print(f"  Matched: {len(merged)} samples")

    # Extract predictions using 'correct' column (already computed)
    # Convert boolean correct to predictions that match ground_truth
    ground_truth = merged['ground_truth_A'].values  # Should be same for both

    # Create predictions from correct column: if correct, pred=gt, else pred=opposite
    def make_predictions(ground_truth, correct):
        preds = []
        for gt, is_correct in zip(ground_truth, correct):
            if is_correct:
                preds.append(gt)
            else:
                # Wrong prediction = opposite class
                preds.append('NONSPEECH' if gt == 'SPEECH' else 'SPEECH')
        return np.array(preds)

    predictions_A = make_predictions(ground_truth, merged['correct_A'].values)
    predictions_B = make_predictions(ground_truth, merged['correct_B'].values)
    labels = merged['clip_id'].values

    # Verify ground truth matches
    assert np.all(merged['ground_truth_A'] == merged['ground_truth_B']), \
        "Ground truth mismatch between A and B!"

    # Compare
    print(f"\nRunning statistical comparison (n_bootstrap={args.n_bootstrap})...")
    results = compare_two_models(
        predictions_A, predictions_B, ground_truth, labels,
        n_bootstrap=args.n_bootstrap
    )

    # Print results
    print(f"\n{'='*70}")
    print("OVERALL ACCURACY")
    print(f"{'='*70}")

    print(f"\n{args.model_A_name}:")
    print(f"  Accuracy: {results['model_A']['n_correct']}/{results['model_A']['n_total']} "
          f"= {results['model_A']['accuracy']:.1f}%")
    print(f"  Bootstrap 95% CI: [{results['model_A']['bootstrap_ci']['ci_lower']:.1f}%, "
          f"{results['model_A']['bootstrap_ci']['ci_upper']:.1f}%]")
    print(f"  CI width: {results['model_A']['bootstrap_ci']['ci_width']:.1f}%")

    print(f"\n{args.model_B_name}:")
    print(f"  Accuracy: {results['model_B']['n_correct']}/{results['model_B']['n_total']} "
          f"= {results['model_B']['accuracy']:.1f}%")
    print(f"  Bootstrap 95% CI: [{results['model_B']['bootstrap_ci']['ci_lower']:.1f}%, "
          f"{results['model_B']['bootstrap_ci']['ci_upper']:.1f}%]")
    print(f"  CI width: {results['model_B']['bootstrap_ci']['ci_width']:.1f}%")

    print(f"\nAccuracy difference: {results['accuracy_difference']:+.1f}% "
          f"({args.model_A_name} - {args.model_B_name})")

    # Per-class accuracy
    print(f"\n{'='*70}")
    print("PER-CLASS ACCURACY")
    print(f"{'='*70}")

    for label in sorted(results['model_A']['per_class'].keys()):
        acc_A = results['model_A']['per_class'][label]
        acc_B = results['model_B']['per_class'][label]
        diff = acc_A - acc_B

        print(f"\n{label}:")
        print(f"  {args.model_A_name}: {acc_A:.1f}%")
        print(f"  {args.model_B_name}: {acc_B:.1f}%")
        print(f"  Difference: {diff:+.1f}%")

    # McNemar test
    print(f"\n{'='*70}")
    print("MCNEMAR'S TEST (Paired Comparison)")
    print(f"{'='*70}")

    mcnemar = results['mcnemar']

    print(f"\nContingency table:")
    print(f"                        {args.model_B_name:20s}")
    print(f"                     Correct      Wrong")
    print(f"{args.model_A_name:12s} Correct  {mcnemar['n00_both_correct']:6d}    {mcnemar['n01_A_correct_B_wrong']:6d}")
    print(f"             Wrong    {mcnemar['n10_A_wrong_B_correct']:6d}    {mcnemar['n11_both_wrong']:6d}")

    print(f"\nTest statistics:")
    print(f"  Chi-squared statistic: {mcnemar['chi2_statistic']:.4f}")
    print(f"  p-value: {mcnemar['p_value']:.4f}")
    print(f"  Disagreements: {mcnemar['disagreements']} samples")

    print(f"\nInterpretation:")
    if mcnemar['significant_at_0.01']:
        print(f"  HIGHLY SIGNIFICANT (p < 0.01)")
        print(f"    Models differ significantly in accuracy")
    elif mcnemar['significant_at_0.05']:
        print(f"  SIGNIFICANT (p < 0.05)")
        print(f"    Models differ in accuracy")
    else:
        print(f"  NOT SIGNIFICANT (p >= 0.05)")
        print(f"    No evidence that models differ in accuracy")

    if abs(mcnemar['effect_size']) > 0.5:
        print(f"  Effect size: {mcnemar['effect_size']:.3f} (LARGE)")
    elif abs(mcnemar['effect_size']) > 0.2:
        print(f"  Effect size: {mcnemar['effect_size']:.3f} (MEDIUM)")
    else:
        print(f"  Effect size: {mcnemar['effect_size']:.3f} (SMALL)")

    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}\n")

    if mcnemar['significant_at_0.05']:
        if results['accuracy_difference'] > 0:
            print(f"RECOMMENDATION: Use {args.model_A_name} - significantly better (p={mcnemar['p_value']:.4f})")
        else:
            print(f"RECOMMENDATION: Use {args.model_B_name} - significantly better (p={mcnemar['p_value']:.4f})")
    else:
        # CIs overlap?
        ci_A = results['model_A']['bootstrap_ci']
        ci_B = results['model_B']['bootstrap_ci']

        overlap = not (ci_A['ci_upper'] < ci_B['ci_lower'] or ci_B['ci_upper'] < ci_A['ci_lower'])

        if overlap:
            print(f"No significant difference detected:")
            print(f"   - McNemar p={mcnemar['p_value']:.4f} (not significant)")
            print(f"   - Bootstrap CIs overlap")
            print(f"   -> Models perform equivalently on this test set")
            print(f"   -> Choose based on other criteria (size, speed, etc.)")
        else:
            print(f"Borderline case:")
            print(f"   - McNemar p={mcnemar['p_value']:.4f} (not significant)")
            print(f"   - Bootstrap CIs don't overlap")
            print(f"   -> Increase test set size for more statistical power")

    print()

    # Save report if requested
    if args.output_report:
        with open(args.output_report, 'w') as f:
            f.write("Statistical Comparison Report\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model A: {args.model_A_name}\n")
            f.write(f"Model B: {args.model_B_name}\n")
            f.write(f"Test samples: {len(ground_truth)}\n")
            f.write(f"Bootstrap samples: {args.n_bootstrap}\n\n")

            f.write("RESULTS\n")
            f.write("-"*70 + "\n")
            f.write(f"Model A accuracy: {results['model_A']['accuracy']:.1f}% "
                   f"[{results['model_A']['bootstrap_ci']['ci_lower']:.1f}%, "
                   f"{results['model_A']['bootstrap_ci']['ci_upper']:.1f}%]\n")
            f.write(f"Model B accuracy: {results['model_B']['accuracy']:.1f}% "
                   f"[{results['model_B']['bootstrap_ci']['ci_lower']:.1f}%, "
                   f"{results['model_B']['bootstrap_ci']['ci_upper']:.1f}%]\n")
            f.write(f"Difference: {results['accuracy_difference']:+.1f}%\n\n")

            f.write("MCNEMAR TEST\n")
            f.write("-"*70 + "\n")
            f.write(f"Chi-squared = {mcnemar['chi2_statistic']:.4f}, p = {mcnemar['p_value']:.4f}\n")
            f.write(f"Significant at alpha=0.05: {mcnemar['significant_at_0.05']}\n")
            f.write(f"Significant at alpha=0.01: {mcnemar['significant_at_0.01']}\n")

        print(f"Detailed report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
