"""Statistical comparison of models using McNemar's test.

Compares paired predictions from two models on the same test set:
- McNemar's test for statistical significance
- Confusion matrix analysis
- Agreement analysis
"""

import sys
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from typing import List, Tuple

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    ground_truth: np.ndarray,
    continuity_correction: bool = True,
) -> Tuple[float, float, dict]:
    """
    Perform McNemar's test for paired binary classifiers.

    Args:
        predictions_a: Predictions from model A (0/1)
        predictions_b: Predictions from model B (0/1)
        ground_truth: True labels (0/1)
        continuity_correction: Apply continuity correction (recommended for small n)

    Returns:
        chi2_stat: Chi-square statistic
        p_value: Two-tailed p-value
        contingency_table: Dict with counts
    """
    # Check if predictions are paired
    assert len(predictions_a) == len(predictions_b) == len(ground_truth)

    # Compute correctness
    correct_a = predictions_a == ground_truth
    correct_b = predictions_b == ground_truth

    # Build contingency table
    # Format: [[both_correct, a_wrong_b_right],
    #          [a_right_b_wrong, both_wrong]]
    both_correct = np.sum(correct_a & correct_b)
    a_right_b_wrong = np.sum(correct_a & ~correct_b)
    a_wrong_b_right = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)

    contingency = {
        "both_correct": int(both_correct),
        "a_right_b_wrong": int(a_right_b_wrong),
        "a_wrong_b_right": int(a_wrong_b_right),
        "both_wrong": int(both_wrong),
    }

    # McNemar statistic (off-diagonal cells)
    b = a_right_b_wrong
    c = a_wrong_b_right

    if b + c == 0:
        # No disagreements - models are identical
        return 0.0, 1.0, contingency

    # Compute chi-square with optional continuity correction
    if continuity_correction:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2_stat = (b - c) ** 2 / (b + c)

    # Two-tailed p-value (df=1)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return chi2_stat, p_value, contingency


def load_predictions(predictions_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load predictions and ground truth from JSON file."""
    with open(predictions_file) as f:
        data = json.load(f)

    predictions = np.array(data["predictions"])
    ground_truth = np.array(data["ground_truth"])

    return predictions, ground_truth


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute basic classification metrics."""
    correct = predictions == ground_truth
    accuracy = correct.mean()

    # Assuming binary: 0=NONSPEECH, 1=SPEECH
    # (adjust if your encoding is different)
    speech_mask = ground_truth == 1
    nonspeech_mask = ground_truth == 0

    speech_acc = correct[speech_mask].mean() if speech_mask.any() else 0.0
    nonspeech_acc = correct[nonspeech_mask].mean() if nonspeech_mask.any() else 0.0

    return {
        "accuracy": accuracy,
        "speech_accuracy": speech_acc,
        "nonspeech_accuracy": nonspeech_acc,
        "n_correct": int(correct.sum()),
        "n_total": len(predictions),
    }


def main():
    """Compare two models using McNemar's test."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two models using McNemar's test"
    )
    parser.add_argument(
        "--model_a",
        type=str,
        required=True,
        help="Path to model A predictions JSON",
    )
    parser.add_argument(
        "--model_b",
        type=str,
        required=True,
        help="Path to model B predictions JSON",
    )
    parser.add_argument(
        "--name_a",
        type=str,
        default="Model A",
        help="Name for model A (for display)",
    )
    parser.add_argument(
        "--name_b",
        type=str,
        default="Model B",
        help="Name for model B (for display)",
    )
    parser.add_argument(
        "--no_correction",
        action="store_true",
        help="Disable continuity correction",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print("MCNEMAR'S TEST FOR MODEL COMPARISON")
    print(f"{'=' * 80}\n")

    # Load predictions
    print(f"Loading {args.name_a} predictions from: {args.model_a}")
    pred_a, truth_a = load_predictions(Path(args.model_a))

    print(f"Loading {args.name_b} predictions from: {args.model_b}")
    pred_b, truth_b = load_predictions(Path(args.model_b))

    # Verify ground truth matches
    if not np.array_equal(truth_a, truth_b):
        print("ERROR: Ground truth differs between models!")
        print("Models must be evaluated on the same test set.")
        sys.exit(1)

    ground_truth = truth_a
    n_samples = len(ground_truth)

    print(f"\nTest set size: {n_samples} samples")

    # Compute individual metrics
    print(f"\n{args.name_a} performance:")
    metrics_a = compute_metrics(pred_a, ground_truth)
    print(f"  Accuracy: {metrics_a['accuracy']:.1%} "
          f"({metrics_a['n_correct']}/{metrics_a['n_total']})")
    print(f"  SPEECH: {metrics_a['speech_accuracy']:.1%}")
    print(f"  NONSPEECH: {metrics_a['nonspeech_accuracy']:.1%}")

    print(f"\n{args.name_b} performance:")
    metrics_b = compute_metrics(pred_b, ground_truth)
    print(f"  Accuracy: {metrics_b['accuracy']:.1%} "
          f"({metrics_b['n_correct']}/{metrics_b['n_total']})")
    print(f"  SPEECH: {metrics_b['speech_accuracy']:.1%}")
    print(f"  NONSPEECH: {metrics_b['nonspeech_accuracy']:.1%}")

    # McNemar's test
    print(f"\n{'=' * 80}")
    print("MCNEMAR'S TEST RESULTS")
    print(f"{'=' * 80}\n")

    chi2_stat, p_value, contingency = mcnemar_test(
        pred_a, pred_b, ground_truth, continuity_correction=not args.no_correction
    )

    print("Contingency table (correctness):")
    print(f"  Both correct:     {contingency['both_correct']:3d}")
    print(f"  {args.name_a} right, {args.name_b} wrong: {contingency['a_right_b_wrong']:3d}")
    print(f"  {args.name_a} wrong, {args.name_b} right: {contingency['a_wrong_b_right']:3d}")
    print(f"  Both wrong:       {contingency['both_wrong']:3d}")

    disagreements = contingency["a_right_b_wrong"] + contingency["a_wrong_b_right"]
    print(f"\nTotal disagreements: {disagreements}/{n_samples} "
          f"({disagreements/n_samples:.1%})")

    print(f"\nMcNemar's chi-square statistic: {chi2_stat:.4f}")
    print(f"p-value (two-tailed): {p_value:.4f}")

    # Interpretation
    print(f"\n{'=' * 80}")
    print("INTERPRETATION")
    print(f"{'=' * 80}\n")

    alpha = 0.05
    if p_value < alpha:
        print(f"✓ Statistically significant difference (p < {alpha})")
        if contingency["a_right_b_wrong"] > contingency["a_wrong_b_right"]:
            print(f"  → {args.name_a} performs significantly better than {args.name_b}")
        else:
            print(f"  → {args.name_b} performs significantly better than {args.name_a}")
    else:
        print(f"✗ No statistically significant difference (p >= {alpha})")
        print(f"  → Cannot conclude that one model is better than the other")

    # Effect size (Cohen's g)
    # g = (b - c) / sqrt(b + c)
    if disagreements > 0:
        effect_size = (
            contingency["a_right_b_wrong"] - contingency["a_wrong_b_right"]
        ) / np.sqrt(disagreements)
        print(f"\nEffect size (Cohen's g): {effect_size:.4f}")
        if abs(effect_size) < 0.2:
            print("  (small effect)")
        elif abs(effect_size) < 0.5:
            print("  (medium effect)")
        else:
            print("  (large effect)")

    print(f"\n{'=' * 80}\n")

    # Save results
    results = {
        "model_a": {
            "name": args.name_a,
            "predictions_file": args.model_a,
            "metrics": metrics_a,
        },
        "model_b": {
            "name": args.name_b,
            "predictions_file": args.model_b,
            "metrics": metrics_b,
        },
        "mcnemar": {
            "chi2_statistic": float(chi2_stat),
            "p_value": float(p_value),
            "contingency_table": contingency,
            "continuity_correction": not args.no_correction,
        },
    }

    output_file = project_root / "results" / "mcnemar_comparison.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
