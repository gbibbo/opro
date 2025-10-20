#!/usr/bin/env python3
"""
Robust Statistical Evaluation Pipeline

Compares multiple models with proper statistical methodology:
1. Logit-based evaluation (faster, more stable than generate())
2. Bootstrap confidence intervals
3. McNemar's test for paired comparison
4. Temperature scaling for calibration

Usage:
    python scripts/run_robust_evaluation.py \
        --models attention_only mlp qwen3_omni \
        --test_csv data/processed/normalized_clips/test_metadata.csv
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict
import argparse
import sys
import numpy as np

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: List[str], description: str) -> subprocess.CompletedProcess:
    """Run a command and print status."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=project_root, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with code {result.returncode}")
        sys.exit(1)

    return result


def bootstrap_ci(predictions: np.ndarray, ground_truth: np.ndarray, n_bootstrap: int = 10000, alpha: float = 0.05) -> tuple:
    """Compute bootstrap confidence interval for accuracy."""
    n = len(predictions)
    accuracies = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        pred_boot = predictions[indices]
        truth_boot = ground_truth[indices]

        # Compute accuracy
        acc = np.mean(pred_boot == truth_boot)
        accuracies.append(acc)

    # Percentile method
    lower = np.percentile(accuracies, 100 * alpha / 2)
    upper = np.percentile(accuracies, 100 * (1 - alpha / 2))

    return lower, upper


def evaluate_model(model_name: str, checkpoint_path: str, test_csv: str, results_dir: Path) -> Dict:
    """Evaluate a single model using logit scoring."""

    output_json = results_dir / f"{model_name}_predictions.json"

    # Run evaluation
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "test_with_logit_scoring.py"),
        "--checkpoint", checkpoint_path,
        "--test_csv", test_csv,
        "--save_predictions", str(output_json),
    ]

    run_command(cmd, f"Evaluating {model_name}")

    # Load results
    with open(output_json) as f:
        data = json.load(f)

    # Compute bootstrap CI
    predictions = np.array(data["predictions"])
    ground_truth = np.array(data["ground_truth"])

    lower, upper = bootstrap_ci(predictions, ground_truth)

    return {
        "name": model_name,
        "checkpoint": checkpoint_path,
        "accuracy": data["accuracy"],
        "bootstrap_ci_95": {"lower": lower, "upper": upper},
        "predictions_file": str(output_json),
    }


def compare_models(model_results: List[Dict], results_dir: Path):
    """Run pairwise McNemar tests between all models."""

    print(f"\n{'='*80}")
    print("PAIRWISE MODEL COMPARISONS (McNemar's Test)")
    print(f"{'='*80}\n")

    comparisons = []

    for i in range(len(model_results)):
        for j in range(i + 1, len(model_results)):
            model_a = model_results[i]
            model_b = model_results[j]

            print(f"\nComparing {model_a['name']} vs {model_b['name']}...")

            # Run McNemar test
            cmd = [
                sys.executable,
                str(project_root / "scripts" / "compare_models_mcnemar.py"),
                "--model_a", model_a["predictions_file"],
                "--model_b", model_b["predictions_file"],
                "--name_a", model_a["name"],
                "--name_b", model_b["name"],
            ]

            run_command(cmd, f"McNemar: {model_a['name']} vs {model_b['name']}")

            # Load McNemar results
            mcnemar_file = project_root / "results" / "mcnemar_comparison.json"
            with open(mcnemar_file) as f:
                mcnemar_data = json.load(f)

            comparisons.append({
                "model_a": model_a["name"],
                "model_b": model_b["name"],
                "mcnemar": mcnemar_data["mcnemar"],
            })

    return comparisons


def generate_report(model_results: List[Dict], comparisons: List[Dict], output_file: Path):
    """Generate comprehensive evaluation report."""

    report = []
    report.append("=" * 80)
    report.append("ROBUST STATISTICAL EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Model Performance Table
    report.append("Model Performance (with 95% Bootstrap CI):")
    report.append("-" * 80)
    report.append(f"{'Model':<30} {'Accuracy':<15} {'95% CI':<20}")
    report.append("-" * 80)

    for result in sorted(model_results, key=lambda x: x["accuracy"], reverse=True):
        ci = result["bootstrap_ci_95"]
        report.append(
            f"{result['name']:<30} "
            f"{result['accuracy']:.3f} ({result['accuracy']*100:.1f}%)  "
            f"[{ci['lower']:.3f}, {ci['upper']:.3f}]"
        )

    report.append("-" * 80)
    report.append("")

    # Pairwise Comparisons
    report.append("Pairwise Statistical Comparisons (McNemar's Test):")
    report.append("-" * 80)
    report.append(f"{'Model A vs Model B':<40} {'χ²':<10} {'p-value':<10} {'Significant?'}")
    report.append("-" * 80)

    for comp in comparisons:
        mcn = comp["mcnemar"]
        sig = "Yes (p<0.05)" if mcn["p_value"] < 0.05 else "No (p≥0.05)"
        report.append(
            f"{comp['model_a']} vs {comp['model_b']:<40} "
            f"{mcn['chi2_statistic']:<10.4f} "
            f"{mcn['p_value']:<10.4f} "
            f"{sig}"
        )

    report.append("-" * 80)
    report.append("")

    # Recommendations
    report.append("Recommendations:")
    report.append("-" * 80)

    # Find best model(s)
    best_acc = max(r["accuracy"] for r in model_results)
    best_models = [r for r in model_results if r["accuracy"] == best_acc]

    if len(best_models) == 1:
        best = best_models[0]
        report.append(f"✓ Best model: {best['name']} ({best['accuracy']:.1%})")

        # Check if significantly better
        sig_better = []
        for comp in comparisons:
            if comp["model_a"] == best["name"] and comp["mcnemar"]["p_value"] < 0.05:
                sig_better.append(comp["model_b"])
            elif comp["model_b"] == best["name"] and comp["mcnemar"]["p_value"] < 0.05:
                sig_better.append(comp["model_a"])

        if sig_better:
            report.append(f"  Significantly better than: {', '.join(sig_better)}")
        else:
            report.append("  (but not statistically significantly better)")
    else:
        report.append(f"✓ Tie between: {', '.join(m['name'] for m in best_models)}")
        report.append("  No statistically significant difference between top models")

    # Check for overlapping CIs
    report.append("")
    report.append("Confidence Interval Overlaps:")
    for i in range(len(model_results)):
        for j in range(i + 1, len(model_results)):
            m_a = model_results[i]
            m_b = model_results[j]
            ci_a = m_a["bootstrap_ci_95"]
            ci_b = m_b["bootstrap_ci_95"]

            # Check overlap
            overlap = not (ci_a["upper"] < ci_b["lower"] or ci_b["upper"] < ci_a["lower"])

            if overlap:
                report.append(f"  ≈ {m_a['name']} and {m_b['name']}: CIs overlap (models may be equivalent)")

    report.append("=" * 80)

    # Write report
    report_text = "\n".join(report)
    print(f"\n{report_text}")

    with open(output_file, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Robust statistical evaluation pipeline")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names to evaluate (e.g., attention_only mlp qwen3_omni)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="Checkpoint paths (must match --models order). If not provided, uses defaults.",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/processed/normalized_clips/test_metadata.csv",
        help="Path to test metadata CSV",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/robust_evaluation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Setup paths
    results_dir = project_root / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    test_csv = project_root / args.test_csv

    # Default checkpoints
    default_checkpoints = {
        "attention_only": "checkpoints/qwen2_audio_speech_detection_multiseed/seed_42/final",
        "mlp": "checkpoints/with_mlp_seed_42/final",
        "qwen3_omni": "Qwen/Qwen3-Omni-30B-A3B-Instruct",  # HuggingFace model ID
    }

    # Build checkpoint list
    if args.checkpoints:
        if len(args.checkpoints) != len(args.models):
            print("ERROR: Number of checkpoints must match number of models")
            sys.exit(1)
        checkpoints = args.checkpoints
    else:
        checkpoints = [default_checkpoints.get(m, m) for m in args.models]

    print(f"\n{'='*80}")
    print("ROBUST STATISTICAL EVALUATION PIPELINE")
    print(f"{'='*80}")
    print(f"\nModels to evaluate: {', '.join(args.models)}")
    print(f"Test set: {test_csv}")
    print(f"Results directory: {results_dir}")
    print()

    # Step 1: Evaluate each model
    model_results = []
    for model_name, checkpoint in zip(args.models, checkpoints):
        result = evaluate_model(model_name, checkpoint, str(test_csv), results_dir)
        model_results.append(result)

    # Step 2: Pairwise comparisons
    comparisons = compare_models(model_results, results_dir)

    # Step 3: Generate report
    report_file = results_dir / "evaluation_report.txt"
    generate_report(model_results, comparisons, report_file)

    # Step 4: Save consolidated JSON
    consolidated = {
        "test_csv": str(test_csv),
        "models": model_results,
        "comparisons": comparisons,
    }

    consolidated_file = results_dir / "consolidated_results.json"
    with open(consolidated_file, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\nConsolidated results saved to: {consolidated_file}")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
