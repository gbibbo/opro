#!/usr/bin/env python3
"""
Sprint 9: Evaluate OPRO-optimized prompt on test set.

CRITICAL: This script should be run ONLY ONCE per OPRO optimization.
Test set is hold-out data and should not be used for prompt tuning.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_prompt import evaluate_prompt
from fit_psychometric_curves import analyze_duration_curves, plot_duration_curves
from qsm.models import Qwen2AudioClassifier


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sprint 9: Evaluate OPRO prompt on TEST set (ONE TIME ONLY)",
        epilog="WARNING: Test set should only be evaluated ONCE per optimization run!",
    )
    parser.add_argument(
        "--opro_dir",
        type=Path,
        default=Path("results/sprint9_opro"),
        help="OPRO results directory (contains best_prompt.txt)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/conditions_final/conditions_manifest_split.parquet"),
        help="Manifest path with test split",
    )
    parser.add_argument(
        "--baseline_test",
        type=Path,
        default=Path("results/test_set_final"),
        help="Baseline test results (for comparison)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: opro_dir/test_set_opro)",
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
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap samples for CI",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation (WARNING: Use with caution!)",
    )

    args = parser.parse_args()

    # Set output dir
    if args.output_dir is None:
        args.output_dir = args.opro_dir / "test_set_opro"

    # Check if test results already exist
    if args.output_dir.exists() and not args.force:
        print(f"\n{'='*60}")
        print("ERROR: Test results already exist!")
        print(f"{'='*60}")
        print(f"\nTest results found at: {args.output_dir}")
        print("\nTest set should only be evaluated ONCE per optimization run.")
        print("If you really want to re-evaluate, use --force flag.")
        print("\nExiting without evaluation.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SPRINT 9: EVALUATE OPRO PROMPT ON TEST SET")
    print(f"{'='*60}")
    print("\n⚠️  WARNING: Test set evaluation is ONE-TIME ONLY!")
    print("This ensures unbiased performance estimation.")
    print(f"\nOPRO results: {args.opro_dir}")
    print(f"Output: {args.output_dir}")

    # Load best prompt
    best_prompt_path = args.opro_dir / "best_prompt.txt"
    if not best_prompt_path.exists():
        print(f"\nERROR: Best prompt not found at {best_prompt_path}")
        print("Run OPRO optimization first!")
        return 1

    with open(best_prompt_path) as f:
        best_prompt = f.read().strip()

    print(f"\nBest prompt loaded:")
    print(f'  "{best_prompt}"')

    # Confirm evaluation
    if not args.force:
        print(f"\n{'='*60}")
        print("CONFIRMATION REQUIRED")
        print(f"{'='*60}")
        response = input("\nProceed with ONE-TIME test set evaluation? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("\nEvaluation cancelled.")
            return 0

    # Evaluate test set
    print(f"\n{'='*60}")
    print("EVALUATING TEST SET")
    print(f"{'='*60}")

    ba_clip, ba_cond, metrics = evaluate_prompt(
        prompt=best_prompt,
        manifest_path=args.manifest,
        model_name=args.model_name,
        device=args.device,
        split="test",
        seed=args.seed,
        save_predictions=True,
        output_dir=args.output_dir,
    )

    print(f"\nTest set evaluation complete:")
    print(f"  BA_clip: {ba_clip:.3f}")
    print(f"  BA_conditions: {ba_cond:.3f}")
    print(f"  Clip accuracy: {metrics['clip_accuracy']:.3f}")

    # Load predictions for psychometric curves
    predictions_path = args.output_dir / "test_predictions.parquet"
    predictions_df = pd.read_parquet(predictions_path)

    # Fit duration curves on test set
    print(f"\n{'='*60}")
    print("FITTING DURATION CURVES ON TEST SET")
    print(f"{'='*60}")

    duration_curves_dir = args.output_dir / "duration_curves"
    duration_curves_dir.mkdir(parents=True, exist_ok=True)

    duration_results = analyze_duration_curves(
        predictions_df,
        duration_curves_dir,
        n_bootstrap=args.n_bootstrap,
    )

    # Plot duration curves
    plot_duration_curves(
        duration_results,
        predictions_df,
        duration_curves_dir / "duration_curve.png",
    )

    # Save test results
    test_results = {
        "prompt": best_prompt,
        "ba_clip": ba_clip,
        "ba_conditions": ba_cond,
        "metrics": metrics,
        "duration_curves": duration_results,
        "metadata": {
            "split": "test",
            "seed": args.seed,
            "n_predictions": len(predictions_df),
            "n_clips": predictions_df["clip_id"].nunique(),
            "source": "OPRO optimization (Sprint 9)",
            "evaluation_date": pd.Timestamp.now().isoformat(),
        },
    }

    results_path = args.output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\nTest results saved to: {results_path}")

    # Compare to baseline test results
    print(f"\n{'='*60}")
    print("COMPARING TO BASELINE TEST RESULTS")
    print(f"{'='*60}")

    baseline_test_path = args.baseline_test / "test_results.json"
    if baseline_test_path.exists():
        with open(baseline_test_path) as f:
            baseline_test = json.load(f)

        print("\nTest Set Performance:")
        print(f"  Baseline BA_clip: {baseline_test.get('ba_clip', 'N/A'):.3f}")
        print(f"  OPRO BA_clip: {ba_clip:.3f}")
        delta_ba = ba_clip - baseline_test.get("ba_clip", 0)
        print(f"  Delta: {delta_ba:+.3f}")

        if "duration_curves" in baseline_test and "overall" in baseline_test["duration_curves"]:
            baseline_dt75 = baseline_test["duration_curves"]["overall"]["dt75"]
            opro_dt75 = duration_results.get("overall", {}).get("dt75")

            if opro_dt75 is not None:
                print(f"\nDuration DT75:")
                print(f"  Baseline: {baseline_dt75:.1f} ms")
                print(f"  OPRO: {opro_dt75:.1f} ms")
                delta_dt75 = opro_dt75 - baseline_dt75
                print(f"  Delta: {delta_dt75:+.1f} ms")

        # Save comparison
        comparison = {
            "baseline": {
                "ba_clip": baseline_test.get("ba_clip"),
                "dt75": baseline_test.get("duration_curves", {}).get("overall", {}).get("dt75"),
            },
            "opro": {
                "ba_clip": ba_clip,
                "dt75": duration_results.get("overall", {}).get("dt75"),
            },
            "delta": {
                "ba_clip": delta_ba,
                "dt75": delta_dt75 if opro_dt75 is not None else None,
            },
        }

        comparison_path = args.output_dir / "baseline_vs_opro_test.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\nComparison saved to: {comparison_path}")

    else:
        print(f"\nWarning: Baseline test results not found at {baseline_test_path}")

    print(f"\n{'='*60}")
    print("TEST SET EVALUATION COMPLETE")
    print(f"{'='*60}")
    print("\n✅ Test set has been evaluated (ONE TIME ONLY)")
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review test results and comparison")
    print("  2. Create final OPRO report")
    print("  3. Tag release: v2.0-opro-baseline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
