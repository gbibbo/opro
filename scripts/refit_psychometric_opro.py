#!/usr/bin/env python3
"""
Sprint 9: Refit psychometric curves with OPRO-optimized prompt.

After OPRO optimization completes, this script:
1. Loads best prompt from OPRO results
2. Evaluates full dev set with best prompt (if not already done)
3. Refits psychometric curves (duration + SNR stratified)
4. Compares thresholds to baseline
5. Generates comparison report

Reuses Sprint 7/8 psychometric fitting infrastructure.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_prompt import evaluate_prompt
from fit_psychometric_curves import (
    analyze_duration_curves,
    plot_duration_curves,
)
from fit_snr_curves_stratified import (
    analyze_snr_stratified_by_duration,
)
from qsm.models import Qwen2AudioClassifier


def load_baseline_thresholds(baseline_path: Path) -> dict:
    """Load baseline thresholds for comparison."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    return baseline


def compare_thresholds(baseline: dict, opro: dict) -> dict:
    """
    Compare OPRO thresholds to baseline.

    Args:
        baseline: Baseline threshold dict
        opro: OPRO threshold dict

    Returns:
        Comparison dict with deltas
    """
    comparison = {
        "baseline": baseline,
        "opro": opro,
        "improvements": {},
    }

    # Duration DT75
    if "duration" in baseline and "duration" in opro:
        base_dt75 = baseline["duration"]["overall"]["dt75"]
        opro_dt75 = opro["duration"]["overall"]["dt75"]
        delta_dt75 = opro_dt75 - base_dt75
        pct_change = (delta_dt75 / base_dt75) * 100

        comparison["improvements"]["dt75"] = {
            "baseline_ms": base_dt75,
            "opro_ms": opro_dt75,
            "delta_ms": delta_dt75,
            "pct_change": pct_change,
            "improved": delta_dt75 < 0,  # Lower is better
        }

    # SNR-75 at 1000ms (if available)
    if "snr_stratified" in opro and "1000" in opro["snr_stratified"]:
        # Baseline SNR-75 at 1000ms
        baseline_snr_path = baseline.get("snr_stratified_path")
        if baseline_snr_path and Path(baseline_snr_path).exists():
            with open(baseline_snr_path) as f:
                baseline_snr = json.load(f)

            base_snr75 = baseline_snr["1000"]["snr75"]
            opro_snr75 = opro["snr_stratified"]["1000"]["snr75"]
            delta_snr75 = opro_snr75 - base_snr75
            pct_change = (delta_snr75 / abs(base_snr75)) * 100 if base_snr75 != 0 else 0

            comparison["improvements"]["snr75_1000ms"] = {
                "baseline_db": base_snr75,
                "opro_db": opro_snr75,
                "delta_db": delta_snr75,
                "pct_change": pct_change,
                "improved": delta_snr75 < 0,  # Lower (more negative) is better
            }

    return comparison


def generate_comparison_report(comparison: dict, output_path: Path, best_prompt: str):
    """Generate markdown comparison report."""
    report = f"""# OPRO Optimization Results - Comparison Report

**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status**: OPRO optimization complete

---

## Best Prompt Found

```
{best_prompt}
```

---

## Threshold Comparison

### Duration Threshold (DT75)

"""

    if "dt75" in comparison["improvements"]:
        dt75 = comparison["improvements"]["dt75"]
        status = "✅ IMPROVED" if dt75["improved"] else "❌ NO IMPROVEMENT"

        report += f"""| Metric | Baseline | OPRO | Delta | Change |
|--------|----------|------|-------|--------|
| **DT75** | {dt75['baseline_ms']:.1f} ms | {dt75['opro_ms']:.1f} ms | {dt75['delta_ms']:+.1f} ms | {dt75['pct_change']:+.1f}% |

**Status**: {status}

"""

    if "snr75_1000ms" in comparison["improvements"]:
        snr75 = comparison["improvements"]["snr75_1000ms"]
        status = "✅ IMPROVED" if snr75["improved"] else "❌ NO IMPROVEMENT"

        report += f"""### SNR Threshold (SNR-75 at 1000ms)

| Metric | Baseline | OPRO | Delta | Change |
|--------|----------|------|-------|--------|
| **SNR-75** | {snr75['baseline_db']:.1f} dB | {snr75['opro_db']:.1f} dB | {snr75['delta_db']:+.1f} dB | {snr75['pct_change']:+.1f}% |

**Status**: {status}

"""

    report += """---

## Interpretation

"""

    # Add interpretation based on results
    if "dt75" in comparison["improvements"]:
        dt75 = comparison["improvements"]["dt75"]
        if dt75["improved"]:
            report += f"- **Duration**: OPRO prompt reduced DT75 by {abs(dt75['delta_ms']):.1f}ms ({abs(dt75['pct_change']):.1f}%), indicating better performance on short-duration clips.\n"
        else:
            report += f"- **Duration**: OPRO prompt increased DT75 by {dt75['delta_ms']:.1f}ms ({dt75['pct_change']:.1f}%), indicating slight degradation on duration sensitivity.\n"

    if "snr75_1000ms" in comparison["improvements"]:
        snr75 = comparison["improvements"]["snr75_1000ms"]
        if snr75["improved"]:
            report += f"- **SNR**: OPRO prompt reduced SNR-75 by {abs(snr75['delta_db']):.1f}dB ({abs(snr75['pct_change']):.1f}%), indicating better noise robustness.\n"
        else:
            report += f"- **SNR**: OPRO prompt increased SNR-75 by {snr75['delta_db']:.1f}dB ({snr75['pct_change']:.1f}%), indicating slight degradation on noise tolerance.\n"

    report += "\n---\n\n## Next Steps\n\n"
    report += "1. ✅ Review OPRO results and thresholds\n"
    report += "2. ⏭️ Evaluate best prompt on test set (ONE TIME ONLY)\n"
    report += "3. ⏭️ Create final report with baseline vs OPRO comparison\n"
    report += "4. ⏭️ Tag release: v2.0-opro-baseline\n"

    with open(output_path, "w") as f:
        f.write(report)

    print(f"\nComparison report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sprint 9: Refit psychometric curves with OPRO prompt")
    parser.add_argument(
        "--opro_dir",
        type=Path,
        default=Path("results/sprint9_opro"),
        help="OPRO results directory (contains best_prompt.txt)",
    )
    parser.add_argument(
        "--baseline_results",
        type=Path,
        default=Path("results/psychometric_curves/psychometric_results.json"),
        help="Baseline psychometric results (for comparison)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/conditions_final/conditions_manifest_split.parquet"),
        help="Manifest path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: opro_dir/psychometric_opro)",
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
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Bootstrap samples for CI",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation (use existing predictions in opro_dir)",
    )

    args = parser.parse_args()

    # Set output dir
    if args.output_dir is None:
        args.output_dir = args.opro_dir / "psychometric_opro"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("SPRINT 9: REFIT PSYCHOMETRIC CURVES WITH OPRO PROMPT")
    print(f"{'='*60}")
    print(f"OPRO results: {args.opro_dir}")
    print(f"Baseline: {args.baseline_results}")
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

    # Check if predictions already exist
    predictions_path = args.opro_dir / "dev_predictions.parquet"

    if not predictions_path.exists() and not args.skip_eval:
        # Need to evaluate full dev set with best prompt
        print(f"\nEvaluating dev set with OPRO prompt...")

        ba_clip, ba_cond, metrics = evaluate_prompt(
            prompt=best_prompt,
            manifest_path=args.manifest,
            model_name=args.model_name,
            device=args.device,
            split="dev",
            seed=args.seed,
            save_predictions=True,
            output_dir=args.opro_dir,
        )

        print(f"\nEvaluation complete:")
        print(f"  BA_clip: {ba_clip:.3f}")
        print(f"  BA_conditions: {ba_cond:.3f}")

    elif predictions_path.exists():
        print(f"\nUsing existing predictions: {predictions_path}")
    else:
        print(f"\nERROR: No predictions found and --skip_eval specified")
        return 1

    # Load predictions
    predictions_df = pd.read_parquet(predictions_path)
    print(f"Loaded {len(predictions_df)} predictions")

    # Refit duration curves
    print(f"\n{'='*60}")
    print("REFITTING DURATION CURVES")
    print(f"{'='*60}")

    duration_results = analyze_duration_curves(
        predictions_df,
        args.output_dir,
        n_bootstrap=args.n_bootstrap,
    )

    # Plot duration curves
    plot_duration_curves(
        duration_results,
        predictions_df,
        args.output_dir / "duration_curve.png",
    )

    # Refit SNR curves (stratified by duration)
    print(f"\n{'='*60}")
    print("REFITTING SNR CURVES (STRATIFIED)")
    print(f"{'='*60}")

    # Check if we have factorial SNR×Duration data
    snr_duration_manifest = Path("data/processed/snr_duration_crossed/metadata.csv")
    if snr_duration_manifest.exists():
        print(f"Found factorial SNR×Duration dataset: {snr_duration_manifest}")
        print("Refitting stratified SNR curves...")

        # Load factorial predictions (if available)
        factorial_predictions_path = Path("results/sprint8_factorial/predictions.parquet")
        if factorial_predictions_path.exists():
            snr_results = analyze_snr_stratified_by_duration(
                pd.read_parquet(factorial_predictions_path),
                args.output_dir,
                n_bootstrap=args.n_bootstrap,
            )
        else:
            print("  Warning: Factorial predictions not found, skipping SNR refit")
            snr_results = {}
    else:
        print("  No factorial SNR×Duration dataset found, skipping SNR refit")
        snr_results = {}

    # Save OPRO psychometric results
    opro_results = {
        "prompt": best_prompt,
        "duration": duration_results,
        "snr_stratified": snr_results,
        "metadata": {
            "method": "MLE binomial fitting",
            "gamma": 0.5,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "n_predictions": len(predictions_df),
            "source": "OPRO optimization (Sprint 9)",
        },
    }

    results_path = args.output_dir / "psychometric_results.json"
    with open(results_path, "w") as f:
        json.dump(opro_results, f, indent=2)

    print(f"\nSaved OPRO psychometric results to: {results_path}")

    # Load baseline for comparison
    print(f"\n{'='*60}")
    print("COMPARING TO BASELINE")
    print(f"{'='*60}")

    if args.baseline_results.exists():
        with open(args.baseline_results) as f:
            baseline_results = json.load(f)

        baseline_results["snr_stratified_path"] = str(
            Path("results/sprint8_stratified/snr_stratified_results.json")
        )

        comparison = compare_thresholds(baseline_results, opro_results)

        # Print comparison
        if "dt75" in comparison["improvements"]:
            dt75 = comparison["improvements"]["dt75"]
            print(f"\nDuration DT75:")
            print(f"  Baseline: {dt75['baseline_ms']:.1f} ms")
            print(f"  OPRO: {dt75['opro_ms']:.1f} ms")
            print(f"  Delta: {dt75['delta_ms']:+.1f} ms ({dt75['pct_change']:+.1f}%)")
            print(f"  Status: {'✅ IMPROVED' if dt75['improved'] else '❌ NO IMPROVEMENT'}")

        if "snr75_1000ms" in comparison["improvements"]:
            snr75 = comparison["improvements"]["snr75_1000ms"]
            print(f"\nSNR-75 (1000ms):")
            print(f"  Baseline: {snr75['baseline_db']:.1f} dB")
            print(f"  OPRO: {snr75['opro_db']:.1f} dB")
            print(f"  Delta: {snr75['delta_db']:+.1f} dB ({snr75['pct_change']:+.1f}%)")
            print(f"  Status: {'✅ IMPROVED' if snr75['improved'] else '❌ NO IMPROVEMENT'}")

        # Save comparison
        comparison_path = args.output_dir / "baseline_vs_opro.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)

        print(f"\nComparison saved to: {comparison_path}")

        # Generate comparison report
        report_path = args.opro_dir / "comparison_report.md"
        generate_comparison_report(comparison, report_path, best_prompt)

    else:
        print(f"Warning: Baseline results not found at {args.baseline_results}")

    print(f"\n{'='*60}")
    print("PSYCHOMETRIC REFIT COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review comparison report")
    print("  2. Evaluate best prompt on test set (run_opro_test.py)")
    print("  3. Tag release: v2.0-opro-baseline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
