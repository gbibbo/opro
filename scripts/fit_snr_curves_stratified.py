#!/usr/bin/env python3
"""
Sprint 8: Fit stratified SNR psychometric curves by duration level.

Implements:
1. Separate MLE curves for each duration: {20, 80, 200, 1000} ms
2. Fixed gamma=0.5, free lapse parameter (Wichmann & Hill 2001)
3. Bootstrap CIs clustered by clip_id
4. Monotonicity verification (slope > 0)
5. Improved pseudo-R² (McFadden & Tjur) vs Sprint 7 overall
6. Paper-ready figures with 4 stratified curves

Primary metric: SNR-75 per duration level
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Import fitting functions from Sprint 7
import sys
sys.path.insert(0, str(Path(__file__).parent))

from fit_psychometric_curves import (
    psychometric_function,
    negative_log_likelihood,
    compute_pseudo_r_squared,
    fit_psychometric_mle,
    bootstrap_threshold,
)


def convert_to_json_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_json_serializable(obj.tolist())
    else:
        return obj


def analyze_snr_stratified_by_duration(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Fit separate SNR curves for each duration level.

    Args:
        predictions_df: DataFrame with duration_ms, snr_db, y_true, y_pred
        output_dir: Output directory
        n_bootstrap: Number of bootstrap samples for CIs

    Returns:
        Dict with results per duration + overall
    """
    print("\n" + "="*70)
    print("SPRINT 8: STRATIFIED SNR CURVES BY DURATION")
    print("="*70)
    print("\nMethodology:")
    print("  - Separate MLE fit for each duration: {20, 80, 200, 1000} ms")
    print("  - Fixed gamma=0.5 (binary task), free lapse lambda in [0, 0.1]")
    print("  - Bootstrap CIs (1000 samples, clustered by clip_id)")
    print("  - Monotonicity check: slope > 0 required")
    print("  - Primary metric: SNR-75 (dB) per duration")

    # Convert to binary correct/incorrect
    predictions_df["correct"] = (predictions_df["y_true"] == predictions_df["y_pred"]).astype(int)

    results = {}
    durations = sorted(predictions_df["duration_ms"].unique())

    print(f"\nDurations to analyze: {durations}")

    for duration_ms in durations:
        print(f"\n{'-'*70}")
        print(f"DURATION: {duration_ms} ms")
        print(f"{'-'*70}")

        # Filter to this duration
        dur_df = predictions_df[predictions_df["duration_ms"] == duration_ms].copy()

        # Extract data
        x_snr = dur_df["snr_db"].values
        y_correct = dur_df["correct"].values
        clip_ids = dur_df["clip_id"].values

        n_samples = len(dur_df)
        n_clips = dur_df["clip_id"].nunique()
        accuracy = y_correct.mean()

        print(f"  Samples: {n_samples} ({n_clips} clips)")
        print(f"  Overall accuracy: {accuracy:.3f}")

        # Check if we have enough variation to fit
        unique_snr = np.unique(x_snr)
        if len(unique_snr) < 3:
            print(f"  [!]  WARNING: Only {len(unique_snr)} SNR levels, skipping fit")
            results[f"dur{int(duration_ms)}ms"] = {
                "converged": False,
                "reason": "insufficient_variation",
                "n_samples": n_samples,
                "n_clips": n_clips,
                "accuracy": float(accuracy),
            }
            continue

        # Fit MLE curve
        print(f"  Fitting MLE psychometric curve...")
        params, fitted_curve = fit_psychometric_mle(
            x_snr,
            y_correct,
            gamma=0.5,
            initial_x50=0.0,  # Start at 0 dB
        )

        if not params["converged"]:
            print(f"  [X] FAIL: Optimization did not converge")
            results[f"dur{int(duration_ms)}ms"] = {
                **params,
                "n_samples": n_samples,
                "n_clips": n_clips,
                "accuracy": float(accuracy),
            }
            continue

        # Check monotonicity
        is_monotonic = params["slope"] > 0
        monotonicity_status = "[OK] PASS" if is_monotonic else "[X] FAIL"

        print(f"\n  Fitted parameters:")
        print(f"    SNR-50: {params['x50']:.2f} dB")
        print(f"    SNR-75: {params['dt75']:.2f} dB (PRIMARY METRIC)")
        print(f"    Slope: {params['slope']:.4f} {monotonicity_status}")
        print(f"    Lapse: {params['lapse']:.4f}")
        print(f"    McFadden R²: {params['mcfadden_r2']:.3f}")
        print(f"    Tjur R²: {params['tjur_r2']:.3f}")

        # Bootstrap CIs
        print(f"\n  Computing bootstrap CIs ({n_bootstrap} samples, clustered by clip)...")
        snr50_mean, snr50_lower, snr50_upper = bootstrap_threshold(
            x_snr, y_correct, clip_ids, n_bootstrap, "x50", gamma=0.5
        )
        snr75_mean, snr75_lower, snr75_upper = bootstrap_threshold(
            x_snr, y_correct, clip_ids, n_bootstrap, "dt75", gamma=0.5
        )

        ci_width_75 = snr75_upper - snr75_lower
        print(f"    SNR-50: {snr50_mean:.2f} dB [CI95: {snr50_lower:.2f}, {snr50_upper:.2f}]")
        print(f"    SNR-75: {snr75_mean:.2f} dB [CI95: {snr75_lower:.2f}, {snr75_upper:.2f}] (width: {ci_width_75:.2f} dB)")

        # Check if threshold is within range
        snr_range = (x_snr.min(), x_snr.max())
        snr75_in_range = snr_range[0] <= params['dt75'] <= snr_range[1]
        range_status = "[OK] IN RANGE" if snr75_in_range else "[!]  OUT OF RANGE"
        print(f"    Range check: SNR in [{snr_range[0]:.0f}, {snr_range[1]:.0f}] dB {range_status}")

        # Store results
        results[f"dur{int(duration_ms)}ms"] = {
            **params,
            "snr50_ci_lower": float(snr50_lower),
            "snr50_ci_upper": float(snr50_upper),
            "snr75_ci_lower": float(snr75_lower),
            "snr75_ci_upper": float(snr75_upper),
            "snr75_ci_width": float(ci_width_75),
            "fitted_curve": fitted_curve.tolist() if fitted_curve is not None else None,
            "is_monotonic": bool(is_monotonic),  # Convert numpy bool to Python bool
            "snr75_in_range": bool(snr75_in_range),  # Convert numpy bool to Python bool
            "snr_range": [float(snr_range[0]), float(snr_range[1])],  # Convert to list of floats
            "n_samples": int(n_samples),
            "n_clips": int(n_clips),
            "accuracy": float(accuracy),
        }

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE: SNR-75 BY DURATION")
    print(f"{'='*70}")
    print(f"{'Duration':<12} {'SNR-75 (dB)':<15} {'CI95':<25} {'R² (McF)':<10} {'Monotonic'}")
    print(f"{'-'*70}")

    for dur_key in sorted(results.keys()):
        res = results[dur_key]
        if res["converged"]:
            dur_ms = dur_key.replace("dur", "").replace("ms", "")
            snr75 = res["dt75"]
            ci_str = f"[{res['snr75_ci_lower']:.1f}, {res['snr75_ci_upper']:.1f}]"
            r2 = res["mcfadden_r2"]
            mono = "[OK]" if res["is_monotonic"] else "[X]"
            print(f"{dur_ms:>4} ms       {snr75:>6.1f}          {ci_str:<25} {r2:<10.3f} {mono}")
        else:
            print(f"{dur_key:<12} {'FAILED':<40}")

    print(f"{'='*70}")

    return results


def plot_stratified_snr_curves(
    results: Dict,
    predictions_df: pd.DataFrame,
    output_path: Path,
):
    """
    Plot 4 stratified SNR curves (one per duration) on the same figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    durations = [20, 80, 200, 1000]
    colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB']  # Red, Orange, Green, Blue

    for idx, (duration_ms, ax, color) in enumerate(zip(durations, axes, colors)):
        dur_key = f"dur{int(duration_ms)}ms"

        # Filter data
        dur_df = predictions_df[predictions_df["duration_ms"] == duration_ms].copy()
        dur_df["correct"] = (dur_df["y_true"] == dur_df["y_pred"]).astype(int)

        # Aggregate empirical proportions
        grouped = dur_df.groupby("snr_db")["correct"].agg(["mean", "count"])

        # Plot empirical data
        ax.scatter(grouped.index, grouped["mean"], s=grouped["count"]*3,
                   alpha=0.6, color=color, label='Empirical', zorder=3, edgecolors='black', linewidths=0.5)

        # Plot fitted curve if available
        if dur_key in results and results[dur_key]["converged"]:
            res = results[dur_key]
            fitted = np.array(res["fitted_curve"])

            ax.plot(fitted[:, 0], fitted[:, 1], '-', color=color, linewidth=2.5, label='MLE Fit', zorder=2)

            # Mark chance and 75% threshold
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.4, linewidth=1, zorder=1)
            ax.axhline(0.75, color='gray', linestyle='--', alpha=0.4, linewidth=1, zorder=1)

            # Mark SNR-75
            snr75 = res["dt75"]
            snr75_lower = res["snr75_ci_lower"]
            snr75_upper = res["snr75_ci_upper"]

            ax.axvline(snr75, color=color, linestyle='--', alpha=0.7, linewidth=2, zorder=1)
            ax.axvspan(snr75_lower, snr75_upper, alpha=0.15, color=color, zorder=0)

            # Annotate SNR-75
            ax.text(snr75, 0.78, f'SNR-75={snr75:.1f}dB\n[{snr75_lower:.1f}, {snr75_upper:.1f}]',
                    ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color))

            # Add R² annotation
            r2 = res["mcfadden_r2"]
            tjur_r2 = res["tjur_r2"]
            ax.text(0.03, 0.97, f"McFadden R²={r2:.3f}\nTjur R²={tjur_r2:.3f}",
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            # Monotonicity status
            mono_emoji = "[OK]" if res["is_monotonic"] else "[X]"
            ax.text(0.97, 0.03, f"Monotonic: {mono_emoji}",
                    transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='lightgreen' if res["is_monotonic"] else 'yellow', alpha=0.5))

        else:
            ax.text(0.5, 0.5, 'FIT FAILED', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='red', weight='bold')

        # Labels and styling
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_ylabel('P(Correct)', fontsize=10)
        ax.set_title(f'Duration: {duration_ms} ms', fontsize=11, fontweight='bold', color=color)
        ax.set_ylim([0.35, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)

    plt.suptitle('Sprint 8: Stratified SNR Psychometric Curves by Duration\n(MLE fit, gamma=0.5 fixed, lambda free - PAPER READY)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved stratified SNR curves: {output_path}")


def compare_with_sprint7(
    sprint8_results: Dict,
    sprint7_path: Path = Path("results/psychometric_curves/psychometric_results.json"),
):
    """
    Compare Sprint 8 stratified results with Sprint 7 overall SNR curve.
    """
    print(f"\n{'='*70}")
    print("COMPARISON: SPRINT 8 (STRATIFIED) vs SPRINT 7 (OVERALL)")
    print(f"{'='*70}")

    if not sprint7_path.exists():
        print(f"[!]  Sprint 7 results not found at {sprint7_path}")
        return

    with open(sprint7_path, 'r') as f:
        sprint7_results = json.load(f)

    sprint7_snr = sprint7_results.get("snr", {}).get("overall", {})

    if not sprint7_snr or not sprint7_snr.get("converged"):
        print("[!]  Sprint 7 SNR overall curve not available")
        return

    print("\nSprint 7 (Overall SNR, collapsed across durations):")
    print(f"  SNR-75: {sprint7_snr['dt75']:.2f} dB")
    print(f"  McFadden R²: {sprint7_snr['mcfadden_r2']:.3f}")
    print(f"  Tjur R²: {sprint7_snr.get('tjur_r2', 'N/A')}")
    print(f"  Status: DIAGNOSTIC ONLY (non-monotonic)")

    print("\nSprint 8 (Stratified by duration):")
    for dur_key in sorted(sprint8_results.keys()):
        res = sprint8_results[dur_key]
        if res["converged"]:
            print(f"  {dur_key}: SNR-75={res['dt75']:.2f} dB, McFadden R²={res['mcfadden_r2']:.3f}, Monotonic={'[OK]' if res['is_monotonic'] else '[X]'}")

    print("\nImprovement:")
    # Average McFadden R² across durations
    valid_r2 = [res["mcfadden_r2"] for res in sprint8_results.values() if res["converged"]]
    if valid_r2:
        avg_r2_sprint8 = np.mean(valid_r2)
        sprint7_r2 = sprint7_snr['mcfadden_r2']
        improvement = avg_r2_sprint8 - sprint7_r2
        print(f"  Average McFadden R² (Sprint 8): {avg_r2_sprint8:.3f}")
        print(f"  Sprint 7 McFadden R²: {sprint7_r2:.3f}")
        print(f"  Improvement: {improvement:+.3f} ({improvement/sprint7_r2*100:+.1f}%)")

        if improvement > 0:
            print(f"  [OK] Sprint 8 stratified curves show improved fit quality")
        else:
            print(f"  [!]  No improvement detected")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Sprint 8: Fit stratified SNR curves by duration")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/sprint8_factorial/predictions.parquet"),
        help="Predictions parquet file from Sprint 8 evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/sprint8_stratified"),
        help="Output directory",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CIs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("="*70)
    print("SPRINT 8: STRATIFIED SNR PSYCHOMETRIC CURVES")
    print("="*70)

    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    pred_df = pd.read_parquet(args.predictions)
    print(f"Loaded {len(pred_df)} predictions from {pred_df['clip_id'].nunique()} clips")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze stratified SNR curves
    results = analyze_snr_stratified_by_duration(pred_df, args.output_dir, args.n_bootstrap)

    # Save results
    output = {
        "stratified_by_duration": results,
        "metadata": {
            "method": "MLE binomial fitting (stratified by duration)",
            "gamma": 0.5,
            "lapse_free": True,
            "lapse_bounds": [0.0, 0.1],
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "n_predictions": len(pred_df),
            "n_clips": pred_df["clip_id"].nunique(),
            "durations_analyzed": sorted(pred_df["duration_ms"].unique()),
            "primary_metric": "SNR-75 (dB) per duration level",
            "reference": "Wichmann & Hill (2001), Perception & Psychophysics",
        }
    }

    results_path = args.output_dir / "snr_stratified_results.json"
    with open(results_path, "w") as f:
        json.dump(convert_to_json_serializable(output), f, indent=2)

    print(f"\nSaved results to: {results_path}")

    # Generate plots
    print("\nGenerating stratified SNR curve plots...")
    plot_stratified_snr_curves(results, pred_df, args.output_dir / "snr_curves_stratified.png")

    # Compare with Sprint 7
    compare_with_sprint7(results)

    print("\n" + "="*70)
    print("SPRINT 8 FITTING COMPLETE")
    print("="*70)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
