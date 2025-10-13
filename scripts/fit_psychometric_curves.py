#!/usr/bin/env python3
"""
Sprint 7: Fit psychometric curves and extract thresholds.

Implements:
1. Logistic curves: P(SPEECH) vs duration_ms and vs snr_db
2. Threshold extraction: DT50/DT75 (duration) and SNR-50 (SNR)
3. Bootstrap confidence intervals (clustered by clip_id)
4. Paper-ready figures and tables

References:
- Wichmann & Hill (2001a): Fitting psychometric functions
- Wichmann & Hill (2001b): Bootstrap-based confidence intervals
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import bootstrap
from tqdm import tqdm

warnings.filterwarnings('ignore', category=RuntimeWarning)


def logistic(x: np.ndarray, x50: float, slope: float, lapse: float = 0.0) -> np.ndarray:
    """
    Logistic psychometric function.

    P(SPEECH) = lapse + (1 - 2*lapse) / (1 + exp(-slope * (x - x50)))

    Args:
        x: Stimulus values (duration_ms or snr_db)
        x50: Threshold (50% point)
        slope: Steepness of curve
        lapse: Lapse rate (0-1, typically small)

    Returns:
        Probability of SPEECH response
    """
    return lapse + (1 - 2 * lapse) / (1 + np.exp(-slope * (x - x50)))


def fit_psychometric_curve(
    x_values: np.ndarray,
    y_binary: np.ndarray,
    x_label: str = "stimulus",
    initial_x50: float = None,
) -> Tuple[Dict, np.ndarray]:
    """
    Fit logistic psychometric curve to binary data.

    Args:
        x_values: Stimulus values (e.g., duration_ms)
        y_binary: Binary responses (1=SPEECH, 0=NONSPEECH)
        x_label: Label for x-axis (for reporting)
        initial_x50: Initial guess for x50 (default: median of x_values)

    Returns:
        (params_dict, fitted_curve)
    """
    # Aggregate to get proportion correct at each x
    unique_x = np.sort(np.unique(x_values))
    proportions = []
    counts = []

    for x in unique_x:
        mask = x_values == x
        p = y_binary[mask].mean()
        n = mask.sum()
        proportions.append(p)
        counts.append(n)

    proportions = np.array(proportions)
    counts = np.array(counts)

    # Initial parameter guesses
    if initial_x50 is None:
        initial_x50 = np.median(unique_x)

    initial_slope = 1.0 / (np.std(unique_x) + 1e-6)
    initial_lapse = 0.01

    try:
        # Fit with weighted least squares (weights = counts)
        popt, pcov = curve_fit(
            logistic,
            unique_x,
            proportions,
            p0=[initial_x50, initial_slope, initial_lapse],
            sigma=1.0 / (np.sqrt(counts) + 1e-6),  # Weight by count
            bounds=([unique_x.min(), 0, 0], [unique_x.max(), np.inf, 0.5]),
            maxfev=10000,
        )

        x50_fit, slope_fit, lapse_fit = popt

        # Compute fitted curve
        x_fine = np.linspace(unique_x.min(), unique_x.max(), 100)
        y_fitted = logistic(x_fine, x50_fit, slope_fit, lapse_fit)

        # Compute thresholds
        dt50 = x50_fit

        # DT75: Find x where P = 0.75
        target_75 = 0.75
        idx_75 = np.argmin(np.abs(y_fitted - target_75))
        dt75 = x_fine[idx_75]

        # Goodness of fit (R²)
        y_pred = logistic(unique_x, x50_fit, slope_fit, lapse_fit)
        ss_res = np.sum((proportions - y_pred) ** 2)
        ss_tot = np.sum((proportions - np.mean(proportions)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-6))

        params = {
            "x50": float(x50_fit),
            "slope": float(slope_fit),
            "lapse": float(lapse_fit),
            "dt75": float(dt75),
            "r_squared": float(r_squared),
            "n_points": len(unique_x),
            "converged": True,
        }

        fitted = np.column_stack([x_fine, y_fitted])

    except Exception as e:
        print(f"Warning: Fit failed for {x_label}: {e}")
        params = {
            "x50": np.nan,
            "slope": np.nan,
            "lapse": np.nan,
            "dt75": np.nan,
            "r_squared": np.nan,
            "n_points": len(unique_x),
            "converged": False,
        }
        fitted = None

    return params, fitted


def bootstrap_threshold(
    x_values: np.ndarray,
    y_binary: np.ndarray,
    clip_ids: np.ndarray,
    n_bootstrap: int = 1000,
    threshold_type: str = "x50",
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for threshold (clustered by clip_id).

    Args:
        x_values: Stimulus values
        y_binary: Binary responses
        clip_ids: Clip IDs for clustering
        n_bootstrap: Number of bootstrap samples
        threshold_type: "x50" or "dt75"
        random_state: Random seed

    Returns:
        (mean, ci_lower, ci_upper)
    """
    # Get unique clips
    unique_clips = np.unique(clip_ids)
    n_clips = len(unique_clips)

    thresholds = []
    np.random.seed(random_state)

    for _ in range(n_bootstrap):
        # Resample clips with replacement
        sampled_clips = np.random.choice(unique_clips, size=n_clips, replace=True)

        # Get all data for sampled clips
        mask = np.isin(clip_ids, sampled_clips)
        x_boot = x_values[mask]
        y_boot = y_binary[mask]

        if len(x_boot) < 3:  # Need at least 3 points
            continue

        # Fit curve
        params, _ = fit_psychometric_curve(x_boot, y_boot)

        if params["converged"]:
            if threshold_type == "x50":
                thresholds.append(params["x50"])
            elif threshold_type == "dt75":
                thresholds.append(params["dt75"])

    if len(thresholds) < 10:
        return np.nan, np.nan, np.nan

    thresholds = np.array(thresholds)
    mean = np.mean(thresholds)
    ci_lower = np.percentile(thresholds, 2.5)
    ci_upper = np.percentile(thresholds, 97.5)

    return mean, ci_lower, ci_upper


def analyze_duration_curves(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Analyze P(SPEECH) vs duration for each SNR level.

    Returns dict with thresholds and fitted curves.
    """
    print("\n" + "="*60)
    print("PSYCHOMETRIC CURVES: P(SPEECH) vs DURATION")
    print("="*60)

    # Filter to duration variants
    duration_df = predictions_df[predictions_df["variant_type"] == "duration"].copy()

    # Convert to binary (1=SPEECH, 0=NONSPEECH)
    duration_df["y_binary"] = (duration_df["y_true"] == "SPEECH").astype(int)
    duration_df["correct"] = (duration_df["y_true"] == duration_df["y_pred"]).astype(int)

    results = {}

    # Overall (all durations)
    print("\nOVERALL (all conditions):")
    x_all = duration_df["duration_ms"].values
    y_all = duration_df["correct"].values
    clip_ids_all = duration_df["clip_id"].values

    params_all, fitted_all = fit_psychometric_curve(x_all, y_all, "duration_ms")

    if params_all["converged"]:
        print(f"  DT50: {params_all['x50']:.1f} ms")
        print(f"  DT75: {params_all['dt75']:.1f} ms")
        print(f"  R²: {params_all['r_squared']:.3f}")

        # Bootstrap CI
        print("  Computing bootstrap CI (1000 samples, clustered by clip)...")
        dt50_mean, dt50_lower, dt50_upper = bootstrap_threshold(
            x_all, y_all, clip_ids_all, n_bootstrap, "x50"
        )
        dt75_mean, dt75_lower, dt75_upper = bootstrap_threshold(
            x_all, y_all, clip_ids_all, n_bootstrap, "dt75"
        )

        print(f"  DT50 CI95: [{dt50_lower:.1f}, {dt50_upper:.1f}]")
        print(f"  DT75 CI95: [{dt75_lower:.1f}, {dt75_upper:.1f}]")

        results["overall"] = {
            **params_all,
            "dt50_ci_lower": float(dt50_lower),
            "dt50_ci_upper": float(dt50_upper),
            "dt75_ci_lower": float(dt75_lower),
            "dt75_ci_upper": float(dt75_upper),
            "fitted_curve": fitted_all.tolist() if fitted_all is not None else None,
        }

    return results


def analyze_snr_curves(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    Analyze P(SPEECH) vs SNR for each duration.

    Returns dict with SNR-50 thresholds and fitted curves.
    """
    print("\n" + "="*60)
    print("PSYCHOMETRIC CURVES: P(SPEECH) vs SNR")
    print("="*60)

    # Filter to SNR variants and remove NaN
    snr_df = predictions_df[predictions_df["variant_type"] == "snr"].copy()
    snr_df = snr_df[snr_df["snr_db"].notna()]  # Remove NaN

    # Convert to binary
    snr_df["y_binary"] = (snr_df["y_true"] == "SPEECH").astype(int)
    snr_df["correct"] = (snr_df["y_true"] == snr_df["y_pred"]).astype(int)

    results = {}

    # Overall (all SNR levels)
    print("\nOVERALL (all conditions):")
    x_all = snr_df["snr_db"].values
    y_all = snr_df["correct"].values
    clip_ids_all = snr_df["clip_id"].values

    params_all, fitted_all = fit_psychometric_curve(x_all, y_all, "snr_db", initial_x50=5.0)

    if params_all["converged"]:
        print(f"  SNR-50: {params_all['x50']:.1f} dB")
        print(f"  SNR-75: {params_all['dt75']:.1f} dB")
        print(f"  R²: {params_all['r_squared']:.3f}")

        # Bootstrap CI
        print("  Computing bootstrap CI (1000 samples, clustered by clip)...")
        snr50_mean, snr50_lower, snr50_upper = bootstrap_threshold(
            x_all, y_all, clip_ids_all, n_bootstrap, "x50"
        )
        snr75_mean, snr75_lower, snr75_upper = bootstrap_threshold(
            x_all, y_all, clip_ids_all, n_bootstrap, "dt75"
        )

        print(f"  SNR-50 CI95: [{snr50_lower:.1f}, {snr50_upper:.1f}]")
        print(f"  SNR-75 CI95: [{snr75_lower:.1f}, {snr75_upper:.1f}]")

        results["overall"] = {
            **params_all,
            "snr50_ci_lower": float(snr50_lower),
            "snr50_ci_upper": float(snr50_upper),
            "snr75_ci_lower": float(snr75_lower),
            "snr75_ci_upper": float(snr75_upper),
            "fitted_curve": fitted_all.tolist() if fitted_all is not None else None,
        }

    return results


def plot_duration_curves(
    duration_results: Dict,
    predictions_df: pd.DataFrame,
    output_path: Path,
):
    """Generate duration psychometric curve plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data
    duration_df = predictions_df[predictions_df["variant_type"] == "duration"].copy()
    duration_df["correct"] = (duration_df["y_true"] == duration_df["y_pred"]).astype(int)

    # Aggregate proportions
    grouped = duration_df.groupby("duration_ms")["correct"].agg(["mean", "count"])

    # Plot empirical data
    ax.scatter(grouped.index, grouped["mean"], s=grouped["count"]*2,
               alpha=0.6, color='steelblue', label='Empirical', zorder=3)

    # Plot fitted curve
    if "overall" in duration_results and duration_results["overall"]["fitted_curve"]:
        fitted = np.array(duration_results["overall"]["fitted_curve"])
        ax.plot(fitted[:, 0], fitted[:, 1], 'r-', linewidth=2, label='Fitted', zorder=2)

        # Mark DT50 and DT75
        dt50 = duration_results["overall"]["x50"]
        dt75 = duration_results["overall"]["dt75"]

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.axhline(0.75, color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.axvline(dt50, color='red', linestyle='--', alpha=0.5, zorder=1)
        ax.axvline(dt75, color='darkred', linestyle='--', alpha=0.5, zorder=1)

        # Annotate
        ax.text(dt50, 0.52, f'DT50={dt50:.0f}ms', ha='center', fontsize=10)
        ax.text(dt75, 0.77, f'DT75={dt75:.0f}ms', ha='center', fontsize=10)

    ax.set_xlabel('Duration (ms)', fontsize=12)
    ax.set_ylabel('P(Correct)', fontsize=12)
    ax.set_title('Psychometric Curve: P(Correct) vs Duration', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved duration curve plot: {output_path}")


def plot_snr_curves(
    snr_results: Dict,
    predictions_df: pd.DataFrame,
    output_path: Path,
):
    """Generate SNR psychometric curve plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get data
    snr_df = predictions_df[predictions_df["variant_type"] == "snr"].copy()
    snr_df["correct"] = (snr_df["y_true"] == snr_df["y_pred"]).astype(int)

    # Aggregate proportions
    grouped = snr_df.groupby("snr_db")["correct"].agg(["mean", "count"])

    # Plot empirical data
    ax.scatter(grouped.index, grouped["mean"], s=grouped["count"]*2,
               alpha=0.6, color='steelblue', label='Empirical', zorder=3)

    # Plot fitted curve
    if "overall" in snr_results and snr_results["overall"]["fitted_curve"]:
        fitted = np.array(snr_results["overall"]["fitted_curve"])
        ax.plot(fitted[:, 0], fitted[:, 1], 'r-', linewidth=2, label='Fitted', zorder=2)

        # Mark SNR-50
        snr50 = snr_results["overall"]["x50"]

        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.axvline(snr50, color='red', linestyle='--', alpha=0.5, zorder=1)

        # Annotate
        ax.text(snr50, 0.52, f'SNR-50={snr50:.1f}dB', ha='center', fontsize=10)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('P(Correct)', fontsize=12)
    ax.set_title('Psychometric Curve: P(Correct) vs SNR', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved SNR curve plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit psychometric curves and extract thresholds")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/sprint6_robust/dev_predictions.parquet"),
        help="Predictions parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/psychometric_curves"),
        help="Output directory",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    print("="*60)
    print("SPRINT 7: PSYCHOMETRIC CURVES AND THRESHOLDS")
    print("="*60)

    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    pred_df = pd.read_parquet(args.predictions)
    print(f"Loaded {len(pred_df)} predictions")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze duration curves
    duration_results = analyze_duration_curves(pred_df, args.output_dir, args.n_bootstrap)

    # Analyze SNR curves
    snr_results = analyze_snr_curves(pred_df, args.output_dir, args.n_bootstrap)

    # Save results
    all_results = {
        "duration": duration_results,
        "snr": snr_results,
        "metadata": {
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "n_predictions": len(pred_df),
            "n_clips": pred_df["clip_id"].nunique(),
        }
    }

    results_path = args.output_dir / "psychometric_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {results_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_duration_curves(duration_results, pred_df, args.output_dir / "duration_curve.png")
    plot_snr_curves(snr_results, pred_df, args.output_dir / "snr_curve.png")

    print("\n" + "="*60)
    print("SPRINT 7 COMPLETE")
    print("="*60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
