#!/usr/bin/env python3
"""
Compute Psychometric Curves with Bootstrap CIs

Analyzes model performance across duration and SNR to compute:
- Duration curve: Accuracy vs Duration (DT50, DT75, DT90)
- SNR curve: Accuracy vs SNR (SNR-75)
- Stratified SNR curves at fixed durations (20, 80, 200, 1000ms)

Bootstrap confidence intervals for all threshold estimates.

Usage:
    python scripts/compute_psychometric_curves.py \
        --input_csvs results/test_final/test_seed*.csv \
        --output_dir results/psychometric_curves \
        --bootstrap_iterations 1000
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


def compute_accuracy_by_condition(df, group_col):
    """Compute accuracy grouped by condition"""
    results = []
    for val, group in df.groupby(group_col):
        acc = (group['prediction'] == group['ground_truth']).mean() * 100
        results.append({'value': val, 'accuracy': acc, 'n': len(group)})
    return pd.DataFrame(results).sort_values('value')


def bootstrap_threshold(df, group_col, target_accuracy, n_iter=1000, percentiles=[50, 75, 90]):
    """Bootstrap confidence interval for threshold (e.g., DT75)"""
    thresholds = []

    for _ in range(n_iter):
        # Resample with replacement
        sample = df.sample(frac=1, replace=True)
        acc_df = compute_accuracy_by_condition(sample, group_col)

        # Interpolate to find threshold
        if len(acc_df) > 1 and acc_df['accuracy'].min() < target_accuracy < acc_df['accuracy'].max():
            f = interp1d(acc_df['accuracy'], acc_df['value'], fill_value='extrapolate')
            thresh = float(f(target_accuracy))
            thresholds.append(thresh)

    if len(thresholds) == 0:
        return None, None, None

    return np.median(thresholds), np.percentile(thresholds, 2.5), np.percentile(thresholds, 97.5)


def plot_duration_curve(df, output_dir, bootstrap_iters):
    """Plot and analyze duration curve"""
    print("\n=== DURATION ANALYSIS ===")

    # Aggregate over SNR
    acc_by_dur = compute_accuracy_by_condition(df, 'duration_ms')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(acc_by_dur['value'], acc_by_dur['accuracy'], 'o-', linewidth=2, markersize=8)
    ax.axhline(75, color='r', linestyle='--', alpha=0.5, label='75% target')
    ax.set_xlabel('Duration (ms)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Psychometric Curve: Duration', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'duration_curve.png', dpi=300)
    plt.close()

    # Compute thresholds
    print("\nDuration Thresholds (with 95% CI):")
    for target_acc, name in [(50, 'DT50'), (75, 'DT75'), (90, 'DT90')]:
        median, ci_low, ci_high = bootstrap_threshold(df, 'duration_ms', target_acc, bootstrap_iters)
        if median:
            print(f"  {name}: {median:.1f}ms [{ci_low:.1f}, {ci_high:.1f}]")

    # Per-duration accuracy
    print("\nAccuracy by Duration:")
    for _, row in acc_by_dur.iterrows():
        print(f"  {int(row['value']):4d}ms: {row['accuracy']:.1f}% (n={int(row['n'])})")

    return acc_by_dur


def plot_snr_curve(df, output_dir, bootstrap_iters, fixed_duration=1000):
    """Plot and analyze SNR curve at fixed duration"""
    print(f"\n=== SNR ANALYSIS (at {fixed_duration}ms) ===")

    # Filter to fixed duration
    df_fixed = df[df['duration_ms'] == fixed_duration].copy()

    if len(df_fixed) == 0:
        print(f"  No data at {fixed_duration}ms")
        return None

    # Aggregate
    acc_by_snr = compute_accuracy_by_condition(df_fixed, 'snr_db')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(acc_by_snr['value'], acc_by_snr['accuracy'], 'o-', linewidth=2, markersize=8, color='green')
    ax.axhline(75, color='r', linestyle='--', alpha=0.5, label='75% target')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Psychometric Curve: SNR (at {fixed_duration}ms)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'snr_curve_{fixed_duration}ms.png', dpi=300)
    plt.close()

    # Compute SNR-75
    median, ci_low, ci_high = bootstrap_threshold(df_fixed, 'snr_db', 75, bootstrap_iters)
    if median:
        print(f"\n  SNR-75: {median:.1f}dB [{ci_low:.1f}, {ci_high:.1f}]")

    print(f"\nAccuracy by SNR (at {fixed_duration}ms):")
    for _, row in acc_by_snr.iterrows():
        print(f"  {int(row['value']):+3d}dB: {row['accuracy']:.1f}% (n={int(row['n'])})")

    return acc_by_snr


def plot_stratified_snr_curves(df, output_dir, durations=[20, 80, 200, 1000]):
    """Plot SNR curves stratified by duration"""
    print("\n=== STRATIFIED SNR CURVES ===")

    stratified_dir = output_dir / 'snr_stratified'
    stratified_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, dur in enumerate(durations):
        df_dur = df[df['duration_ms'] == dur].copy()

        if len(df_dur) == 0:
            continue

        acc_by_snr = compute_accuracy_by_condition(df_dur, 'snr_db')

        ax = axes[idx]
        ax.plot(acc_by_snr['value'], acc_by_snr['accuracy'], 'o-', linewidth=2, markersize=6)
        ax.axhline(75, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Duration = {dur}ms')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(stratified_dir / 'snr_curves_stratified.png', dpi=300)
    plt.close()

    print(f"  ✓ Saved stratified curves to {stratified_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csvs', nargs='+', required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--bootstrap_iterations', type=int, default=1000)
    parser.add_argument('--stratified_durations', type=int, nargs='+', default=[20, 80, 200, 1000])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PSYCHOMETRIC CURVES ANALYSIS")
    print("=" * 80)

    # Load and aggregate data
    dfs = [pd.read_csv(f) for f in args.input_csvs]
    df = pd.concat(dfs, ignore_index=True)

    print(f"\nLoaded {len(args.input_csvs)} files, {len(df)} total predictions")

    # Duration curve
    plot_duration_curve(df, args.output_dir, args.bootstrap_iterations)

    # SNR curve (at 1000ms)
    plot_snr_curve(df, args.output_dir, args.bootstrap_iterations, fixed_duration=1000)

    # Stratified SNR curves
    plot_stratified_snr_curves(df, args.output_dir, args.stratified_durations)

    print(f"\n{'=' * 80}")
    print("✓ PSYCHOMETRIC ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
