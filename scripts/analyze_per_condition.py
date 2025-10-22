#!/usr/bin/env python3
"""
Per-Condition Analysis Script

Analyzes model performance broken down by:
1. Duration (200ms vs 1000ms)
2. SNR level (0dB, 5dB, 10dB, 20dB)
3. Class (SPEECH vs NONSPEECH)

Usage:
    python scripts/analyze_per_condition.py \\
        --predictions results/ablations/LORA_attn_mlp_seed42.csv \\
        --output_dir results/per_condition_analysis

Generates:
    - CSV table with per-condition metrics
    - Plots showing performance across conditions
    - Statistical summary
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def extract_duration_from_clip_id(clip_id: str) -> str:
    """Extract duration (200ms or 1000ms) from clip_id."""
    if '200ms' in clip_id:
        return '200ms'
    elif '1000ms' in clip_id:
        return '1000ms'
    else:
        return 'unknown'


def extract_snr_from_audio_path(audio_path: str) -> str:
    """Extract SNR level from audio path."""
    # Handle both formats: _snrXX_ and _snr+XXdB
    if '_snr+0dB' in audio_path or '_snr0_' in audio_path or '_snr0.' in audio_path:
        return '0dB'
    elif '_snr+5dB' in audio_path or '_snr5_' in audio_path or '_snr5.' in audio_path:
        return '5dB'
    elif '_snr+10dB' in audio_path or '_snr10_' in audio_path or '_snr10.' in audio_path:
        return '10dB'
    elif '_snr+20dB' in audio_path or '_snr20_' in audio_path or '_snr20.' in audio_path:
        return '20dB'
    else:
        return 'unknown'


def compute_metrics_by_condition(df: pd.DataFrame, group_by: list) -> pd.DataFrame:
    """
    Compute accuracy metrics grouped by specified conditions.

    Args:
        df: DataFrame with 'ground_truth', 'correct' columns
        group_by: List of column names to group by

    Returns:
        DataFrame with metrics per condition
    """
    results = []

    for group_vals, group_df in df.groupby(group_by):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)

        total = len(group_df)
        correct = group_df['correct'].sum()
        accuracy = correct / total * 100 if total > 0 else 0

        # Confidence interval (Wilson score interval for small samples)
        if total > 0:
            p = correct / total
            z = 1.96  # 95% CI
            denominator = 1 + z**2 / total
            center = (p + z**2 / (2 * total)) / denominator
            margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
            ci_lower = max(0, center - margin) * 100
            ci_upper = min(1, center + margin) * 100
        else:
            ci_lower = ci_upper = 0

        result = {col: val for col, val in zip(group_by, group_vals)}
        result.update({
            'n_samples': total,
            'n_correct': int(correct),
            'accuracy': accuracy,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        })

        results.append(result)

    return pd.DataFrame(results)


def plot_duration_comparison(df: pd.DataFrame, output_path: Path):
    """Plot accuracy comparison between 200ms and 1000ms."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by duration and class
    grouped = df.groupby(['duration', 'ground_truth'], as_index=False).agg(
        accuracy=('correct', lambda x: x.mean() * 100),
        n=('correct', 'count')
    )

    # Plot grouped bar chart
    x = np.arange(len(grouped['ground_truth'].unique()))
    width = 0.35

    durations = grouped['duration'].unique()
    for i, duration in enumerate(durations):
        data = grouped[grouped['duration'] == duration]
        offset = width * (i - len(durations) / 2 + 0.5)
        bars = ax.bar(x + offset, data['accuracy'], width,
                      label=duration, alpha=0.8)

        # Add sample counts
        for j, (idx, row) in enumerate(data.iterrows()):
            ax.text(x[j] + offset, row['accuracy'] + 2,
                   f"n={int(row['n'])}",
                   ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy by Duration and Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(grouped['ground_truth'].unique())
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Duration comparison plot saved to: {output_path}")


def plot_snr_comparison(df: pd.DataFrame, output_path: Path):
    """Plot accuracy vs SNR level."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overall accuracy vs SNR
    snr_order = ['0dB', '5dB', '10dB', '20dB']
    snr_metrics = compute_metrics_by_condition(df, ['snr'])
    snr_metrics = snr_metrics[snr_metrics['snr'].isin(snr_order)]
    snr_metrics['snr'] = pd.Categorical(snr_metrics['snr'], categories=snr_order, ordered=True)
    snr_metrics = snr_metrics.sort_values('snr')

    ax1.plot(snr_metrics['snr'], snr_metrics['accuracy'],
            marker='o', linewidth=2, markersize=8)
    ax1.fill_between(range(len(snr_metrics)),
                     snr_metrics['ci_lower'],
                     snr_metrics['ci_upper'],
                     alpha=0.3)
    ax1.set_xlabel('SNR Level', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Overall Accuracy vs SNR', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.grid(alpha=0.3)

    # Add sample counts
    for i, row in snr_metrics.iterrows():
        ax1.text(row['snr'], row['accuracy'] + 3,
                f"n={int(row['n_samples'])}",
                ha='center', fontsize=8)

    # Per-class accuracy vs SNR
    class_snr_metrics = compute_metrics_by_condition(df, ['snr', 'ground_truth'])
    class_snr_metrics = class_snr_metrics[class_snr_metrics['snr'].isin(snr_order)]

    for class_name in ['SPEECH', 'NONSPEECH']:
        data = class_snr_metrics[class_snr_metrics['ground_truth'] == class_name]
        data['snr'] = pd.Categorical(data['snr'], categories=snr_order, ordered=True)
        data = data.sort_values('snr')

        ax2.plot(data['snr'], data['accuracy'],
                marker='o', linewidth=2, markersize=8, label=class_name)

    ax2.set_xlabel('SNR Level', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Per-Class Accuracy vs SNR', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SNR comparison plot saved to: {output_path}")


def plot_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot heatmap of accuracy by duration × SNR."""
    # Compute metrics by duration and SNR
    metrics = compute_metrics_by_condition(df, ['duration', 'snr'])

    # Pivot to matrix form
    snr_order = ['0dB', '5dB', '10dB', '20dB']
    duration_order = ['200ms', '1000ms']

    pivot = metrics.pivot(index='duration', columns='snr', values='accuracy')
    pivot = pivot.reindex(index=duration_order, columns=snr_order)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'},
                linewidths=1, linecolor='black', ax=ax)

    ax.set_xlabel('SNR Level', fontsize=12)
    ax.set_ylabel('Duration', fontsize=12)
    ax.set_title('Accuracy Heatmap: Duration × SNR', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Per-condition analysis of model predictions')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV (must have clip_id, audio_path, ground_truth, correct)')
    parser.add_argument('--output_dir', type=str, default='results/per_condition_analysis',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    print(f"Loading predictions from: {args.predictions}")
    df = pd.read_csv(args.predictions)

    # Extract conditions
    print("Extracting condition metadata...")
    df['duration'] = df['clip_id'].apply(extract_duration_from_clip_id)
    df['snr'] = df['audio_path'].apply(extract_snr_from_audio_path)

    # Filter out unknown conditions
    df = df[(df['duration'] != 'unknown') & (df['snr'] != 'unknown')]

    print(f"Total samples: {len(df)}")
    print(f"  - Durations: {df['duration'].value_counts().to_dict()}")
    print(f"  - SNR levels: {df['snr'].value_counts().to_dict()}")
    print(f"  - Classes: {df['ground_truth'].value_counts().to_dict()}")

    # Compute per-condition metrics
    print("\nComputing per-condition metrics...")

    # 1. By duration
    duration_metrics = compute_metrics_by_condition(df, ['duration'])
    print("\nBy Duration:")
    print(duration_metrics.to_string(index=False))

    # 2. By SNR
    snr_metrics = compute_metrics_by_condition(df, ['snr'])
    print("\nBy SNR:")
    print(snr_metrics.to_string(index=False))

    # 3. By class
    class_metrics = compute_metrics_by_condition(df, ['ground_truth'])
    print("\nBy Class:")
    print(class_metrics.to_string(index=False))

    # 4. By duration × class
    dur_class_metrics = compute_metrics_by_condition(df, ['duration', 'ground_truth'])
    print("\nBy Duration × Class:")
    print(dur_class_metrics.to_string(index=False))

    # 5. By SNR × class
    snr_class_metrics = compute_metrics_by_condition(df, ['snr', 'ground_truth'])
    print("\nBy SNR × Class:")
    print(snr_class_metrics.to_string(index=False))

    # 6. By duration × SNR (overall)
    dur_snr_metrics = compute_metrics_by_condition(df, ['duration', 'snr'])
    print("\nBy Duration × SNR:")
    print(dur_snr_metrics.to_string(index=False))

    # 7. Full breakdown: duration × SNR × class
    full_metrics = compute_metrics_by_condition(df, ['duration', 'snr', 'ground_truth'])
    print("\nFull Breakdown (Duration × SNR × Class):")
    print(full_metrics.to_string(index=False))

    # Save all metrics to CSV
    output_csv = output_dir / 'per_condition_metrics.csv'
    full_metrics.to_csv(output_csv, index=False)
    print(f"\nFull metrics saved to: {output_csv}")

    # Generate plots
    print("\nGenerating plots...")
    plot_duration_comparison(df, output_dir / 'duration_comparison.png')
    plot_snr_comparison(df, output_dir / 'snr_comparison.png')
    plot_heatmap(df, output_dir / 'duration_snr_heatmap.png')

    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)

    # Test 1: Duration effect (200ms vs 1000ms)
    df_200ms = df[df['duration'] == '200ms']
    df_1000ms = df[df['duration'] == '1000ms']

    if len(df_200ms) > 0 and len(df_1000ms) > 0:
        acc_200ms = df_200ms['correct'].mean()
        acc_1000ms = df_1000ms['correct'].mean()

        # Chi-square test
        contingency = pd.crosstab(df['duration'], df['correct'])
        chi2, p_duration, dof, expected = stats.chi2_contingency(contingency)

        print(f"\nDuration Effect (200ms vs 1000ms):")
        print(f"  200ms accuracy:  {acc_200ms*100:.1f}% (n={len(df_200ms)})")
        print(f"  1000ms accuracy: {acc_1000ms*100:.1f}% (n={len(df_1000ms)})")
        print(f"  Chi-squared: {chi2:.4f}, p-value: {p_duration:.4f}")
        if p_duration < 0.05:
            print(f"  SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  NOT significant (p >= 0.05)")

    # Test 2: SNR effect (trend test)
    snr_order_num = {'0dB': 0, '5dB': 5, '10dB': 10, '20dB': 20}
    df['snr_numeric'] = df['snr'].map(snr_order_num)

    if df['snr_numeric'].notna().sum() > 0:
        # Spearman correlation (ordinal SNR vs accuracy)
        correlation, p_snr = stats.spearmanr(df['snr_numeric'], df['correct'])

        print(f"\nSNR Effect (trend across 0/5/10/20 dB):")
        print(f"  Spearman correlation: {correlation:.3f}, p-value: {p_snr:.4f}")
        if p_snr < 0.05:
            direction = "positive" if correlation > 0 else "negative"
            print(f"  SIGNIFICANT {direction} trend (p < 0.05)")
        else:
            print(f"  NOT significant (p >= 0.05)")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
