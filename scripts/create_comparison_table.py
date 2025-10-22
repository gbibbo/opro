#!/usr/bin/env python3
"""
Create comprehensive comparison table across all methods.

Usage:
    python scripts/create_comparison_table.py \
        --prediction_csvs results/*.csv \
        --output_table results/final_comparison_table.md \
        --output_plot results/final_comparison_plot.png
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats


def compute_wilson_ci(successes, n, confidence=0.95):
    """Compute Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n

    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator

    return max(0, center - margin), min(1, center + margin)


def load_predictions(csv_path):
    """Load prediction CSV and standardize format."""
    df = pd.read_csv(csv_path)

    # Detect format and standardize
    if 'correct' in df.columns:
        # Format 1: Has 'correct' column
        return df
    elif 'y_true' in df.columns and 'y_pred' in df.columns:
        # Format 2: Has y_true, y_pred
        df['correct'] = (df['y_true'] == df['y_pred'])
        df['ground_truth'] = df['label']
        return df
    else:
        raise ValueError(f"Unknown CSV format in {csv_path}")


def compute_metrics(df):
    """Compute all metrics for a prediction set."""
    total = len(df)
    correct = df['correct'].sum()
    accuracy = correct / total if total > 0 else 0

    # Per-class metrics
    speech_df = df[df['ground_truth'] == 'SPEECH']
    nonspeech_df = df[df['ground_truth'] == 'NONSPEECH']

    speech_acc = speech_df['correct'].mean() if len(speech_df) > 0 else 0
    nonspeech_acc = nonspeech_df['correct'].mean() if len(nonspeech_df) > 0 else 0

    # Confidence intervals
    ci_low, ci_high = compute_wilson_ci(correct, total)
    speech_ci_low, speech_ci_high = compute_wilson_ci(
        speech_df['correct'].sum(), len(speech_df)
    ) if len(speech_df) > 0 else (0, 0)
    nonspeech_ci_low, nonspeech_ci_high = compute_wilson_ci(
        nonspeech_df['correct'].sum(), len(nonspeech_df)
    ) if len(nonspeech_df) > 0 else (0, 0)

    # Confidence statistics (if available)
    conf_mean = df['confidence'].mean() if 'confidence' in df.columns else None

    # ROC-AUC (if logit_diff available)
    roc_auc = None
    if 'logit_diff' in df.columns and 'ground_truth' in df.columns:
        from sklearn.metrics import roc_auc_score
        y_true = (df['ground_truth'] == 'SPEECH').astype(int)
        y_score = df['logit_diff']
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except:
            roc_auc = None

    return {
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'speech_samples': len(speech_df),
        'speech_correct': speech_df['correct'].sum() if len(speech_df) > 0 else 0,
        'speech_acc': speech_acc,
        'speech_ci_low': speech_ci_low,
        'speech_ci_high': speech_ci_high,
        'nonspeech_samples': len(nonspeech_df),
        'nonspeech_correct': nonspeech_df['correct'].sum() if len(nonspeech_df) > 0 else 0,
        'nonspeech_acc': nonspeech_acc,
        'nonspeech_ci_low': nonspeech_ci_low,
        'nonspeech_ci_high': nonspeech_ci_high,
        'confidence_mean': conf_mean,
        'roc_auc': roc_auc
    }


def create_markdown_table(results, method_names):
    """Create markdown comparison table."""
    lines = []
    lines.append("# Model Comparison Results\n")
    lines.append("## Overall Performance\n")

    # Header
    lines.append("| Method | Overall | SPEECH | NONSPEECH | ROC-AUC | Samples |")
    lines.append("|--------|---------|--------|-----------|---------|---------|")

    # Sort by overall accuracy
    sorted_indices = sorted(range(len(results)), key=lambda i: results[i]['accuracy'], reverse=True)

    for idx in sorted_indices:
        metrics = results[idx]
        name = method_names[idx]

        overall = f"{metrics['accuracy']:.1%} [{metrics['ci_low']:.1%}, {metrics['ci_high']:.1%}]"
        speech = f"{metrics['speech_acc']:.1%}"
        nonspeech = f"{metrics['nonspeech_acc']:.1%}"
        roc = f"{metrics['roc_auc']:.3f}" if metrics['roc_auc'] is not None else "N/A"
        samples = f"{metrics['total_samples']}"

        lines.append(f"| {name} | {overall} | {speech} | {nonspeech} | {roc} | {samples} |")

    # Detailed breakdown
    lines.append("\n## Detailed Breakdown\n")

    for idx in sorted_indices:
        metrics = results[idx]
        name = method_names[idx]

        lines.append(f"\n### {name}\n")
        lines.append(f"- **Overall**: {metrics['correct']}/{metrics['total_samples']} = {metrics['accuracy']:.1%}")
        lines.append(f"  - 95% CI: [{metrics['ci_low']:.1%}, {metrics['ci_high']:.1%}]")
        lines.append(f"- **SPEECH**: {metrics['speech_correct']}/{metrics['speech_samples']} = {metrics['speech_acc']:.1%}")
        lines.append(f"  - 95% CI: [{metrics['speech_ci_low']:.1%}, {metrics['speech_ci_high']:.1%}]")
        lines.append(f"- **NONSPEECH**: {metrics['nonspeech_correct']}/{metrics['nonspeech_samples']} = {metrics['nonspeech_acc']:.1%}")
        lines.append(f"  - 95% CI: [{metrics['nonspeech_ci_low']:.1%}, {metrics['nonspeech_ci_high']:.1%}]")

        if metrics['confidence_mean'] is not None:
            lines.append(f"- **Mean Confidence**: {metrics['confidence_mean']:.3f}")

        if metrics['roc_auc'] is not None:
            lines.append(f"- **ROC-AUC**: {metrics['roc_auc']:.4f}")

    return "\n".join(lines)


def create_comparison_plot(results, method_names, output_path):
    """Create bar plot comparing methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Sort by overall accuracy
    sorted_indices = sorted(range(len(results)), key=lambda i: results[i]['accuracy'], reverse=True)
    sorted_names = [method_names[i] for i in sorted_indices]
    sorted_results = [results[i] for i in sorted_indices]

    x = np.arange(len(sorted_names))
    width = 0.6

    # Plot 1: Overall accuracy
    overall_acc = [r['accuracy'] * 100 for r in sorted_results]
    overall_ci_low = [(r['accuracy'] - r['ci_low']) * 100 for r in sorted_results]
    overall_ci_high = [(r['ci_high'] - r['accuracy']) * 100 for r in sorted_results]

    axes[0].bar(x, overall_acc, width, color='steelblue')
    axes[0].errorbar(x, overall_acc, yerr=[overall_ci_low, overall_ci_high],
                     fmt='none', color='black', capsize=5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Overall Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sorted_names, rotation=45, ha='right')
    axes[0].set_ylim(0, 105)
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Per-class accuracy
    speech_acc = [r['speech_acc'] * 100 for r in sorted_results]
    nonspeech_acc = [r['nonspeech_acc'] * 100 for r in sorted_results]

    x_offset = width / 2
    axes[1].bar(x - x_offset / 2, speech_acc, width / 2, label='SPEECH', color='coral')
    axes[1].bar(x + x_offset / 2, nonspeech_acc, width / 2, label='NONSPEECH', color='lightgreen')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sorted_names, rotation=45, ha='right')
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    # Plot 3: ROC-AUC (if available)
    roc_aucs = [r['roc_auc'] if r['roc_auc'] is not None else 0 for r in sorted_results]
    has_roc = [r['roc_auc'] is not None for r in sorted_results]

    colors = ['steelblue' if h else 'lightgray' for h in has_roc]
    axes[2].bar(x, roc_aucs, width, color=colors)
    axes[2].set_ylabel('ROC-AUC', fontsize=12)
    axes[2].set_title('ROC-AUC Score', fontsize=14, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sorted_names, rotation=45, ha='right')
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create comprehensive comparison table")
    parser.add_argument('--prediction_csvs', nargs='+', required=True,
                        help='List of prediction CSV files')
    parser.add_argument('--method_names', nargs='+', required=True,
                        help='Names for each method (same order as CSVs)')
    parser.add_argument('--output_table', type=str, required=True,
                        help='Output markdown table path')
    parser.add_argument('--output_plot', type=str, default=None,
                        help='Output plot path (optional)')

    args = parser.parse_args()

    if len(args.prediction_csvs) != len(args.method_names):
        print("ERROR: Number of CSVs must match number of method names")
        return

    print("="*80)
    print("MODEL COMPARISON TABLE GENERATOR")
    print("="*80)

    # Load all results
    results = []
    for csv_path, name in zip(args.prediction_csvs, args.method_names):
        print(f"\nLoading: {name}")
        print(f"  File: {csv_path}")

        try:
            df = load_predictions(csv_path)
            metrics = compute_metrics(df)
            results.append(metrics)
            print(f"  Accuracy: {metrics['accuracy']:.1%}")
        except Exception as e:
            print(f"  ERROR: {e}")
            return

    # Create markdown table
    print(f"\nGenerating comparison table...")
    markdown = create_markdown_table(results, args.method_names)

    # Save
    output_path = Path(args.output_table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)
    print(f"Table saved to: {output_path}")

    # Create plot if requested
    if args.output_plot:
        print(f"\nGenerating comparison plot...")
        create_comparison_plot(results, args.method_names, args.output_plot)

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()
