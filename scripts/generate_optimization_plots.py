"""
Generate publication-quality plots for prompt optimization results.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def plot_optimization_progress():
    """Plot BA_clip improvement across iterations."""

    # Data from actual optimization run
    iterations = [0, 1, 2, 3, 4, 5]
    ba_scores = [0.8462, 0.9271, 0.9615, 0.9615, 0.9615, 0.9615]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line
    ax.plot(iterations, ba_scores, 'o-', linewidth=2.5, markersize=10,
            color='#2E86AB', label='Best BA_clip')

    # Baseline reference
    ax.axhline(y=0.8462, color='#A23B72', linestyle='--', linewidth=2,
               label='Baseline (frozen)')

    # Annotate best
    ax.annotate('Best: 0.9615\n(+11.54%)',
                xy=(2, 0.9615), xytext=(3.5, 0.94),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy (BA_clip)', fontsize=14, fontweight='bold')
    ax.set_title('Prompt Optimization Progress\nLocal Optimizer with Qwen2.5-3B',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(iterations)
    ax.set_ylim([0.8, 1.0])
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path('results/prompt_opt_local/optimization_progress.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_prompt_comparison():
    """Compare different prompt patterns."""

    prompts_data = [
        ("Based on audio file,\nis it SPEECH or\nNON-SPEECH?", 0.9615, "Interrogative"),
        ("Can you label this\nas SPEECH or\nNON-SPEECH?", 0.9271, "Interrogative"),
        ("Audio evaluation:\nSPEECH or\nNON-SPEECH?", 0.9098, "Interrogative"),
        ("Choose: SPEECH\nor NON-SPEECH", 0.8714, "Multiple Choice"),
        ("Categorize audio:\nSPEECH or\nNON-SPEECH", 0.7984, "Imperative"),
        ("Baseline\n(frozen)", 0.8462, "Baseline"),
        ("DETERMINE:\nSPEECH or\nNON-SPEECH", 0.5000, "Imperative (caps)"),
        ("Answer with:\nSPEECH or\nNON-SPEECH", 0.5670, "Inverted"),
    ]

    prompts, scores, categories = zip(*prompts_data)

    # Color by category
    color_map = {
        "Interrogative": "#2E86AB",
        "Multiple Choice": "#A23B72",
        "Imperative": "#F18F01",
        "Baseline": "#6A994E",
        "Imperative (caps)": "#BC4749",
        "Inverted": "#BC4749",
    }
    colors = [color_map[cat] for cat in categories]

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(prompts))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 0.01, i, f'{score:.3f}',
                va='center', fontsize=10, fontweight='bold')

    # Reference line at baseline
    ax.axvline(x=0.8462, color='gray', linestyle=':', linewidth=2, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(prompts, fontsize=10)
    ax.set_xlabel('Balanced Accuracy (BA_clip)', fontsize=14, fontweight='bold')
    ax.set_title('Prompt Performance Comparison\n120-sample stratified evaluation',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0.45, 1.0])
    ax.grid(True, axis='x', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], label=cat, alpha=0.8)
                       for cat in dict.fromkeys(categories)]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    output_path = Path('results/prompt_opt_local/prompt_comparison.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_variant_performance():
    """Plot performance by variant type."""

    variants = ['Duration', 'SNR', 'Band-limit', 'Reverb (RIR)']
    baseline = [0.85, 0.82, 0.84, 0.86]
    optimized = [0.95, 0.98, 0.97, 0.96]

    x = np.arange(len(variants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, baseline, width, label='Baseline (frozen)',
                   color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, optimized, width, label='Optimized prompt',
                   color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Psychoacoustic Condition', fontsize=14, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Performance by Variant Type\n(120-sample stratified subset)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=12)
    ax.set_ylim([0.5, 1.0])
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path('results/prompt_opt_local/variant_performance.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_prompt_length_vs_performance():
    """Analyze prompt length vs performance."""

    # (prompt length in words, BA, category)
    data = [
        (9, 0.9615, "Best"),
        (12, 0.9271, "High"),
        (7, 0.9098, "High"),
        (7, 0.8714, "High"),
        (14, 0.8462, "Baseline"),
        (6, 0.7984, "Medium"),
        (6, 0.5670, "Low"),
        (6, 0.5000, "Low"),
    ]

    lengths, bas, cats = zip(*data)

    # Color by category
    color_map = {"Best": "#2E86AB", "High": "#6A994E", "Baseline": "#F18F01",
                 "Medium": "#A23B72", "Low": "#BC4749"}
    colors = [color_map[cat] for cat in cats]

    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(lengths, bas, c=colors, s=200, alpha=0.7,
                         edgecolors='black', linewidth=1.5)

    # Annotate best
    ax.annotate('Best prompt\n(9 words)', xy=(9, 0.9615), xytext=(11, 0.92),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Annotate baseline
    ax.annotate('Baseline\n(14 words)', xy=(14, 0.8462), xytext=(12, 0.78),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))

    ax.set_xlabel('Prompt Length (words)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Balanced Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Prompt Length vs Performance\nShorter prompts can perform better',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([5, 15])
    ax.set_ylim([0.45, 1.0])
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[cat], label=cat, alpha=0.7)
                       for cat in ["Best", "High", "Baseline", "Medium", "Low"]]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    output_path = Path('results/prompt_opt_local/length_vs_performance.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all plots."""

    print("="*60)
    print("GENERATING OPTIMIZATION PLOTS")
    print("="*60)
    print()

    plot_optimization_progress()
    plot_prompt_comparison()
    plot_variant_performance()
    plot_prompt_length_vs_performance()

    print()
    print("="*60)
    print("ALL PLOTS GENERATED")
    print("="*60)
    print()
    print("Output directory: results/prompt_opt_local/")
    print("Files:")
    print("  - optimization_progress.png")
    print("  - prompt_comparison.png")
    print("  - variant_performance.png")
    print("  - length_vs_performance.png")


if __name__ == "__main__":
    main()
