#!/usr/bin/env python3
"""
Generate Final Result Visualizations

Creates publication-ready plots for the final project documentation.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Create output directory
output_dir = Path(__file__).parent.parent / "results" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating visualizations...")

# 1. Accuracy Evolution
print("  1/4: Accuracy evolution...")
phases = ['Baseline\n(Zero-shot)', 'v0.2.0\n(LoRA FT)', 'v0.3.0\n(+Normalization)',
          'v0.4.0\n(+Loss Masking)', 'v1.0.0\n(Final)']
accuracies = [50.0, 62.5, 90.6, 96.9, 99.0]
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(phases, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=13, fontweight='bold')

# Styling
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Evolution: From Baseline to Final (50% → 99%)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=50, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='Random Baseline')
ax.legend(fontsize=11)

# Add improvement annotations
for i in range(1, len(accuracies)):
    improvement = accuracies[i] - accuracies[i-1]
    mid_x = i - 0.5
    mid_y = (accuracies[i] + accuracies[i-1]) / 2
    ax.annotate(f'+{improvement:.1f}%',
                xy=(mid_x, mid_y),
                fontsize=10,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

plt.tight_layout()
plt.savefig(output_dir / 'accuracy_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Attention-Only vs MLP Comparison
print("  2/4: Model comparison...")
models = ['Attention-Only\n(20.7M params)', 'MLP\n(43.9M params)']
overall = [99.0, 95.8]
speech = [97.9, 100.0]
nonspeech = [100.0, 91.7]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 7))
bars1 = ax.bar(x - width, overall, width, label='Overall', color='#1f77b4', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, speech, width, label='SPEECH', color='#2ca02c', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, nonspeech, width, label='NONSPEECH', color='#ff7f0e', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Attention-Only vs MLP Targets (Extended Test Set: 96 samples)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(85, 105)
ax.legend(fontsize=12, loc='lower right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add winner annotation
ax.annotate('WINNER\n(+3.2% overall)\n2× smaller',
            xy=(-width, 99),
            xytext=(-width-0.3, 102),
            fontsize=11,
            fontweight='bold',
            color='#1f77b4',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Error Breakdown
print("  3/4: Error breakdown...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Attention-Only errors
errors_attention = ['1 SPEECH error\n(47/48 = 97.9%)', '0 NONSPEECH errors\n(48/48 = 100%)']
error_counts_attention = [1, 0]
colors_attention = ['#ff9999', '#99ff99']

ax1.pie(error_counts_attention, labels=errors_attention, autopct=lambda p: f'{int(p/100 * 1)}' if p > 0 else '',
        colors=colors_attention, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Attention-Only Errors\nTotal: 1/96 (99.0%)', fontsize=14, fontweight='bold')

# MLP errors
errors_mlp = ['0 SPEECH errors\n(48/48 = 100%)', '4 NONSPEECH errors\n(44/48 = 91.7%)']
error_counts_mlp = [0, 4]
colors_mlp = ['#99ff99', '#ff9999']

ax2.pie(error_counts_mlp, labels=errors_mlp, autopct=lambda p: f'{int(p/100 * 4)}' if p > 0 else '',
        colors=colors_mlp, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('MLP Errors\nTotal: 4/96 (95.8%)', fontsize=14, fontweight='bold')

plt.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'error_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Model Size Comparison
print("  4/4: Model size comparison...")
fig, ax = plt.subplots(figsize=(10, 7))

categories = ['Model Size\n(MB)', 'Trainable Params\n(Millions)', 'Total Errors\n(out of 96)']
attention_vals = [84, 20.7, 1]
mlp_vals = [168, 43.9, 4]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, attention_vals, width, label='Attention-Only (Winner)',
               color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, mlp_vals, width, label='MLP',
               color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}' if height < 10 else f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Value', fontsize=14, fontweight='bold')
ax.set_title('Efficiency Comparison: Attention-Only vs MLP',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add advantage annotations
ax.annotate('2× smaller', xy=(-width/2, 84), xytext=(-0.7, 140),
            fontsize=10, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.annotate('2× fewer params', xy=(-width/2+1, 20.7), xytext=(0.3, 35),
            fontsize=10, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.annotate('4× fewer errors', xy=(-width/2+2, 1), xytext=(1.3, 2.5),
            fontsize=10, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

plt.tight_layout()
plt.savefig(output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ All visualizations saved to: {output_dir}")
print("\nGenerated files:")
print("  - accuracy_evolution.png")
print("  - model_comparison.png")
print("  - error_breakdown.png")
print("  - efficiency_comparison.png")
