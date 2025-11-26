#!/usr/bin/env python3
"""
Generate comprehensive analysis plots for OPRO Qwen speech detection.

This script creates:
1. Psychometric curves (BA vs duration, BA vs SNR)
2. BASE vs LoRA comparison
3. p_first_token distribution analysis
4. Per-condition heatmap (duration x SNR)
5. Summary figure for paper/presentation

Usage:
    python scripts/generate_analysis_plots.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Output directory
OUTPUT_DIR = Path("results/analysis_plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading prediction data...")
base_df = pd.read_csv("results/eval_base_thresh0.50/predictions.csv")
lora_df = pd.read_csv("results/eval_lora_thresh0.50/predictions.csv")

print(f"  BASE: {len(base_df)} samples")
print(f"  LoRA: {len(lora_df)} samples")


def calculate_metrics_by_group(df, group_col, threshold=None, p_col='p_first_token'):
    """Calculate BA, speech_acc, nonspeech_acc by group."""
    results = []

    for group_val in sorted(df[group_col].dropna().unique()):
        group_df = df[df[group_col] == group_val]

        if threshold is not None:
            # Apply custom threshold
            pred = (group_df[p_col] > threshold).map({True: 'SPEECH', False: 'NONSPEECH'})
            correct = pred == group_df['ground_truth']
        else:
            # Use existing predictions
            correct = group_df['correct']

        speech_mask = group_df['ground_truth'] == 'SPEECH'
        nonspeech_mask = group_df['ground_truth'] == 'NONSPEECH'

        speech_acc = correct[speech_mask].mean() if speech_mask.sum() > 0 else 0
        nonspeech_acc = correct[nonspeech_mask].mean() if nonspeech_mask.sum() > 0 else 0
        ba = (speech_acc + nonspeech_acc) / 2

        results.append({
            group_col: group_val,
            'ba': ba,
            'speech_acc': speech_acc,
            'nonspeech_acc': nonspeech_acc,
            'n_samples': len(group_df)
        })

    return pd.DataFrame(results)


# =============================================================================
# 1. PSYCHOMETRIC CURVES
# =============================================================================
print("\n1. Creating psychometric curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Duration curve
ax1 = axes[0]
base_dur = calculate_metrics_by_group(base_df, 'duration_ms', threshold=0.50)
lora_dur = calculate_metrics_by_group(lora_df, 'duration_ms', threshold=0.60)

ax1.plot(base_dur['duration_ms'], base_dur['ba'] * 100, 'b-o', linewidth=2,
         markersize=8, label='BASE (thresh=0.50)')
ax1.plot(lora_dur['duration_ms'], lora_dur['ba'] * 100, 'r-s', linewidth=2,
         markersize=8, label='LoRA (thresh=0.60)')
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax1.set_xlabel('Speech Duration (ms)')
ax1.set_ylabel('Balanced Accuracy (%)')
ax1.set_title('Psychometric Curve: BA vs Speech Duration')
ax1.legend(loc='lower right')
ax1.set_ylim(45, 95)
ax1.grid(True, alpha=0.3)

# SNR curve
ax2 = axes[1]
base_snr = calculate_metrics_by_group(base_df, 'snr_db', threshold=0.50)
lora_snr = calculate_metrics_by_group(lora_df, 'snr_db', threshold=0.60)

ax2.plot(base_snr['snr_db'], base_snr['ba'] * 100, 'b-o', linewidth=2,
         markersize=8, label='BASE (thresh=0.50)')
ax2.plot(lora_snr['snr_db'], lora_snr['ba'] * 100, 'r-s', linewidth=2,
         markersize=8, label='LoRA (thresh=0.60)')
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('Balanced Accuracy (%)')
ax2.set_title('Psychometric Curve: BA vs Signal-to-Noise Ratio')
ax2.legend(loc='lower right')
ax2.set_ylim(45, 95)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "psychometric_curves.png", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "psychometric_curves.pdf", bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'psychometric_curves.png'}")


# =============================================================================
# 2. BASE vs LoRA COMPARISON
# =============================================================================
print("\n2. Creating BASE vs LoRA comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Overall metrics bar chart
ax1 = axes[0]
metrics = ['BA', 'Speech Acc', 'Nonspeech Acc']
base_vals = [73.26, 72.57, 73.96]
lora_vals = [78.21, 82.47, 73.96]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, base_vals, width, label='BASE', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, lora_vals, width, label='LoRA', color='indianred', alpha=0.8)

ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Overall Performance Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.set_ylim(0, 100)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

# BA by duration
ax2 = axes[1]
durations = base_dur['duration_ms'].values
x = np.arange(len(durations))
width = 0.35

bars1 = ax2.bar(x - width/2, base_dur['ba'] * 100, width, label='BASE', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x + width/2, lora_dur['ba'] * 100, width, label='LoRA', color='indianred', alpha=0.8)

ax2.set_ylabel('Balanced Accuracy (%)')
ax2.set_title('BA by Speech Duration')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{int(d)}ms' for d in durations], rotation=45, ha='right')
ax2.legend()
ax2.set_ylim(40, 100)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

# BA by SNR
ax3 = axes[2]
snrs = base_snr['snr_db'].values
x = np.arange(len(snrs))

bars1 = ax3.bar(x - width/2, base_snr['ba'] * 100, width, label='BASE', color='steelblue', alpha=0.8)
bars2 = ax3.bar(x + width/2, lora_snr['ba'] * 100, width, label='LoRA', color='indianred', alpha=0.8)

ax3.set_ylabel('Balanced Accuracy (%)')
ax3.set_title('BA by SNR')
ax3.set_xticks(x)
ax3.set_xticklabels([f'{int(s)}dB' for s in snrs])
ax3.legend()
ax3.set_ylim(40, 100)
ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "base_vs_lora_comparison.png", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "base_vs_lora_comparison.pdf", bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'base_vs_lora_comparison.png'}")


# =============================================================================
# 3. p_first_token DISTRIBUTION ANALYSIS
# =============================================================================
print("\n3. Creating p_first_token distribution analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# BASE distribution
ax1 = axes[0, 0]
base_speech = base_df[base_df['ground_truth'] == 'SPEECH']['p_first_token']
base_nonspeech = base_df[base_df['ground_truth'] == 'NONSPEECH']['p_first_token']
ax1.hist(base_speech, bins=30, alpha=0.6, color='green', label=f'SPEECH (n={len(base_speech)})', density=True)
ax1.hist(base_nonspeech, bins=30, alpha=0.6, color='orange', label=f'NONSPEECH (n={len(base_nonspeech)})', density=True)
ax1.axvline(x=0.50, color='red', linestyle='--', linewidth=2, label='Optimal threshold (0.50)')
ax1.set_xlabel('p_first_token')
ax1.set_ylabel('Density')
ax1.set_title('BASE Model: p_first_token Distribution')
ax1.legend()

# LoRA distribution
ax2 = axes[0, 1]
lora_speech = lora_df[lora_df['ground_truth'] == 'SPEECH']['p_first_token']
lora_nonspeech = lora_df[lora_df['ground_truth'] == 'NONSPEECH']['p_first_token']
ax2.hist(lora_speech, bins=30, alpha=0.6, color='green', label=f'SPEECH (n={len(lora_speech)})', density=True)
ax2.hist(lora_nonspeech, bins=30, alpha=0.6, color='orange', label=f'NONSPEECH (n={len(lora_nonspeech)})', density=True)
ax2.axvline(x=0.60, color='red', linestyle='--', linewidth=2, label='Optimal threshold (0.60)')
ax2.set_xlabel('p_first_token')
ax2.set_ylabel('Density')
ax2.set_title('LoRA Model: p_first_token Distribution')
ax2.legend()

# Threshold sweep comparison
ax3 = axes[1, 0]
thresholds = np.arange(0.2, 0.8, 0.02)

base_ba = []
lora_ba = []
for t in thresholds:
    # BASE
    pred = (base_df['p_first_token'] > t).map({True: 'SPEECH', False: 'NONSPEECH'})
    sp = (pred[base_df['ground_truth'] == 'SPEECH'] == 'SPEECH').mean()
    ns = (pred[base_df['ground_truth'] == 'NONSPEECH'] == 'NONSPEECH').mean()
    base_ba.append((sp + ns) / 2)

    # LoRA
    pred = (lora_df['p_first_token'] > t).map({True: 'SPEECH', False: 'NONSPEECH'})
    sp = (pred[lora_df['ground_truth'] == 'SPEECH'] == 'SPEECH').mean()
    ns = (pred[lora_df['ground_truth'] == 'NONSPEECH'] == 'NONSPEECH').mean()
    lora_ba.append((sp + ns) / 2)

ax3.plot(thresholds, np.array(base_ba) * 100, 'b-', linewidth=2, label='BASE')
ax3.plot(thresholds, np.array(lora_ba) * 100, 'r-', linewidth=2, label='LoRA')
ax3.axvline(x=0.50, color='blue', linestyle='--', alpha=0.5, label='BASE optimal')
ax3.axvline(x=0.60, color='red', linestyle='--', alpha=0.5, label='LoRA optimal')
ax3.set_xlabel('Threshold')
ax3.set_ylabel('Balanced Accuracy (%)')
ax3.set_title('Threshold Sweep: BA vs Decision Threshold')
ax3.legend()
ax3.set_xlim(0.2, 0.8)
ax3.set_ylim(50, 85)

# Box plot comparison
ax4 = axes[1, 1]
data = [
    base_speech.values, base_nonspeech.values,
    lora_speech.values, lora_nonspeech.values
]
positions = [1, 2, 4, 5]
colors = ['lightgreen', 'lightsalmon', 'green', 'darkorange']
bp = ax4.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax4.set_xticks([1.5, 4.5])
ax4.set_xticklabels(['BASE', 'LoRA'])
ax4.set_ylabel('p_first_token')
ax4.set_title('p_first_token by Model and Class')
ax4.axhline(y=0.50, color='blue', linestyle='--', alpha=0.5, label='BASE thresh')
ax4.axhline(y=0.60, color='red', linestyle='--', alpha=0.5, label='LoRA thresh')

# Add legend manually
speech_patch = mpatches.Patch(color='green', alpha=0.7, label='SPEECH')
nonspeech_patch = mpatches.Patch(color='darkorange', alpha=0.7, label='NONSPEECH')
ax4.legend(handles=[speech_patch, nonspeech_patch], loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "p_first_token_analysis.png", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "p_first_token_analysis.pdf", bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'p_first_token_analysis.png'}")


# =============================================================================
# 4. PER-CONDITION HEATMAP (Duration x SNR)
# =============================================================================
print("\n4. Creating per-condition heatmap...")

def create_heatmap_data(df, threshold):
    """Create duration x SNR heatmap matrix."""
    durations = sorted(df['duration_ms'].dropna().unique())
    snrs = sorted(df['snr_db'].dropna().unique())

    matrix = np.zeros((len(durations), len(snrs)))

    for i, dur in enumerate(durations):
        for j, snr in enumerate(snrs):
            mask = (df['duration_ms'] == dur) & (df['snr_db'] == snr)
            subset = df[mask]

            if len(subset) > 0:
                pred = (subset['p_first_token'] > threshold).map({True: 'SPEECH', False: 'NONSPEECH'})
                correct = pred == subset['ground_truth']

                speech_mask = subset['ground_truth'] == 'SPEECH'
                nonspeech_mask = subset['ground_truth'] == 'NONSPEECH'

                sp_acc = correct[speech_mask].mean() if speech_mask.sum() > 0 else 0
                ns_acc = correct[nonspeech_mask].mean() if nonspeech_mask.sum() > 0 else 0
                ba = (sp_acc + ns_acc) / 2
                matrix[i, j] = ba * 100
            else:
                matrix[i, j] = np.nan

    return matrix, durations, snrs

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# BASE heatmap
ax1 = axes[0]
base_matrix, durations, snrs = create_heatmap_data(base_df, 0.50)
im1 = ax1.imshow(base_matrix, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
ax1.set_xticks(range(len(snrs)))
ax1.set_xticklabels([f'{int(s)}' for s in snrs])
ax1.set_yticks(range(len(durations)))
ax1.set_yticklabels([f'{int(d)}' for d in durations])
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('Duration (ms)')
ax1.set_title('BASE Model: BA by Duration x SNR')
plt.colorbar(im1, ax=ax1, label='Balanced Accuracy (%)')

# Add text annotations
for i in range(len(durations)):
    for j in range(len(snrs)):
        if not np.isnan(base_matrix[i, j]):
            text = ax1.text(j, i, f'{base_matrix[i, j]:.0f}', ha='center', va='center',
                           fontsize=8, color='black' if 50 < base_matrix[i, j] < 85 else 'white')

# LoRA heatmap
ax2 = axes[1]
lora_matrix, _, _ = create_heatmap_data(lora_df, 0.60)
im2 = ax2.imshow(lora_matrix, cmap='RdYlGn', aspect='auto', vmin=40, vmax=100)
ax2.set_xticks(range(len(snrs)))
ax2.set_xticklabels([f'{int(s)}' for s in snrs])
ax2.set_yticks(range(len(durations)))
ax2.set_yticklabels([f'{int(d)}' for d in durations])
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('Duration (ms)')
ax2.set_title('LoRA Model: BA by Duration x SNR')
plt.colorbar(im2, ax=ax2, label='Balanced Accuracy (%)')

# Add text annotations
for i in range(len(durations)):
    for j in range(len(snrs)):
        if not np.isnan(lora_matrix[i, j]):
            text = ax2.text(j, i, f'{lora_matrix[i, j]:.0f}', ha='center', va='center',
                           fontsize=8, color='black' if 50 < lora_matrix[i, j] < 85 else 'white')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "condition_heatmap.png", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "condition_heatmap.pdf", bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'condition_heatmap.png'}")


# =============================================================================
# 5. IMPROVEMENT HEATMAP (LoRA - BASE)
# =============================================================================
print("\n5. Creating improvement heatmap...")

fig, ax = plt.subplots(figsize=(10, 6))

improvement_matrix = lora_matrix - base_matrix
im = ax.imshow(improvement_matrix, cmap='RdBu', aspect='auto', vmin=-20, vmax=20)
ax.set_xticks(range(len(snrs)))
ax.set_xticklabels([f'{int(s)}' for s in snrs])
ax.set_yticks(range(len(durations)))
ax.set_yticklabels([f'{int(d)}' for d in durations])
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('Duration (ms)')
ax.set_title('LoRA Improvement over BASE (BA percentage points)')
plt.colorbar(im, ax=ax, label='Improvement (%)')

# Add text annotations
for i in range(len(durations)):
    for j in range(len(snrs)):
        if not np.isnan(improvement_matrix[i, j]):
            val = improvement_matrix[i, j]
            color = 'black' if -10 < val < 10 else 'white'
            sign = '+' if val > 0 else ''
            text = ax.text(j, i, f'{sign}{val:.0f}', ha='center', va='center',
                          fontsize=8, color=color)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "improvement_heatmap.png", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "improvement_heatmap.pdf", bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'improvement_heatmap.png'}")


# =============================================================================
# 6. SUMMARY FIGURE (for paper/presentation)
# =============================================================================
print("\n6. Creating summary figure...")

fig = plt.figure(figsize=(16, 12))

# Create grid
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Overall comparison (top-left)
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['BA', 'Speech', 'Nonspeech']
base_vals = [73.26, 72.57, 73.96]
lora_vals = [78.21, 82.47, 73.96]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax1.bar(x - width/2, base_vals, width, label='BASE', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, lora_vals, width, label='LoRA', color='indianred', alpha=0.8)
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('A) Overall Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(loc='lower right')
ax1.set_ylim(60, 90)

# 2. Duration curve (top-middle)
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(base_dur['duration_ms'], base_dur['ba'] * 100, 'b-o', linewidth=2, markersize=6, label='BASE')
ax2.plot(lora_dur['duration_ms'], lora_dur['ba'] * 100, 'r-s', linewidth=2, markersize=6, label='LoRA')
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Duration (ms)')
ax2.set_ylabel('BA (%)')
ax2.set_title('B) BA vs Duration')
ax2.legend(loc='lower right')
ax2.set_ylim(45, 95)

# 3. SNR curve (top-right)
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(base_snr['snr_db'], base_snr['ba'] * 100, 'b-o', linewidth=2, markersize=6, label='BASE')
ax3.plot(lora_snr['snr_db'], lora_snr['ba'] * 100, 'r-s', linewidth=2, markersize=6, label='LoRA')
ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('SNR (dB)')
ax3.set_ylabel('BA (%)')
ax3.set_title('C) BA vs SNR')
ax3.legend(loc='lower right')
ax3.set_ylim(45, 95)

# 4. p_first_token distributions (middle row, spans 2 columns)
ax4 = fig.add_subplot(gs[1, :2])
# Violin plot
parts = ax4.violinplot([base_speech, base_nonspeech, lora_speech, lora_nonspeech],
                        positions=[1, 2, 4, 5], showmeans=True, showmedians=True)
colors = ['lightgreen', 'lightsalmon', 'green', 'darkorange']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
ax4.axhline(y=0.50, color='blue', linestyle='--', alpha=0.5, label='BASE thresh')
ax4.axhline(y=0.60, color='red', linestyle='--', alpha=0.5, label='LoRA thresh')
ax4.set_xticks([1.5, 4.5])
ax4.set_xticklabels(['BASE Model', 'LoRA Model'])
ax4.set_ylabel('p_first_token')
ax4.set_title('D) p_first_token Distribution by Class')

# 5. Threshold sweep (middle-right)
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(thresholds, np.array(base_ba) * 100, 'b-', linewidth=2, label='BASE')
ax5.plot(thresholds, np.array(lora_ba) * 100, 'r-', linewidth=2, label='LoRA')
ax5.axvline(x=0.50, color='blue', linestyle='--', alpha=0.5)
ax5.axvline(x=0.60, color='red', linestyle='--', alpha=0.5)
ax5.set_xlabel('Threshold')
ax5.set_ylabel('BA (%)')
ax5.set_title('E) Threshold Sweep')
ax5.legend()
ax5.set_xlim(0.3, 0.7)

# 6. Improvement heatmap (bottom, spans all columns)
ax6 = fig.add_subplot(gs[2, :])
im = ax6.imshow(improvement_matrix.T, cmap='RdBu', aspect='auto', vmin=-20, vmax=20)
ax6.set_yticks(range(len(snrs)))
ax6.set_yticklabels([f'{int(s)}dB' for s in snrs])
ax6.set_xticks(range(len(durations)))
ax6.set_xticklabels([f'{int(d)}ms' for d in durations])
ax6.set_ylabel('SNR')
ax6.set_xlabel('Duration')
ax6.set_title('F) LoRA Improvement over BASE (BA percentage points)')
cbar = plt.colorbar(im, ax=ax6, orientation='vertical', pad=0.02)
cbar.set_label('Improvement (%)')

# Add text annotations to heatmap
for i in range(len(durations)):
    for j in range(len(snrs)):
        if not np.isnan(improvement_matrix[i, j]):
            val = improvement_matrix[i, j]
            color = 'black' if -10 < val < 10 else 'white'
            sign = '+' if val > 0 else ''
            ax6.text(i, j, f'{sign}{val:.0f}', ha='center', va='center', fontsize=7, color=color)

# Add main title
fig.suptitle('Speech Detection: Qwen2-Audio with OPRO-Optimized Prompts',
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig(OUTPUT_DIR / "summary_figure.png", bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "summary_figure.pdf", bbox_inches='tight')
print(f"  Saved: {OUTPUT_DIR / 'summary_figure.png'}")


# =============================================================================
# 7. SAVE METRICS SUMMARY
# =============================================================================
print("\n7. Saving metrics summary...")

summary = {
    "base_model": {
        "threshold": 0.50,
        "overall": {
            "ba": 73.26,
            "speech_acc": 72.57,
            "nonspeech_acc": 73.96
        },
        "by_duration": base_dur.to_dict('records'),
        "by_snr": base_snr.to_dict('records')
    },
    "lora_model": {
        "threshold": 0.60,
        "overall": {
            "ba": 78.21,
            "speech_acc": 82.47,
            "nonspeech_acc": 73.96
        },
        "by_duration": lora_dur.to_dict('records'),
        "by_snr": lora_snr.to_dict('records')
    },
    "improvement": {
        "ba": 4.95,
        "speech_acc": 9.90,
        "nonspeech_acc": 0.0
    },
    "prompt": "Is this short clip speech or noise? Your answer should be SPEECH or NON-SPEECH.",
    "test_samples": len(base_df)
}

with open(OUTPUT_DIR / "metrics_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {OUTPUT_DIR / 'metrics_summary.json'}")


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
for f in sorted(OUTPUT_DIR.glob("*")):
    print(f"  - {f.name}")
