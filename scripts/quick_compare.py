#!/usr/bin/env python3
"""Quick comparison of two model predictions."""

import sys
import pandas as pd

if len(sys.argv) < 3:
    print("Usage: python scripts/quick_compare.py <predictions_A.csv> <predictions_B.csv> [name_A] [name_B]")
    sys.exit(1)

file_A = sys.argv[1]
file_B = sys.argv[2]
name_A = sys.argv[3] if len(sys.argv) > 3 else "Model A"
name_B = sys.argv[4] if len(sys.argv) > 4 else "Model B"

# Load
df_A = pd.read_csv(file_A)
df_B = pd.read_csv(file_B)

# Compute metrics
def compute_metrics(df):
    # Use the 'correct' column (already computed)
    total = len(df)
    correct = df['correct'].sum()

    speech_df = df[df['ground_truth'] == 'SPEECH']
    speech_correct = speech_df['correct'].sum()
    speech_total = len(speech_df)

    nonspeech_df = df[df['ground_truth'] == 'NONSPEECH']
    nonspeech_correct = nonspeech_df['correct'].sum()
    nonspeech_total = len(nonspeech_df)

    return {
        'overall': 100 * correct / total if total > 0 else 0,
        'overall_n': f"{correct}/{total}",
        'speech': 100 * speech_correct / speech_total if speech_total > 0 else 0,
        'speech_n': f"{speech_correct}/{speech_total}",
        'nonspeech': 100 * nonspeech_correct / nonspeech_total if nonspeech_total > 0 else 0,
        'nonspeech_n': f"{nonspeech_correct}/{nonspeech_total}",
    }

metrics_A = compute_metrics(df_A)
metrics_B = compute_metrics(df_B)

# Print
print("="*70)
print("QUICK MODEL COMPARISON")
print("="*70)
print()
print(f"{name_A}:")
print(f"  Overall:   {metrics_A['overall']:.1f}% ({metrics_A['overall_n']})")
print(f"  SPEECH:    {metrics_A['speech']:.1f}% ({metrics_A['speech_n']})")
print(f"  NONSPEECH: {metrics_A['nonspeech']:.1f}% ({metrics_A['nonspeech_n']})")
print()
print(f"{name_B}:")
print(f"  Overall:   {metrics_B['overall']:.1f}% ({metrics_B['overall_n']})")
print(f"  SPEECH:    {metrics_B['speech']:.1f}% ({metrics_B['speech_n']})")
print(f"  NONSPEECH: {metrics_B['nonspeech']:.1f}% ({metrics_B['nonspeech_n']})")
print()
print("Difference (B - A):")
print(f"  Overall:   {metrics_B['overall'] - metrics_A['overall']:+.1f}%")
print(f"  SPEECH:    {metrics_B['speech'] - metrics_A['speech']:+.1f}%")
print(f"  NONSPEECH: {metrics_B['nonspeech'] - metrics_A['nonspeech']:+.1f}%")
print("="*70)
