#!/usr/bin/env python3
"""
Aggregate predictions from multiple seeds and compute mean/std statistics.

Usage:
    python scripts/aggregate_multi_seed.py \
        --model_name "LORA_attn+mlp" \
        --seeds 42 123 456 \
        --predictions_dir results/ablations \
        --output_file results/multi_seed_attn_mlp_summary.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats


def load_seed_predictions(predictions_dir, model_name, seed, use_fixed=False):
    """Load predictions CSV for a specific seed."""
    predictions_dir = Path(predictions_dir)

    # Try different naming patterns
    patterns = [
        f"{model_name}_seed{seed}_FIXED.csv",
        f"{model_name}_seed{seed}.csv",
        f"LORA_attn_mlp_seed{seed}_FIXED.csv",
        f"LORA_attn_mlp_seed{seed}.csv"
    ]

    for pattern in patterns:
        csv_path = predictions_dir / pattern
        if csv_path.exists():
            print(f"  Loading: {csv_path}")
            return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Could not find predictions for seed {seed} in {predictions_dir}")


def compute_accuracy(df):
    """Compute overall and per-class accuracy."""
    correct = df['correct'].sum()
    total = len(df)
    accuracy = correct / total

    # Per-class accuracy
    speech_df = df[df['ground_truth'] == 'SPEECH']
    nonspeech_df = df[df['ground_truth'] == 'NONSPEECH']

    speech_acc = speech_df['correct'].sum() / len(speech_df) if len(speech_df) > 0 else 0.0
    nonspeech_acc = nonspeech_df['correct'].sum() / len(nonspeech_df) if len(nonspeech_df) > 0 else 0.0

    return {
        'overall': accuracy,
        'speech': speech_acc,
        'nonspeech': nonspeech_acc,
        'n_correct': correct,
        'n_total': total
    }


def compute_confidence_stats(df):
    """Compute confidence statistics."""
    return {
        'mean': df['confidence'].mean(),
        'std': df['confidence'].std(),
        'min': df['confidence'].min(),
        'max': df['confidence'].max()
    }


def aggregate_seeds(predictions_dir, model_name, seeds):
    """Aggregate results across multiple seeds."""
    results = []
    all_predictions = {}

    print(f"\nLoading predictions for {len(seeds)} seeds...")
    for seed in seeds:
        df = load_seed_predictions(predictions_dir, model_name, seed)
        acc = compute_accuracy(df)
        conf = compute_confidence_stats(df)

        results.append({
            'seed': seed,
            'accuracy': acc['overall'],
            'speech_acc': acc['speech'],
            'nonspeech_acc': acc['nonspeech'],
            'n_correct': acc['n_correct'],
            'n_total': acc['n_total'],
            'conf_mean': conf['mean'],
            'conf_std': conf['std']
        })

        all_predictions[seed] = df

        print(f"  Seed {seed}: {acc['overall']:.1%} overall ({acc['n_correct']}/{acc['n_total']})")

    return results, all_predictions


def compute_aggregated_stats(results):
    """Compute mean and std across seeds."""
    accuracies = [r['accuracy'] for r in results]
    speech_accs = [r['speech_acc'] for r in results]
    nonspeech_accs = [r['nonspeech_acc'] for r in results]

    return {
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0,
        'speech_acc_mean': np.mean(speech_accs),
        'speech_acc_std': np.std(speech_accs, ddof=1) if len(speech_accs) > 1 else 0.0,
        'nonspeech_acc_mean': np.mean(nonspeech_accs),
        'nonspeech_acc_std': np.std(nonspeech_accs, ddof=1) if len(nonspeech_accs) > 1 else 0.0
    }


def analyze_disagreements(all_predictions):
    """Find samples where predictions differ across seeds."""
    seeds = sorted(all_predictions.keys())
    base_df = all_predictions[seeds[0]]

    disagreements = []
    for idx, row in base_df.iterrows():
        clip_id = row['clip_id']

        # Get predictions for this clip across all seeds
        predictions = []
        correct_flags = []
        for seed in seeds:
            seed_df = all_predictions[seed]
            seed_row = seed_df[seed_df['clip_id'] == clip_id]
            if len(seed_row) > 0:
                predictions.append(seed_row.iloc[0]['prediction'])
                correct_flags.append(seed_row.iloc[0]['correct'])

        # Check if there's disagreement
        if len(set(predictions)) > 1:
            disagreements.append({
                'clip_id': clip_id,
                'ground_truth': row['ground_truth'],
                'predictions': dict(zip(seeds, predictions)),
                'correct_flags': dict(zip(seeds, correct_flags)),
                'agreement_rate': sum(correct_flags) / len(correct_flags)
            })

    return disagreements


def write_summary_report(output_file, model_name, seeds, results, stats, disagreements):
    """Write comprehensive summary report."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"MULTI-SEED EVALUATION SUMMARY: {model_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Seeds evaluated: {seeds}\n")
        f.write(f"Number of seeds: {len(seeds)}\n\n")

        # Overall statistics
        f.write("-" * 80 + "\n")
        f.write("OVERALL ACCURACY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean: {stats['accuracy_mean']:.1%} ± {stats['accuracy_std']:.1%}\n\n")

        # Per-seed breakdown
        f.write("Per-seed breakdown:\n")
        for r in results:
            f.write(f"  Seed {r['seed']:3d}: {r['accuracy']:.1%} ({r['n_correct']}/{r['n_total']})\n")
        f.write("\n")

        # Per-class accuracy
        f.write("-" * 80 + "\n")
        f.write("PER-CLASS ACCURACY\n")
        f.write("-" * 80 + "\n")
        f.write(f"SPEECH:    {stats['speech_acc_mean']:.1%} ± {stats['speech_acc_std']:.1%}\n")
        f.write(f"NONSPEECH: {stats['nonspeech_acc_mean']:.1%} ± {stats['nonspeech_acc_std']:.1%}\n\n")

        # Per-seed per-class breakdown
        f.write("Per-seed breakdown:\n")
        for r in results:
            f.write(f"  Seed {r['seed']:3d}: SPEECH={r['speech_acc']:.1%}, NONSPEECH={r['nonspeech_acc']:.1%}\n")
        f.write("\n")

        # Confidence statistics
        f.write("-" * 80 + "\n")
        f.write("CONFIDENCE STATISTICS\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"  Seed {r['seed']:3d}: mean={r['conf_mean']:.3f}, std={r['conf_std']:.3f}\n")
        f.write("\n")

        # Disagreements
        f.write("-" * 80 + "\n")
        f.write("CROSS-SEED DISAGREEMENTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples: {results[0]['n_total']}\n")
        f.write(f"Disagreements: {len(disagreements)}\n")
        f.write(f"Agreement rate: {1 - len(disagreements)/results[0]['n_total']:.1%}\n\n")

        if len(disagreements) > 0:
            f.write("Samples with disagreements:\n")
            for d in disagreements:
                f.write(f"\n  Clip: {d['clip_id']}\n")
                f.write(f"  Ground truth: {d['ground_truth']}\n")
                f.write(f"  Predictions:\n")
                for seed, pred in d['predictions'].items():
                    correct = d['correct_flags'][seed]
                    status = "CORRECT" if correct else "WRONG"
                    f.write(f"    Seed {seed}: {pred} ({status})\n")
                f.write(f"  Agreement rate: {d['agreement_rate']:.1%}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nSummary report written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed predictions")
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name (e.g., "LORA_attn+mlp")')
    parser.add_argument('--seeds', type=int, nargs='+', required=True,
                        help='List of seeds to aggregate (e.g., 42 123 456)')
    parser.add_argument('--predictions_dir', type=str, required=True,
                        help='Directory containing prediction CSVs')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file for summary report')

    args = parser.parse_args()

    print("=" * 80)
    print("MULTI-SEED AGGREGATION")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Seeds: {args.seeds}")
    print(f"Predictions dir: {args.predictions_dir}")

    # Load and aggregate results
    results, all_predictions = aggregate_seeds(
        args.predictions_dir,
        args.model_name,
        args.seeds
    )

    # Compute aggregated statistics
    stats = compute_aggregated_stats(results)

    print("\n" + "-" * 80)
    print("AGGREGATED RESULTS")
    print("-" * 80)
    print(f"Overall accuracy: {stats['accuracy_mean']:.1%} ± {stats['accuracy_std']:.1%}")
    print(f"SPEECH:    {stats['speech_acc_mean']:.1%} ± {stats['speech_acc_std']:.1%}")
    print(f"NONSPEECH: {stats['nonspeech_acc_mean']:.1%} ± {stats['nonspeech_acc_std']:.1%}")

    # Analyze disagreements
    disagreements = analyze_disagreements(all_predictions)
    print(f"\nCross-seed agreement: {1 - len(disagreements)/results[0]['n_total']:.1%}")
    print(f"Disagreements: {len(disagreements)}/{results[0]['n_total']}")

    # Write summary report
    write_summary_report(
        args.output_file,
        args.model_name,
        args.seeds,
        results,
        stats,
        disagreements
    )

    print("\n" + "=" * 80)
    print("[OK] Multi-seed aggregation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
