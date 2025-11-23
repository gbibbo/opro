#!/usr/bin/env python3
"""
Consolidate OPRO legacy experiment results.

Reads iteration CSVs from results/opro_ab, results/opro_mc, results/opro_open
and generates:
- summary.json for each format (accuracy per iteration)
- comparison table across all formats
"""

import json
import pandas as pd
from pathlib import Path


def consolidate_format_results(results_dir, format_name, num_iterations=7):
    """
    Consolidate results for a single format (ab/mc/open).

    Args:
        results_dir: Path to results directory for this format
        format_name: Name of format (ab/mc/open)
        num_iterations: Number of iterations to process

    Returns:
        dict: Summary statistics
    """
    results_dir = Path(results_dir)

    print(f"\nProcessing {format_name.upper()} format...")
    print(f"  Directory: {results_dir}")

    # Check directory exists
    if not results_dir.exists():
        print(f"  ERROR: Directory not found: {results_dir}")
        return None

    iteration_stats = []
    all_iterations_data = []

    for iter_num in range(1, num_iterations + 1):
        csv_file = results_dir / f"iter{iter_num:02d}_all_predictions.csv"

        if not csv_file.exists():
            print(f"  WARNING: File not found: {csv_file}")
            continue

        # Read CSV
        df = pd.read_csv(csv_file)

        # The CSV has these columns:
        # audio_path, ground_truth, raw_text, normalized_label,
        # is_correct, p_first_token, prompt_id, iteration, decoding_mode

        # Calculate accuracy for this iteration
        total_predictions = len(df)
        correct_predictions = df['is_correct'].sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Get unique prompts tested in this iteration
        num_prompts = df['prompt_id'].nunique()

        # Calculate per-prompt accuracies
        prompt_accuracies = df.groupby('prompt_id')['is_correct'].mean().to_dict()
        best_prompt_id = max(prompt_accuracies, key=prompt_accuracies.get)
        best_prompt_acc = prompt_accuracies[best_prompt_id]

        iteration_stats.append({
            'iteration': iter_num,
            'total_samples': total_predictions,
            'correct': int(correct_predictions),
            'accuracy': round(accuracy, 4),
            'num_prompts_tested': num_prompts,
            'best_prompt_id': best_prompt_id,
            'best_prompt_accuracy': round(best_prompt_acc, 4),
            'prompt_accuracies': {k: round(v, 4) for k, v in prompt_accuracies.items()}
        })

        print(f"  Iter {iter_num:02d}: {accuracy:.1%} ({correct_predictions}/{total_predictions}) - "
              f"{num_prompts} prompts, best: {best_prompt_id} ({best_prompt_acc:.1%})")

        # Collect data for aggregation
        all_iterations_data.append(df)

    if not iteration_stats:
        print(f"  ERROR: No valid iterations found for {format_name}")
        return None

    # Aggregate all iterations
    all_df = pd.concat(all_iterations_data, ignore_index=True)

    # Overall statistics
    overall_accuracy = all_df['is_correct'].mean()

    # Best iteration (by accuracy)
    best_iter = max(iteration_stats, key=lambda x: x['accuracy'])

    summary = {
        'format': format_name,
        'num_iterations': len(iteration_stats),
        'total_samples': len(all_df),
        'overall_accuracy': round(overall_accuracy, 4),
        'best_iteration': best_iter['iteration'],
        'best_iteration_accuracy': best_iter['accuracy'],
        'iterations': iteration_stats,
        'final_iteration_accuracy': iteration_stats[-1]['accuracy']
    }

    # Save summary
    summary_file = results_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to: {summary_file}")

    return summary


def create_comparison_table(summaries):
    """
    Create comparison table across formats.

    Args:
        summaries: Dict of {format_name: summary_dict}
    """
    print("\n" + "="*80)
    print("OPRO LEGACY EXPERIMENTS - COMPARISON TABLE")
    print("="*80)

    # Header
    print(f"\n{'Format':<10} {'Iterations':<12} {'Total Samples':<15} "
          f"{'Overall Acc':<12} {'Best Iter':<10} {'Best Acc':<10} {'Final Acc':<10}")
    print("-" * 80)

    # Sort formats
    format_order = ['ab', 'mc', 'open']

    for fmt in format_order:
        if fmt not in summaries or summaries[fmt] is None:
            print(f"{fmt.upper():<10} {'N/A':<12} {'N/A':<15} {'N/A':<12} "
                  f"{'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue

        s = summaries[fmt]
        print(f"{fmt.upper():<10} {s['num_iterations']:<12} {s['total_samples']:<15} "
              f"{s['overall_accuracy']:.1%}{'':<8} {s['best_iteration']:<10} "
              f"{s['best_iteration_accuracy']:.1%}{'':<6} {s['final_iteration_accuracy']:.1%}{'':<6}")

    print("="*80)

    # Per-iteration comparison
    print("\nPer-Iteration Accuracy Comparison:")
    print(f"\n{'Iteration':<12} ", end='')
    for fmt in format_order:
        if fmt in summaries and summaries[fmt] is not None:
            print(f"{fmt.upper():<12}", end='')
    print()
    print("-" * 60)

    max_iters = max(
        s['num_iterations'] for s in summaries.values() if s is not None
    )

    for i in range(1, max_iters + 1):
        print(f"Iter {i:02d}{'':<6}", end='')
        for fmt in format_order:
            if fmt in summaries and summaries[fmt] is not None:
                iter_stats = [it for it in summaries[fmt]['iterations'] if it['iteration'] == i]
                if iter_stats:
                    acc = iter_stats[0]['accuracy']
                    print(f"{acc:.1%}{'':<8}", end='')
                else:
                    print(f"{'N/A':<12}", end='')
        print()

    print("="*80)


def main():
    """Main consolidation function."""

    print("="*80)
    print("OPRO LEGACY EXPERIMENTS - RESULTS CONSOLIDATION")
    print("="*80)

    base_dir = Path("results")

    # Process each format
    summaries = {}

    for format_name in ['ab', 'mc', 'open']:
        format_dir = base_dir / f"opro_{format_name}"
        summary = consolidate_format_results(format_dir, format_name, num_iterations=7)
        summaries[format_name] = summary

    # Create comparison table
    if any(s is not None for s in summaries.values()):
        create_comparison_table(summaries)

        # Save comparison to file
        comparison_file = base_dir / "opro_legacy_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(summaries, f, indent=2)
        print(f"\nComparison data saved to: {comparison_file}")
    else:
        print("\nERROR: No valid results found for any format!")
        return 1

    print("\n" + "="*80)
    print("CONSOLIDATION COMPLETE")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
