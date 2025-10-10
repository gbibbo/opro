#!/usr/bin/env python3
"""
Extended evaluation of Qwen2-Audio with noise padding.

Tests ~210 samples (30 per duration) to validate statistical significance
of the padding strategy findings.

This is 10x larger than the previous test (21 samples).
"""

import sys
from pathlib import Path, PureWindowsPath
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from qsm.models import Qwen2AudioClassifier


def main():
    print("="*80)
    print("EXTENDED EVALUATION: Noise Padding Strategy")
    print("Testing ~210 samples (30 per duration) for statistical validation")
    print("="*80)

    # Load model with auto-padding enabled (2000ms target - optimal)
    print("\nLoading model with auto-padding enabled...")
    model = Qwen2AudioClassifier(
        device="cuda",
        torch_dtype="float16",
        load_in_4bit=True,
        auto_pad=True,  # Explicit (though it's default)
        pad_target_ms=2000,  # Optimal: 2000ms provides best performance (validated)
        pad_noise_amplitude=0.0001,  # Low noise to minimize interference
    )

    print("\nPrompt configuration:")
    print(f"  System: {model.system_prompt}")
    print(f"  User: {model.user_prompt[:80]}...")

    # Load samples
    vox_meta = pd.read_parquet("data/segments/voxconverse/dev/segments_metadata.parquet")
    vox_dir = Path("data/segments/voxconverse/dev")

    # Test durations (including 1000ms to validate baseline)
    test_durations = [20, 40, 60, 80, 100, 200, 500, 1000]
    samples_per_duration = 30

    print(f"\nTest configuration:")
    print(f"  Durations: {test_durations}")
    print(f"  Samples per duration: {samples_per_duration}")
    print(f"  Total samples: {len(test_durations) * samples_per_duration}")

    # Initialize results tracking
    results_by_duration = {}
    all_results = []

    start_time = time.time()

    print("\nStarting evaluation...")
    print("="*80)

    for duration in test_durations:
        print(f"\n[Duration: {duration}ms]")

        # Get samples for this duration
        duration_samples = vox_meta[vox_meta["duration_ms"] == duration]

        if len(duration_samples) < samples_per_duration:
            print(f"  WARNING: Only {len(duration_samples)} samples available (requested {samples_per_duration})")
            samples_to_test = duration_samples
        else:
            samples_to_test = duration_samples.head(samples_per_duration)

        duration_results = []
        correct_count = 0

        for idx, (_, sample) in enumerate(samples_to_test.iterrows(), 1):
            audio_path = vox_dir / PureWindowsPath(sample["audio_path"]).name

            # Run prediction
            pred = model.predict(audio_path)

            is_correct = (pred.label == "SPEECH")
            if is_correct:
                correct_count += 1

            duration_results.append({
                'duration_ms': duration,
                'sample_idx': idx,
                'audio_path': sample['audio_path'],
                'predicted': pred.label,
                'ground_truth': 'SPEECH',
                'correct': is_correct,
                'confidence': pred.confidence,
                'raw_output': pred.raw_output,
                'latency_ms': pred.latency_ms,
            })

            # Progress indicator every 10 samples
            if idx % 10 == 0:
                current_accuracy = (correct_count / idx) * 100
                print(f"  Progress: {idx}/{len(samples_to_test)} | Accuracy so far: {current_accuracy:.0f}%")

        # Calculate statistics for this duration
        accuracy = (correct_count / len(duration_results)) * 100
        avg_latency = sum(r['latency_ms'] for r in duration_results) / len(duration_results)

        results_by_duration[duration] = {
            'samples': duration_results,
            'accuracy': accuracy,
            'correct': correct_count,
            'total': len(duration_results),
            'avg_latency_ms': avg_latency,
        }

        all_results.extend(duration_results)

        print(f"  Final: {accuracy:.1f}% ({correct_count}/{len(duration_results)}) | Avg latency: {avg_latency:.0f}ms")

    # Calculate overall statistics
    total_time = time.time() - start_time
    total_correct = sum(r['correct'] for r in results_by_duration.values())
    total_samples = sum(r['total'] for r in results_by_duration.values())
    overall_accuracy = (total_correct / total_samples) * 100
    avg_time_per_sample = total_time / total_samples

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}\n")

    print(f"{'Duration':<12} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Latency'}")
    print("-"*65)

    for duration in test_durations:
        r = results_by_duration[duration]
        print(f"{duration}ms{' '*7} {r['accuracy']:>6.1f}%{' '*5} "
              f"{r['correct']:>3}/{r['total']:<10} {r['avg_latency_ms']:>7.0f}ms")

    print("-"*65)
    print(f"{'OVERALL':<12} {overall_accuracy:>6.1f}%{' '*5} "
          f"{total_correct:>3}/{total_samples:<10} {sum(r['avg_latency_ms'] for r in results_by_duration.values())/len(results_by_duration):>7.0f}ms")

    # Statistical analysis
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*80}\n")

    # Identify performance tiers
    perfect_durations = [d for d, r in results_by_duration.items() if r['accuracy'] >= 95]
    good_durations = [d for d, r in results_by_duration.items() if 80 <= r['accuracy'] < 95]
    partial_durations = [d for d, r in results_by_duration.items() if 60 <= r['accuracy'] < 80]
    poor_durations = [d for d, r in results_by_duration.items() if r['accuracy'] < 60]

    if perfect_durations:
        print(f"Excellent (>=95%):  {', '.join(str(d) + 'ms' for d in perfect_durations)}")
    if good_durations:
        print(f"Good (80-94%):      {', '.join(str(d) + 'ms' for d in good_durations)}")
    if partial_durations:
        print(f"Partial (60-79%):   {', '.join(str(d) + 'ms' for d in partial_durations)}")
    if poor_durations:
        print(f"Poor (<60%):        {', '.join(str(d) + 'ms' for d in poor_durations)}")

    # Find minimum reliable duration
    reliable_threshold = 90  # 90% accuracy threshold
    reliable_durations = [d for d, r in results_by_duration.items() if r['accuracy'] >= reliable_threshold]

    if reliable_durations:
        min_reliable = min(reliable_durations)
        print(f"\nMinimum reliable duration (>={reliable_threshold}% accuracy): {min_reliable}ms")
    else:
        min_reliable = None
        print(f"\nNo duration achieved >={reliable_threshold}% accuracy")

    # Performance metrics
    print(f"\nPerformance metrics:")
    print(f"  Total evaluation time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"  Average time per sample: {avg_time_per_sample:.2f}s")
    print(f"  Samples per minute: {60/avg_time_per_sample:.1f}")

    # Compare with previous small-scale test
    print(f"\n{'='*80}")
    print("COMPARISON WITH SMALL-SCALE TEST (21 samples)")
    print(f"{'='*80}\n")

    print("Previous results (3 samples per duration):")
    previous = {
        20: 67, 40: 67, 60: 67, 80: 100, 100: 100, 200: 67, 500: 100, 1000: 100
    }

    print(f"\n{'Duration':<12} {'Previous (n=3)':<18} {'Current (n=30)':<18} {'Difference'}")
    print("-"*65)

    for duration in test_durations:
        prev_acc = previous.get(duration, 0)
        curr_acc = results_by_duration[duration]['accuracy']
        diff = curr_acc - prev_acc
        diff_str = f"{diff:+.1f}%"

        print(f"{duration}ms{' '*7} {prev_acc:>5.0f}%{' '*12} {curr_acc:>5.1f}%{' '*12} {diff_str}")

    # Conclusions
    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print(f"{'='*80}\n")

    if overall_accuracy >= 80:
        print(f"SUCCESS: Overall accuracy of {overall_accuracy:.1f}% validates padding strategy.")
    elif overall_accuracy >= 70:
        print(f"PARTIAL SUCCESS: {overall_accuracy:.1f}% accuracy shows promise but needs refinement.")
    else:
        print(f"CONCERN: {overall_accuracy:.1f}% accuracy lower than expected.")

    if min_reliable is not None:
        if min_reliable in [80, 100]:
            print(f"Minimum reliable threshold confirmed at {min_reliable}ms (consistent with small test).")
        else:
            print(f"Minimum reliable threshold differs from small test: {min_reliable}ms vs 80ms.")
    else:
        print(f"No reliable threshold found at {reliable_threshold}% level.")

    # Save detailed results
    results_df = pd.DataFrame(all_results)
    output_path = Path("results/qwen_extended_evaluation_with_padding.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path, index=False)

    print(f"\nDetailed results saved to: {output_path}")

    # Generate summary statistics file
    summary = {
        'duration_ms': test_durations,
        'accuracy': [results_by_duration[d]['accuracy'] for d in test_durations],
        'correct': [results_by_duration[d]['correct'] for d in test_durations],
        'total': [results_by_duration[d]['total'] for d in test_durations],
        'avg_latency_ms': [results_by_duration[d]['avg_latency_ms'] for d in test_durations],
    }
    summary_df = pd.DataFrame(summary)
    summary_path = Path("results/qwen_extended_summary.parquet")
    summary_df.to_parquet(summary_path, index=False)

    print(f"Summary statistics saved to: {summary_path}")

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
