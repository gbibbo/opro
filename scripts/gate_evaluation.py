"""
Gate Evaluation - Step 1: Verify readiness for fine-tuning.

This script performs a comprehensive evaluation of the best prompt from optimization:
1. Evaluate on full dev set (1,400 samples) with and without constrained decoding
2. Compute psychometric curves (DT75, SNR-75) by duration
3. Analyze robustness across SNR and duration conditions
4. Generate comparison report vs baseline

Gate criteria for proceeding to fine-tuning:
- BA_clip improvement >= +2-3% on dev set
- Constrained decoding maintains or improves performance
- Monotonic relationship with duration
- Reasonable performance across SNR conditions
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.qsm.models.qwen_audio import PredictionResult, Qwen2AudioClassifier


def resolve_audio_path(raw_path: str, project_root: Path) -> str:
    """Convert relative path to absolute path."""
    p = Path(str(raw_path))
    if not p.is_absolute():
        # Path is relative to project root, not script dir
        p = (project_root / p).resolve()

    return str(p)


def compute_clip_level_ba(results_df: pd.DataFrame) -> float:
    """Compute balanced accuracy at clip level with majority voting."""
    if len(results_df) == 0:
        return 0.0

    clip_predictions = []

    for clip_id in results_df['clip_id'].unique():
        clip_rows = results_df[results_df['clip_id'] == clip_id]

        # Majority vote
        speech_votes = (clip_rows['y_pred'] == 1).sum()
        nonspeech_votes = (clip_rows['y_pred'] == 0).sum()

        y_true_clip = clip_rows['y_true'].iloc[0]
        y_pred_clip = 1 if speech_votes > nonspeech_votes else 0

        clip_predictions.append({
            'clip_id': clip_id,
            'y_true': y_true_clip,
            'y_pred': y_pred_clip
        })

    clip_df = pd.DataFrame(clip_predictions)
    return balanced_accuracy_score(clip_df['y_true'], clip_df['y_pred'])


def compute_psychometric_curve(results_df: pd.DataFrame,
                                 condition_col: str,
                                 n_bootstrap: int = 1000) -> dict:
    """
    Compute psychometric curve with threshold estimation.

    Args:
        results_df: DataFrame with predictions and condition values
        condition_col: Column name ('duration_ms' or 'snr_db')
        n_bootstrap: Number of bootstrap samples for confidence intervals

    Returns:
        Dictionary with threshold, curve points, and confidence intervals
    """
    # Get unique condition values
    condition_values = sorted(results_df[condition_col].dropna().unique())

    # Compute accuracy at each condition level (clip-level with majority voting)
    accuracies = []
    ci_lower = []
    ci_upper = []

    for val in condition_values:
        subset = results_df[results_df[condition_col] == val]

        # Clip-level accuracy
        clip_preds = []
        for clip_id in subset['clip_id'].unique():
            clip_rows = subset[subset['clip_id'] == clip_id]
            speech_votes = (clip_rows['y_pred'] == 1).sum()
            nonspeech_votes = (clip_rows['y_pred'] == 0).sum()

            y_true = clip_rows['y_true'].iloc[0]
            y_pred = 1 if speech_votes > nonspeech_votes else 0

            clip_preds.append(int(y_true == y_pred))

        acc = np.mean(clip_preds)
        accuracies.append(acc)

        # Bootstrap CI
        bootstrap_accs = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(clip_preds, size=len(clip_preds), replace=True)
            bootstrap_accs.append(np.mean(sample))

        ci_lower.append(np.percentile(bootstrap_accs, 2.5))
        ci_upper.append(np.percentile(bootstrap_accs, 97.5))

    # Estimate threshold (e.g., DT75 or SNR-75)
    threshold = None
    if len(condition_values) > 3:
        # Linear interpolation to find 75% accuracy point
        try:
            threshold = np.interp(0.75, accuracies, condition_values)
        except:
            pass

    return {
        'condition_values': condition_values,
        'accuracies': accuracies,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'threshold_75': threshold
    }


def evaluate_prompt(prompt: str,
                     manifest_path: Path,
                     constrained: bool = False,
                     max_samples: int = None) -> pd.DataFrame:
    """
    Evaluate a prompt on dataset with optional constrained decoding.

    Returns DataFrame with per-segment predictions and metadata.
    """
    print(f"\n{'='*80}")
    print(f"Evaluating prompt: {prompt[:60]}...")
    print(f"Constrained decoding: {constrained}")
    print(f"{'='*80}\n")

    # Load manifest
    manifest_df = pd.read_csv(manifest_path)
    print(f"Loaded {len(manifest_df)} samples from manifest")

    if max_samples:
        manifest_df = manifest_df.head(max_samples)
        print(f"Limited to {max_samples} samples for testing")

    # Initialize model
    print("Loading Qwen2-Audio model...")
    model = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda",
        load_in_4bit=True,
        constrained_decoding=constrained
    )

    # Set the prompt
    model.set_prompt(user_prompt=prompt)

    # Evaluate each segment
    results = []
    errors = 0

    start_time = time.time()

    for idx, row in manifest_df.iterrows():
        # Resolve audio path
        audio_path = resolve_audio_path(row['audio_path'], project_root)

        # Get ground truth label (handle both 'label' and 'ground_truth' columns)
        label_col = 'ground_truth' if 'ground_truth' in row else 'label'
        label_str = str(row[label_col]).strip().upper()
        if label_str == "SPEECH":
            y_true = 1
        elif label_str in ["NON-SPEECH", "NONSPEECH"]:
            y_true = 0
        else:
            print(f"[WARN] Unknown label '{row[label_col]}' at row {idx}, skipping")
            continue

        # Predict
        try:
            result = model.predict(audio_path)

            if result.label == "SPEECH":
                y_pred = 1
            elif result.label == "NONSPEECH":
                y_pred = 0
            else:
                y_pred = -1  # Unknown
                errors += 1

            # Handle both metadata formats
            clip_id = row.get('clip_id', row.get('variant_name', idx))
            variant_type = row.get('variant_type', 'unknown')

            results.append({
                'segment_id': idx,
                'clip_id': clip_id,
                'audio_path': audio_path,
                'y_true': y_true,
                'y_pred': y_pred,
                'confidence': result.confidence,
                'label': label_str,
                'pred_label': result.label,
                'duration_ms': row.get('duration_ms', np.nan),
                'snr_db': row.get('snr_db', np.nan),
                'variant_type': variant_type
            })

            # Progress update
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                remaining = (len(manifest_df) - idx - 1) / rate / 60
                print(f"Progress: {idx+1}/{len(manifest_df)} | "
                      f"Rate: {rate:.1f} samples/s | "
                      f"ETA: {remaining:.1f} min | "
                      f"Errors: {errors}")

        except Exception as e:
            print(f"[ERROR] Failed on segment {idx}: {e}")
            errors += 1
            continue

    elapsed = time.time() - start_time
    print(f"\n{'-'*80}")
    print(f"Evaluation completed in {elapsed/60:.1f} minutes")
    print(f"Total errors: {errors} ({errors/len(manifest_df)*100:.1f}%)")
    print(f"{'-'*80}\n")

    # Clean up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)  # Give CUDA time to release memory

    return pd.DataFrame(results)


def analyze_robustness(results_df: pd.DataFrame) -> dict:
    """
    Analyze performance robustness across conditions.

    Returns dictionary with:
    - Overall BA_clip
    - BA by variant_type
    - Psychometric curves for duration and SNR
    - Confusion matrix
    """
    print("\n" + "="*80)
    print("ROBUSTNESS ANALYSIS")
    print("="*80 + "\n")

    # Overall metrics
    ba_clip = compute_clip_level_ba(results_df)
    print(f"Overall BA_clip: {ba_clip:.4f}")

    # Confusion matrix (segment-level)
    valid_preds = results_df[results_df['y_pred'] != -1]
    cm = confusion_matrix(valid_preds['y_true'], valid_preds['y_pred'])
    print(f"\nConfusion Matrix (segment-level):")
    print(f"                Pred NONSPEECH  Pred SPEECH")
    print(f"True NONSPEECH:      {cm[0,0]:6d}       {cm[0,1]:6d}")
    print(f"True SPEECH:         {cm[1,0]:6d}       {cm[1,1]:6d}")

    # BA by variant type
    print(f"\n{'Variant Type':<20} {'N clips':<10} {'BA_clip':<10}")
    print("-" * 40)

    ba_by_variant = {}
    for variant in results_df['variant_type'].unique():
        subset = results_df[results_df['variant_type'] == variant]
        if len(subset) > 0:
            ba = compute_clip_level_ba(subset)
            n_clips = subset['clip_id'].nunique()
            ba_by_variant[variant] = ba
            print(f"{variant:<20} {n_clips:<10} {ba:.4f}")

    # Psychometric curves
    print("\n" + "-"*80)
    print("Computing psychometric curves...")
    print("-"*80)

    # Duration curve (only for duration variant)
    duration_subset = results_df[results_df['variant_type'] == 'duration']
    duration_curve = None
    if len(duration_subset) > 50:
        duration_curve = compute_psychometric_curve(duration_subset, 'duration_ms')
        if duration_curve['threshold_75']:
            print(f"DT75 (75% accuracy threshold): {duration_curve['threshold_75']:.1f} ms")

    # SNR curves by duration bin
    snr_subset = results_df[results_df['variant_type'] == 'snr']
    snr_curves = {}
    if len(snr_subset) > 50:
        # Bin durations
        duration_bins = [0, 100, 250, 500, 1000, 2000]
        for i in range(len(duration_bins)-1):
            bin_name = f"{duration_bins[i]}-{duration_bins[i+1]}ms"
            bin_subset = snr_subset[
                (snr_subset['duration_ms'] > duration_bins[i]) &
                (snr_subset['duration_ms'] <= duration_bins[i+1])
            ]
            if len(bin_subset) > 20:
                curve = compute_psychometric_curve(bin_subset, 'snr_db')
                snr_curves[bin_name] = curve
                if curve['threshold_75']:
                    print(f"SNR-75 ({bin_name}): {curve['threshold_75']:.1f} dB")

    return {
        'ba_clip': ba_clip,
        'ba_by_variant': ba_by_variant,
        'confusion_matrix': cm,
        'duration_curve': duration_curve,
        'snr_curves': snr_curves,
        'n_errors': len(results_df[results_df['y_pred'] == -1])
    }


def generate_gate_report(baseline_results: dict,
                          optimized_results: dict,
                          optimized_constrained_results: dict,
                          output_path: Path):
    """Generate comprehensive gate evaluation report."""

    report = []
    report.append("="*80)
    report.append("GATE EVALUATION REPORT - FINE-TUNING READINESS")
    report.append("="*80)
    report.append("")

    # Summary
    baseline_ba = baseline_results['ba_clip']
    optimized_ba = optimized_results['ba_clip']
    constrained_ba = optimized_constrained_results['ba_clip']

    improvement = (optimized_ba - baseline_ba) * 100
    constrained_delta = (constrained_ba - optimized_ba) * 100

    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Baseline BA_clip:               {baseline_ba:.4f}")
    report.append(f"Optimized BA_clip:              {optimized_ba:.4f}")
    report.append(f"Optimized + Constrained BA_clip: {constrained_ba:.4f}")
    report.append("")
    report.append(f"Improvement over baseline:      {improvement:+.2f}%")
    report.append(f"Constrained decoding impact:    {constrained_delta:+.2f}%")
    report.append("")

    # Gate criteria
    report.append("GATE CRITERIA")
    report.append("-" * 80)

    gate_pass = True

    # Criterion 1: >= +2% improvement
    criterion_1 = improvement >= 2.0
    report.append(f"[{'PASS' if criterion_1 else 'FAIL'}] 1. BA_clip improvement >= +2%: {improvement:+.2f}%")
    gate_pass = gate_pass and criterion_1

    # Criterion 2: Constrained decoding doesn't hurt
    criterion_2 = constrained_delta >= -1.0
    report.append(f"[{'PASS' if criterion_2 else 'FAIL'}] 2. Constrained decoding delta >= -1%: {constrained_delta:+.2f}%")
    gate_pass = gate_pass and criterion_2

    # Criterion 3: Low error rate (calculate from total samples in results)
    n_total_samples = len(baseline_results.get('confusion_matrix', [[0, 0], [0, 0]])[0]) + len(baseline_results.get('confusion_matrix', [[0, 0], [0, 0]])[1]) + optimized_results['n_errors']
    error_rate = optimized_results['n_errors'] / max(n_total_samples, 1) * 100
    criterion_3 = error_rate < 5.0
    report.append(f"[{'PASS' if criterion_3 else 'FAIL'}] 3. Error rate < 5%: {error_rate:.2f}%")
    gate_pass = gate_pass and criterion_3

    # Criterion 4: Monotonic duration relationship
    if optimized_results['duration_curve']:
        accs = optimized_results['duration_curve']['accuracies']
        # Check if mostly increasing
        increasing_pairs = sum([accs[i+1] >= accs[i] for i in range(len(accs)-1)])
        monotonic_ratio = increasing_pairs / (len(accs) - 1)
        criterion_4 = monotonic_ratio >= 0.7
        report.append(f"[{'PASS' if criterion_4 else 'FAIL'}] 4. Monotonic duration relationship: {monotonic_ratio:.1%}")
        gate_pass = gate_pass and criterion_4

    report.append("")
    report.append("="*80)
    report.append(f"GATE DECISION: {'PASS - PROCEED TO FINE-TUNING' if gate_pass else 'FAIL - IMPROVE PROMPTS FIRST'}")
    report.append("="*80)
    report.append("")

    # Detailed psychometric results
    report.append("PSYCHOMETRIC THRESHOLDS")
    report.append("-" * 80)

    if baseline_results['duration_curve'] and baseline_results['duration_curve']['threshold_75']:
        report.append(f"Baseline DT75:   {baseline_results['duration_curve']['threshold_75']:.1f} ms")
    if optimized_results['duration_curve'] and optimized_results['duration_curve']['threshold_75']:
        report.append(f"Optimized DT75:  {optimized_results['duration_curve']['threshold_75']:.1f} ms")

    report.append("")

    # Performance by condition
    report.append("PERFORMANCE BY VARIANT TYPE")
    report.append("-" * 80)
    report.append(f"{'Variant':<15} {'Baseline':<12} {'Optimized':<12} {'Delta':<10}")
    report.append("-" * 80)

    for variant in baseline_results['ba_by_variant'].keys():
        base_ba = baseline_results['ba_by_variant'].get(variant, 0)
        opt_ba = optimized_results['ba_by_variant'].get(variant, 0)
        delta = (opt_ba - base_ba) * 100
        report.append(f"{variant:<15} {base_ba:.4f}      {opt_ba:.4f}      {delta:+.2f}%")

    report.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print('\n'.join(report))
    print(f"\nReport saved to: {output_path}")

    return gate_pass


def main():
    """Main gate evaluation workflow."""

    # Paths
    dev_manifest = project_root / "data" / "processed" / "snr_duration_crossed" / "metadata.csv"
    output_dir = project_root / "results" / "gate_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prompts
    baseline_prompt = "Is this audio clip SPEECH or NON-SPEECH?"
    optimized_prompt = "Based on the audio file, is it SPEECH or NON-SPEECH?"

    print("\n" + "="*80)
    print("GATE EVALUATION - FINE-TUNING READINESS CHECK")
    print("="*80)
    print(f"\nBaseline prompt: {baseline_prompt}")
    print(f"Optimized prompt: {optimized_prompt}")
    print(f"\nDev set: {dev_manifest}")
    print(f"Output directory: {output_dir}")

    # Step 1: Evaluate baseline (unconstrained)
    print("\n\n" + "="*80)
    print("STEP 1/3: Baseline Prompt (Unconstrained)")
    print("="*80)

    baseline_df = evaluate_prompt(baseline_prompt, dev_manifest)
    baseline_df.to_csv(output_dir / "baseline_predictions.csv", index=False)
    baseline_results = analyze_robustness(baseline_df)

    # Step 2: Evaluate optimized (unconstrained)
    print("\n\n" + "="*80)
    print("STEP 2/3: Optimized Prompt (Unconstrained)")
    print("="*80)

    optimized_df = evaluate_prompt(optimized_prompt, dev_manifest)
    optimized_df.to_csv(output_dir / "optimized_predictions.csv", index=False)
    optimized_results = analyze_robustness(optimized_df)

    # Step 3: Evaluate optimized (constrained)
    print("\n\n" + "="*80)
    print("STEP 3/3: Optimized Prompt (Constrained Decoding)")
    print("="*80)

    optimized_constrained_df = evaluate_prompt(optimized_prompt, dev_manifest, constrained=True)
    optimized_constrained_df.to_csv(output_dir / "optimized_constrained_predictions.csv", index=False)
    optimized_constrained_results = analyze_robustness(optimized_constrained_df)

    # Generate gate report
    print("\n\n" + "="*80)
    print("GENERATING GATE REPORT")
    print("="*80)

    gate_pass = generate_gate_report(
        baseline_results,
        optimized_results,
        optimized_constrained_results,
        output_dir / "gate_report.txt"
    )

    # Final recommendation
    print("\n" + "="*80)
    if gate_pass:
        print("RECOMMENDATION: Proceed to fine-tuning with LoRA/QLoRA")
        print("Next steps:")
        print("  1. Prepare fine-tuning dataset (train/val/test splits)")
        print("  2. Set up PEFT with LoRA config (r=8-16, alpha=16-32)")
        print("  3. Train with constrained decoding")
        print("  4. Re-run psychometric evaluation on test set")
    else:
        print("RECOMMENDATION: Improve prompts before fine-tuning")
        print("Suggested actions:")
        print("  1. Run more optimization iterations")
        print("  2. Explore canonical templates")
        print("  3. Investigate failure modes by condition")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
