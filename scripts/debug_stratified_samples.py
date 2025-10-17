"""Debug script with stratified sampling: easy, medium, hard conditions."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.qsm.models.qwen_audio import Qwen2AudioClassifier


def select_stratified_samples(df: pd.DataFrame, n_per_stratum: int = 2) -> pd.DataFrame:
    """
    Select samples stratified by difficulty (duration x SNR).

    Strata:
    - EASY: duration >= 500ms, SNR >= +10dB
    - MEDIUM: duration >= 200ms, SNR >= 0dB
    - HARD: duration < 100ms, SNR < -5dB
    """
    samples = []

    # EASY conditions (should get ~100% accuracy)
    easy = df[
        (df['duration_ms'] >= 500) &
        (df['snr_db'] >= 10)
    ]
    if len(easy) > 0:
        speech_easy = easy[easy['ground_truth'] == 'SPEECH'].head(n_per_stratum)
        nonspeech_easy = easy[easy['ground_truth'].str.upper().isin(['NONSPEECH', 'NON-SPEECH'])].head(n_per_stratum)
        samples.append(speech_easy)
        samples.append(nonspeech_easy)

    # MEDIUM conditions (should get ~70-80% accuracy)
    medium = df[
        (df['duration_ms'] >= 200) &
        (df['duration_ms'] < 500) &
        (df['snr_db'] >= 0) &
        (df['snr_db'] < 10)
    ]
    if len(medium) > 0:
        speech_med = medium[medium['ground_truth'] == 'SPEECH'].head(n_per_stratum)
        nonspeech_med = medium[medium['ground_truth'].str.upper().isin(['NONSPEECH', 'NON-SPEECH'])].head(n_per_stratum)
        samples.append(speech_med)
        samples.append(nonspeech_med)

    # HARD conditions (expected ~50-60% accuracy)
    hard = df[
        (df['duration_ms'] < 100) &
        (df['snr_db'] < -5)
    ]
    if len(hard) > 0:
        speech_hard = hard[hard['ground_truth'] == 'SPEECH'].head(n_per_stratum)
        nonspeech_hard = hard[hard['ground_truth'].str.upper().isin(['NONSPEECH', 'NON-SPEECH'])].head(n_per_stratum)
        samples.append(speech_hard)
        samples.append(nonspeech_hard)

    return pd.concat(samples, ignore_index=True)


def main():
    """Stratified smoke test."""

    # Load metadata
    metadata_path = project_root / "data" / "processed" / "snr_duration_crossed" / "metadata.csv"
    df = pd.read_csv(metadata_path)

    print("="*80)
    print("STRATIFIED SMOKE TEST - Audio Format & Parser Verification")
    print("="*80)
    print(f"\nTotal samples in dataset: {len(df)}")
    print(f"SPEECH samples: {(df['ground_truth'] == 'SPEECH').sum()}")
    print(f"NONSPEECH samples: {(df['ground_truth'].str.upper().isin(['NONSPEECH', 'NON-SPEECH'])).sum()}")

    # Select stratified samples
    test_df = select_stratified_samples(df, n_per_stratum=3)

    print(f"\nSelected {len(test_df)} stratified samples:")
    print(f"  EASY (>=500ms, >=+10dB): {len(test_df[(test_df['duration_ms']>=500) & (test_df['snr_db']>=10)])}")
    print(f"  MEDIUM (200-500ms, 0-10dB): {len(test_df[(test_df['duration_ms']>=200) & (test_df['duration_ms']<500) & (test_df['snr_db']>=0) & (test_df['snr_db']<10)])}")
    print(f"  HARD (<100ms, <-5dB): {len(test_df[(test_df['duration_ms']<100) & (test_df['snr_db']<-5)])}")

    # Test both prompts and both decoding modes
    configs = [
        ("Baseline", "Is this audio clip SPEECH or NON-SPEECH?", False),
        ("Optimized", "Based on the audio file, is it SPEECH or NON-SPEECH?", False),
        ("Optimized+Constrained", "Based on the audio file, is it SPEECH or NON-SPEECH?", True),
    ]

    results_summary = []

    for config_name, prompt_text, constrained in configs:
        print(f"\n{'='*80}")
        print(f"CONFIG: {config_name}")
        print(f"Prompt: {prompt_text}")
        print(f"Constrained: {constrained}")
        print(f"{'='*80}\n")

        # Load model
        print("Loading model...")
        model = Qwen2AudioClassifier(
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
            device="cuda",
            load_in_4bit=True,
            constrained_decoding=constrained
        )

        # Set prompt
        model.set_prompt(user_prompt=prompt_text)

        # Test on samples
        correct_by_stratum = {
            'EASY': {'correct': 0, 'total': 0},
            'MEDIUM': {'correct': 0, 'total': 0},
            'HARD': {'correct': 0, 'total': 0}
        }

        all_results = []

        for idx, row in test_df.iterrows():
            audio_path = project_root / row['audio_path']
            true_label = row['ground_truth'].strip().upper()
            if true_label not in ['SPEECH', 'NONSPEECH', 'NON-SPEECH']:
                continue

            if true_label in ['NONSPEECH', 'NON-SPEECH']:
                true_label = 'NONSPEECH'

            # Determine stratum
            if row['duration_ms'] >= 500 and row['snr_db'] >= 10:
                stratum = 'EASY'
            elif row['duration_ms'] >= 200 and row['duration_ms'] < 500 and row['snr_db'] >= 0 and row['snr_db'] < 10:
                stratum = 'MEDIUM'
            elif row['duration_ms'] < 100 and row['snr_db'] < -5:
                stratum = 'HARD'
            else:
                stratum = 'OTHER'

            # Predict
            try:
                result = model.predict(audio_path)

                # Check correctness
                is_correct = result.label == true_label
                if stratum in correct_by_stratum:
                    correct_by_stratum[stratum]['correct'] += int(is_correct)
                    correct_by_stratum[stratum]['total'] += 1

                status = "✓" if is_correct else "✗"

                # Store result
                all_results.append({
                    'stratum': stratum,
                    'duration_ms': row['duration_ms'],
                    'snr_db': row['snr_db'],
                    'true_label': true_label,
                    'pred_label': result.label,
                    'confidence': result.confidence,
                    'correct': is_correct,
                    'raw_output': result.raw_output[:80]
                })

                print(f"[{stratum:6s}] {status} TRUE: {true_label:<10} | PRED: {result.label:<10} | "
                      f"CONF: {result.confidence:.2f} | DUR: {row['duration_ms']:4.0f}ms | SNR: {row['snr_db']:+.0f}dB")
                print(f"          RAW: '{result.raw_output[:100]}'")

            except Exception as e:
                print(f"[ERROR] Failed on {audio_path}: {e}")

        # Summary by stratum
        print(f"\n{'-'*80}")
        print(f"SUMMARY BY STRATUM:")
        print(f"{'-'*80}")

        overall_correct = 0
        overall_total = 0

        for stratum in ['EASY', 'MEDIUM', 'HARD']:
            correct = correct_by_stratum[stratum]['correct']
            total = correct_by_stratum[stratum]['total']
            overall_correct += correct
            overall_total += total

            if total > 0:
                acc = correct / total
                print(f"{stratum:8s}: {correct}/{total} = {acc:.1%}")
            else:
                print(f"{stratum:8s}: No samples")

        overall_acc = overall_correct / overall_total if overall_total > 0 else 0
        print(f"{'OVERALL':8s}: {overall_correct}/{overall_total} = {overall_acc:.1%}")
        print(f"{'-'*80}\n")

        results_summary.append({
            'config': config_name,
            'prompt': prompt_text[:40],
            'constrained': constrained,
            'easy_acc': correct_by_stratum['EASY']['correct'] / max(correct_by_stratum['EASY']['total'], 1),
            'medium_acc': correct_by_stratum['MEDIUM']['correct'] / max(correct_by_stratum['MEDIUM']['total'], 1),
            'hard_acc': correct_by_stratum['HARD']['correct'] / max(correct_by_stratum['HARD']['total'], 1),
            'overall_acc': overall_acc
        })

        # Save detailed results
        results_df = pd.DataFrame(all_results)
        output_path = project_root / "results" / "debug_stratified" / f"{config_name.lower().replace('+', '_')}_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Detailed results saved to: {output_path}")

        # Clean up
        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        print("\nMemory cleaned. Waiting 2 seconds...\n")
        import time
        time.sleep(2)

    # Final summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY - ACCURACY BY STRATUM AND CONFIG")
    print("="*80)

    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_path = project_root / "results" / "debug_stratified" / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)

    best_config = summary_df.loc[summary_df['overall_acc'].idxmax()]
    print(f"\nBest config: {best_config['config']}")
    print(f"  Overall accuracy: {best_config['overall_acc']:.1%}")
    print(f"  EASY accuracy: {best_config['easy_acc']:.1%}")

    if best_config['easy_acc'] < 0.8:
        print("\n⚠️  WARNING: Accuracy on EASY samples < 80%")
        print("   This suggests a problem with:")
        print("   1. Audio input format (missing audio tokens?)")
        print("   2. Parser not recognizing model outputs")
        print("   3. Model not processing audio correctly")
        print("\n   RECOMMENDATION: Check Qwen2-Audio input format before gate evaluation")
    elif best_config['overall_acc'] >= 0.7:
        print("\n✓ Model appears to be working correctly")
        print("  RECOMMENDATION: Proceed with full gate evaluation")
    else:
        print("\n⚠️  Moderate performance - parser may need tuning")
        print("  RECOMMENDATION: Review raw outputs and adjust parser if needed")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
