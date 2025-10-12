#!/usr/bin/env python3
"""
Analyze SNR samples to understand why SNR=0dB performs poorly.

Measures actual SNR in generated audio files and compares with expected values.
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS energy."""
    return np.sqrt(np.mean(audio**2))


def estimate_snr_from_audio(audio_path: Path, effective_start_ms: float = 500, effective_dur_ms: float = 1000, sr: int = 16000):
    """
    Estimate SNR from an audio file by comparing effective segment vs padding.

    Assumes:
    - Audio is 2000ms total (padded)
    - Effective segment (1000ms) is centered
    - Padding regions contain noise

    Returns:
        dict with estimated SNR and RMS values
    """
    audio, file_sr = sf.read(audio_path, dtype='float32')

    if file_sr != sr:
        print(f"Warning: {audio_path.name} has sr={file_sr}, expected {sr}")
        return None

    # Calculate sample indices
    total_samples = len(audio)
    effective_samples = int(sr * effective_dur_ms / 1000.0)
    effective_start = (total_samples - effective_samples) // 2
    effective_end = effective_start + effective_samples

    # Extract regions
    effective_segment = audio[effective_start:effective_end]
    padding_left = audio[:effective_start]
    padding_right = audio[effective_end:]
    padding = np.concatenate([padding_left, padding_right])

    # Compute RMS
    rms_total = compute_rms(audio)
    rms_effective = compute_rms(effective_segment)
    rms_padding = compute_rms(padding)

    # Estimate noise from padding (assumes padding is mostly noise)
    rms_noise_estimate = rms_padding

    # Estimate SNR
    if rms_noise_estimate > 1e-8:
        snr_estimate = 20 * np.log10(rms_effective / rms_noise_estimate)
    else:
        snr_estimate = np.inf

    return {
        'rms_total': rms_total,
        'rms_effective': rms_effective,
        'rms_padding': rms_padding,
        'rms_noise_estimate': rms_noise_estimate,
        'snr_estimate_db': snr_estimate,
    }


def main():
    # Load debug results
    results_path = Path("results/debug_2clips_v2/debug_results.parquet")

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Please run the debug evaluation first:")
        print("  python scripts/debug_evaluate.py --n_clips 2 --output_dir results/debug_2clips_v2")
        return

    df = pd.read_parquet(results_path)

    # Filter SNR variants only
    snr_df = df[df['variant_type'] == 'snr'].copy()

    if len(snr_df) == 0:
        print("No SNR samples found in results")
        return

    print("=" * 80)
    print("SNR SAMPLE ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing {len(snr_df)} SNR samples...")
    print()

    # Analyze each SNR sample
    analysis_results = []

    for idx, row in tqdm(snr_df.iterrows(), total=len(snr_df), desc="Analyzing"):
        # Convert Windows backslashes to forward slashes for cross-platform compatibility
        audio_path_str = str(row['audio_path']).replace('\\', '/')
        audio_path = Path(audio_path_str)

        if not audio_path.exists():
            print(f"\nWarning: Audio not found: {audio_path}")
            continue

        # Analyze audio
        analysis = estimate_snr_from_audio(audio_path)

        if analysis is None:
            continue

        # Combine with metadata
        result = {
            'audio_filename': audio_path.name,
            'clip_id': row['clip_id'],
            'ground_truth': row['ground_truth'],
            'predicted': row['predicted'],
            'correct': row['correct'],
            'expected_snr_db': row['snr_db'],
            **analysis,
            'snr_error_db': analysis['snr_estimate_db'] - row['snr_db'],
        }

        analysis_results.append(result)

    # Create DataFrame
    analysis_df = pd.DataFrame(analysis_results)

    # Save results
    output_path = Path("results/debug_2clips_v2/snr_analysis.csv")
    analysis_df.to_csv(output_path, index=False)
    print(f"\nSaved analysis to: {output_path}")

    # Check if we have any results
    if len(analysis_df) == 0:
        print("\n" + "=" * 80)
        print("ERROR: No audio files were successfully analyzed!")
        print("=" * 80)
        print("\nPossible causes:")
        print("  1. Audio files not found at expected paths")
        print("  2. Path format issues (Windows vs Linux)")
        print("  3. No SNR samples in the results")
        print("\nPlease check:")
        print("  - That audio files exist in data/processed/conditions_final/")
        print("  - That debug_evaluate.py ran successfully")
        return

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY BY EXPECTED SNR")
    print("=" * 80)
    print()

    for expected_snr in sorted(analysis_df['expected_snr_db'].unique()):
        subset = analysis_df[analysis_df['expected_snr_db'] == expected_snr]

        print(f"Expected SNR: {expected_snr:+.0f} dB ({len(subset)} samples)")
        print(f"  Measured SNR: {subset['snr_estimate_db'].mean():+.1f} ± {subset['snr_estimate_db'].std():.1f} dB")
        print(f"  SNR Error: {subset['snr_error_db'].mean():+.1f} ± {subset['snr_error_db'].std():.1f} dB")
        print(f"  Accuracy: {subset['correct'].mean() * 100:.1f}%")
        print(f"  RMS Effective: {subset['rms_effective'].mean():.6f}")
        print(f"  RMS Noise: {subset['rms_noise_estimate'].mean():.6f}")
        print()

    # Identify problematic samples
    print("=" * 80)
    print("PROBLEMATIC SAMPLES (large SNR error or incorrect)")
    print("=" * 80)
    print()

    problematic = analysis_df[
        (np.abs(analysis_df['snr_error_db']) > 2.0) | (~analysis_df['correct'])
    ].sort_values('snr_error_db', key=lambda x: np.abs(x), ascending=False)

    if len(problematic) > 0:
        print(f"Found {len(problematic)} problematic samples:")
        print()
        for idx, row in problematic.iterrows():
            print(f"File: {row['audio_filename']}")
            print(f"  Ground truth: {row['ground_truth']}")
            print(f"  Predicted: {row['predicted']} ({'CORRECT' if row['correct'] else 'INCORRECT'})")
            print(f"  Expected SNR: {row['expected_snr_db']:+.1f} dB")
            print(f"  Measured SNR: {row['snr_estimate_db']:+.1f} dB")
            print(f"  Error: {row['snr_error_db']:+.1f} dB")
            print(f"  RMS Effective: {row['rms_effective']:.6f}")
            print(f"  RMS Noise: {row['rms_noise_estimate']:.6f}")
            print()
    else:
        print("No problematic samples found (all within ±2dB and correct predictions)")

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Check the following files:")
    print(f"  - Analysis CSV: {output_path}")
    print(f"  - Debug results: {results_path}")
    print()


if __name__ == "__main__":
    main()
