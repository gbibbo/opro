#!/usr/bin/env python3
"""
Test SNR accuracy: verify that generated SNR matches target.

Samples random SNR variants and computes actual SNR from audio.

Usage:
    python scripts/test_snr_accuracy.py
"""

import argparse
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import Path
import random


def compute_rms(audio):
    """Compute RMS energy."""
    return np.sqrt(np.mean(audio ** 2))


def estimate_snr(noisy_audio, clean_audio):
    """
    Estimate SNR from noisy and clean audio.

    Args:
        noisy_audio: Audio with noise
        clean_audio: Original clean audio

    Returns:
        Estimated SNR in dB
    """
    # Estimate noise as difference
    noise = noisy_audio - clean_audio

    rms_signal = compute_rms(clean_audio)
    rms_noise = compute_rms(noise)

    if rms_noise < 1e-10:
        return float('inf')

    snr = 20 * np.log10(rms_signal / rms_noise)
    return snr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/conditions/conditions_manifest.parquet"))
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to test")
    args = parser.parse_args()

    # Load manifest
    df = pd.read_parquet(args.manifest)

    # Filter SNR variants only
    df_snr = df[df['variant_type'] == 'snr'].copy()
    print(f"Testing {args.n_samples} random SNR variants...")
    print()

    # Sample random variants
    samples = df_snr.sample(n=min(args.n_samples, len(df_snr)))

    errors = []
    for idx, row in samples.iterrows():
        target_snr = row['snr_db']

        # Load noisy audio
        noisy_audio, sr = sf.read(row['audio_path'])

        # Load original (clean) audio
        clean_audio, _ = sf.read(row['original_path'])

        # Estimate SNR from effective segment
        duration_ms = row['duration_ms']
        total_samples = len(noisy_audio)
        effective_samples = int(sr * duration_ms / 1000.0)
        start_idx = (total_samples - effective_samples) // 2
        end_idx = start_idx + effective_samples

        noisy_seg = noisy_audio[start_idx:end_idx]
        clean_seg = clean_audio[start_idx:end_idx]

        # Check for silent segment
        if row.get('silent_segment', False):
            print(f"⚠ SKIP (silent): {Path(row['audio_path']).name}")
            print(f"   Target SNR: {target_snr:.1f} dB (undefined for silent)")
            print()
            continue

        actual_snr = estimate_snr(noisy_seg, clean_seg)
        error = actual_snr - target_snr
        errors.append(error)

        status = "✓" if abs(error) < 0.5 else "✗"
        print(f"{status} {Path(row['audio_path']).name}")
        print(f"   Target SNR: {target_snr:+.1f} dB")
        print(f"   Actual SNR: {actual_snr:+.1f} dB")
        print(f"   Error: {error:+.2f} dB")
        print()

    # Summary
    if len(errors) > 0:
        print("="*60)
        print(f"Summary (n={len(errors)} non-silent samples):")
        print(f"  Mean error: {np.mean(errors):+.3f} dB")
        print(f"  Std error:  {np.std(errors):.3f} dB")
        print(f"  Max |error|: {np.max(np.abs(errors)):.3f} dB")

        if np.max(np.abs(errors)) < 0.5:
            print("\n✅ SNR accuracy test PASSED (all errors < 0.5 dB)")
        else:
            print("\n⚠ SNR accuracy test WARNING (some errors > 0.5 dB)")
    else:
        print("No non-silent samples to test!")


if __name__ == "__main__":
    main()
