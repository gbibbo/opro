#!/usr/bin/env python3
"""
Visualize spectrograms of different psychoacoustic conditions.

Shows side-by-side comparison of:
- Original (clean)
- SNR sweep (-10, 0, +10 dB)
- Band-limiting (telephony, LP, HP)

Usage:
    python scripts/visualize_conditions.py
"""

import argparse
import pandas as pd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_spectrogram(audio, sr, nperseg=512):
    """Compute spectrogram using STFT."""
    from scipy import signal
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=nperseg, noverlap=nperseg//2)
    return f, t, 10 * np.log10(Sxx + 1e-10)  # dB scale


def plot_spectrogram(ax, audio, sr, title):
    """Plot spectrogram on given axes."""
    f, t, Sxx = compute_spectrogram(audio, sr)
    im = ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis', vmin=-80, vmax=0)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim([0, sr/2])
    return im


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/conditions/conditions_manifest.parquet"))
    parser.add_argument("--output", type=Path, default=Path("assets/condition_spectrograms.png"))
    parser.add_argument("--duration_ms", type=int, default=200, help="Duration to visualize (ms)")
    parser.add_argument("--label", type=str, default="SPEECH", choices=["SPEECH", "NON-SPEECH"])
    args = parser.parse_args()

    # Load manifest
    df = pd.read_parquet(args.manifest)
    print(f"Loaded {len(df)} conditions")

    # Filter by duration and label
    df_filt = df[(df['duration_ms'] == args.duration_ms) & (df['label'] == args.label)]
    print(f"Filtered to {len(df_filt)} conditions with duration={args.duration_ms}ms, label={args.label}")

    if len(df_filt) == 0:
        print("No samples found!")
        return

    # Get original padded audio (for reference)
    sample_row = df_filt.iloc[0]
    original_path = Path(sample_row['original_path'])
    audio_orig, sr = sf.read(original_path)
    print(f"Sample rate: {sr} Hz")

    # Select conditions to compare
    conditions = []

    # Original (from padded, before conditions)
    conditions.append(("Original (padded)", audio_orig))

    # SNR sweep
    for snr in [-10, 0, 10]:
        row = df_filt[(df_filt['snr_db'] == snr) & (df_filt['variant_type'] == 'snr')]
        if len(row) > 0:
            audio, _ = sf.read(row.iloc[0]['audio_path'])
            conditions.append((f"SNR = {snr:+.0f} dB", audio))

    # Band-limiting
    for band in ['telephony', 'lp3400', 'hp300']:
        row = df_filt[(df_filt['band_filter'] == band) & (df_filt['variant_type'] == 'band')]
        if len(row) > 0:
            audio, _ = sf.read(row.iloc[0]['audio_path'])
            band_label = {'telephony': 'Telephony (300-3400 Hz)', 'lp3400': 'LP 3400 Hz', 'hp300': 'HP 300 Hz'}[band]
            conditions.append((band_label, audio))

    # Plot
    n_conditions = len(conditions)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(12, 2.5 * n_conditions))
    if n_conditions == 1:
        axes = [axes]

    for i, (title, audio) in enumerate(conditions):
        im = plot_spectrogram(axes[i], audio, sr, title)

    # Colorbar
    fig.colorbar(im, ax=axes, label='Power (dB)', shrink=0.6)

    plt.tight_layout()

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
