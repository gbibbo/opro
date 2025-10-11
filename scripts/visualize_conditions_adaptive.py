#!/usr/bin/env python3
"""
Visualize spectrograms with ADAPTIVE scaling for better visibility.

Shows side-by-side comparison of:
- Original (clean)
- SNR sweep (-10, 0, +10 dB)
- Band-limiting (telephony, LP, HP)

Uses percentile-based color scaling to reveal signal details.

Usage:
    python scripts/visualize_conditions_adaptive.py
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


def plot_spectrogram(ax, audio, sr, title, global_vmin=None, global_vmax=None):
    """Plot spectrogram with adaptive or global scaling."""
    f, t, Sxx = compute_spectrogram(audio, sr)

    # Use global scaling if provided, else adaptive
    if global_vmin is not None and global_vmax is not None:
        vmin, vmax = global_vmin, global_vmax
    else:
        # Adaptive: use percentiles to capture dynamic range
        vmin = np.percentile(Sxx, 1)  # 1st percentile
        vmax = np.percentile(Sxx, 99.5)  # 99.5th percentile

    im = ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    ax.set_ylim([0, sr/2])

    # Add text with scaling info
    ax.text(0.02, 0.95, f'[{vmin:.0f}, {vmax:.0f}] dB',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    return im, vmin, vmax


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/conditions/conditions_manifest.parquet"))
    parser.add_argument("--output", type=Path, default=Path("assets/condition_spectrograms_adaptive.png"))
    parser.add_argument("--duration_ms", type=int, default=200, help="Duration to visualize (ms)")
    parser.add_argument("--label", type=str, default="SPEECH", choices=["SPEECH", "NON-SPEECH"])
    parser.add_argument("--global_scale", action="store_true", help="Use global color scale across all plots")
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

    # Compute global scale if requested
    global_vmin, global_vmax = None, None
    if args.global_scale:
        all_spectrograms = []
        for title, audio in conditions:
            _, _, Sxx = compute_spectrogram(audio, sr)
            all_spectrograms.append(Sxx)
        all_Sxx = np.concatenate([S.flatten() for S in all_spectrograms])
        global_vmin = np.percentile(all_Sxx, 1)
        global_vmax = np.percentile(all_Sxx, 99.5)
        print(f"Global scale: [{global_vmin:.1f}, {global_vmax:.1f}] dB")

    # Plot
    n_conditions = len(conditions)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(14, 2.5 * n_conditions))
    if n_conditions == 1:
        axes = [axes]

    for i, (title, audio) in enumerate(conditions):
        im, vmin, vmax = plot_spectrogram(axes[i], audio, sr, title, global_vmin, global_vmax)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, label='Power (dB)', shrink=0.6)

    plt.tight_layout()

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to {args.output}")

    # Don't show in headless mode
    # plt.show()


if __name__ == "__main__":
    main()
