#!/usr/bin/env python3
"""
Generate silence and noise clips for NONSPEECH augmentation.

Creates clips with different noise types and levels:
- Gaussian (white) noise
- Pink noise
- Brown noise
- Very low amplitude (near silence)

Usage:
    python scripts/generate_silence_noise.py --output_dir data/raw/silence_noise --n_clips 500
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def generate_white_noise(duration_sec, sr, amplitude):
    """Generate white (gaussian) noise."""
    n_samples = int(duration_sec * sr)
    return np.random.randn(n_samples).astype(np.float32) * amplitude


def generate_pink_noise(duration_sec, sr, amplitude):
    """Generate pink noise (1/f spectrum)."""
    n_samples = int(duration_sec * sr)

    # Generate white noise
    white = np.random.randn(n_samples)

    # Apply pink filter (simple approximation)
    # Pink noise has 1/f power spectrum
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/sr)

    # Avoid division by zero
    freqs[0] = 1

    # Apply 1/sqrt(f) filter
    pink_fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(pink_fft, n=n_samples)

    # Normalize and scale
    pink = pink / np.max(np.abs(pink))
    return pink.astype(np.float32) * amplitude


def generate_brown_noise(duration_sec, sr, amplitude):
    """Generate brown noise (1/f^2 spectrum)."""
    n_samples = int(duration_sec * sr)

    white = np.random.randn(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, 1/sr)

    freqs[0] = 1

    # Apply 1/f filter for brown noise
    brown_fft = fft / freqs
    brown = np.fft.irfft(brown_fft, n=n_samples)

    brown = brown / np.max(np.abs(brown))
    return brown.astype(np.float32) * amplitude


def main():
    parser = argparse.ArgumentParser(description="Generate silence/noise clips")
    parser.add_argument("--output_dir", type=str, default="data/raw/silence_noise",
                        help="Output directory")
    parser.add_argument("--n_clips", type=int, default=500,
                        help="Total number of clips to generate")
    parser.add_argument("--duration_sec", type=float, default=2.0,
                        help="Duration of each clip in seconds")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define noise types and amplitude ranges
    noise_configs = [
        # Very quiet (near silence)
        {"type": "white", "amp_db": -60, "count": int(args.n_clips * 0.15)},  # 15%
        {"type": "white", "amp_db": -50, "count": int(args.n_clips * 0.15)},  # 15%

        # Low noise
        {"type": "white", "amp_db": -40, "count": int(args.n_clips * 0.15)},  # 15%
        {"type": "pink", "amp_db": -40, "count": int(args.n_clips * 0.10)},   # 10%
        {"type": "brown", "amp_db": -40, "count": int(args.n_clips * 0.10)},  # 10%

        # Medium noise
        {"type": "white", "amp_db": -30, "count": int(args.n_clips * 0.10)},  # 10%
        {"type": "pink", "amp_db": -30, "count": int(args.n_clips * 0.10)},   # 10%
        {"type": "brown", "amp_db": -30, "count": int(args.n_clips * 0.10)},  # 10%

        # Moderate noise
        {"type": "white", "amp_db": -20, "count": int(args.n_clips * 0.05)},  # 5%
    ]

    print(f"Generating {args.n_clips} noise clips...")
    print(f"Output: {output_dir}")
    print(f"Duration: {args.duration_sec}s @ {args.sr}Hz")

    clip_idx = 0

    for config in noise_configs:
        noise_type = config["type"]
        amp_db = config["amp_db"]
        count = config["count"]

        # Convert dB to linear amplitude
        amplitude = 10 ** (amp_db / 20)

        print(f"\nGenerating {count} clips of {noise_type} noise at {amp_db}dB...")

        for i in tqdm(range(count), desc=f"{noise_type} {amp_db}dB"):
            # Generate noise
            if noise_type == "white":
                audio = generate_white_noise(args.duration_sec, args.sr, amplitude)
            elif noise_type == "pink":
                audio = generate_pink_noise(args.duration_sec, args.sr, amplitude)
            elif noise_type == "brown":
                audio = generate_brown_noise(args.duration_sec, args.sr, amplitude)

            # Save
            filename = f"noise_{noise_type}_{amp_db}db_{clip_idx:04d}.wav"
            filepath = output_dir / filename
            sf.write(filepath, audio, args.sr)

            clip_idx += 1

    print(f"\nDone! Generated {clip_idx} clips in {output_dir}")
    print(f"\nTo use in training, run:")
    print(f"  python scripts/augment_dataset_with_noise.py")


if __name__ == "__main__":
    main()
