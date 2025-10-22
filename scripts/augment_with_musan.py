#!/usr/bin/env python3
"""
MUSAN Noise Augmentation for Audio Data.

Adds realistic background noise from MUSAN dataset to training samples.

MUSAN Categories:
- music: Background music
- speech: Babble (multi-speaker background)
- noise: Environmental sounds (traffic, cafeteria, etc.)

Reference:
- Snyder et al. (2015): "MUSAN: A Music, Speech, and Noise Corpus"
- http://www.openslr.org/17/

Usage:
    python scripts/augment_with_musan.py \
        --audio_file data/audio.wav \
        --musan_root data/musan \
        --noise_type noise \
        --snr_db 10 \
        --output augmented.wav
"""

import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
import random


class MUSANAugmenter:
    """Add MUSAN noise to audio samples."""

    def __init__(self, musan_root, snr_range=(-5, 20), target_sr=16000):
        """
        Args:
            musan_root: Path to MUSAN dataset root
            snr_range: (min_snr, max_snr) in dB
            target_sr: Target sample rate (default 16kHz for Qwen2-Audio)
        """
        self.musan_root = Path(musan_root)
        self.snr_range = snr_range
        self.target_sr = target_sr

        # Load noise file lists
        self.noise_files = {}

        for category in ['music', 'speech', 'noise']:
            category_path = self.musan_root / category
            if category_path.exists():
                files = list(category_path.glob('**/*.wav'))
                self.noise_files[category] = files
                print(f"Found {len(files)} {category} files")
            else:
                print(f"WARNING: {category_path} not found")
                self.noise_files[category] = []

        total_files = sum(len(v) for v in self.noise_files.values())
        if total_files == 0:
            raise ValueError(f"No MUSAN files found in {musan_root}")

    def load_random_noise(self, noise_type='noise', duration_samples=None):
        """
        Load random noise file of specified type.

        Args:
            noise_type: 'music', 'speech', or 'noise'
            duration_samples: Required duration in samples (will crop/repeat)

        Returns:
            noise_tensor: (1, duration_samples) audio tensor
        """
        if noise_type not in self.noise_files or len(self.noise_files[noise_type]) == 0:
            raise ValueError(f"No {noise_type} files available")

        # Select random file
        noise_file = random.choice(self.noise_files[noise_type])

        # Load audio
        noise, sr = torchaudio.load(noise_file)

        # Convert to mono if needed
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            noise = resampler(noise)

        # Adjust duration
        if duration_samples is not None:
            if noise.shape[1] < duration_samples:
                # Repeat if too short
                repeats = int(np.ceil(duration_samples / noise.shape[1]))
                noise = noise.repeat(1, repeats)

            # Crop to exact duration
            if noise.shape[1] > duration_samples:
                # Random starting point for variety
                start = random.randint(0, noise.shape[1] - duration_samples)
                noise = noise[:, start:start + duration_samples]

        return noise

    def add_noise(self, audio, snr_db=None, noise_type='noise'):
        """
        Add noise to audio at specified SNR.

        Args:
            audio: (1, samples) or (samples,) audio tensor
            snr_db: Signal-to-noise ratio in dB (if None, random from range)
            noise_type: Type of noise to add

        Returns:
            augmented: Audio with noise added
        """
        # Ensure audio is 2D (1, samples)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Load matching noise
        noise = self.load_random_noise(noise_type, duration_samples=audio.shape[1])

        # Select SNR
        if snr_db is None:
            snr_db = np.random.uniform(*self.snr_range)

        # Compute power
        audio_power = audio.pow(2).mean()
        noise_power = noise.pow(2).mean()

        # Compute scale factor
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(audio_power / (snr_linear * noise_power + 1e-10))

        # Mix
        augmented = audio + scale * noise

        return augmented.squeeze(0)  # Return same shape as input

    def augment_batch(self, audio_batch, noise_type='noise', prob=0.5):
        """
        Augment a batch of audio with probability prob.

        Args:
            audio_batch: (batch, samples) tensor
            noise_type: Type of noise
            prob: Probability of applying augmentation

        Returns:
            augmented_batch: Same shape as input
        """
        batch_size = audio_batch.shape[0]
        augmented = []

        for i in range(batch_size):
            if random.random() < prob:
                # Apply augmentation
                aug = self.add_noise(audio_batch[i], noise_type=noise_type)
                augmented.append(aug)
            else:
                # Keep original
                augmented.append(audio_batch[i])

        return torch.stack(augmented)


def main():
    parser = argparse.ArgumentParser(description="Augment audio with MUSAN noise")
    parser.add_argument('--audio_file', type=str, required=True,
                        help='Input audio file')
    parser.add_argument('--musan_root', type=str, required=True,
                        help='MUSAN dataset root directory')
    parser.add_argument('--noise_type', type=str, default='noise',
                        choices=['music', 'speech', 'noise'],
                        help='Type of MUSAN noise to add')
    parser.add_argument('--snr_db', type=float, default=10.0,
                        help='Signal-to-noise ratio in dB')
    parser.add_argument('--output', type=str, required=True,
                        help='Output augmented audio file')
    parser.add_argument('--visualize', action='store_true',
                        help='Show waveform comparison')

    args = parser.parse_args()

    print("="*80)
    print("MUSAN NOISE AUGMENTATION")
    print("="*80)

    # Load audio
    print(f"\nLoading audio: {args.audio_file}")
    audio, sr = torchaudio.load(args.audio_file)

    # Initialize augmenter
    print(f"\nInitializing MUSAN augmenter from: {args.musan_root}")
    augmenter = MUSANAugmenter(musan_root=args.musan_root, target_sr=sr)

    # Apply augmentation
    print(f"\nAdding {args.noise_type} noise at SNR={args.snr_db} dB")
    augmented = augmenter.add_noise(audio, snr_db=args.snr_db, noise_type=args.noise_type)

    # Save
    print(f"\nSaving to: {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, augmented.unsqueeze(0), sr)

    # Visualize
    if args.visualize:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # Original
        axes[0].plot(audio[0].numpy())
        axes[0].set_title('Original Audio')
        axes[0].set_xlabel('Sample')
        axes[0].set_ylabel('Amplitude')

        # Augmented
        axes[1].plot(augmented.numpy())
        axes[1].set_title(f'Augmented (SNR={args.snr_db} dB, {args.noise_type})')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylabel('Amplitude')

        plt.tight_layout()
        plt.savefig(args.output.replace('.wav', '_comparison.png'))
        print(f"Visualization saved to: {args.output.replace('.wav', '_comparison.png')}")

    print("\n" + "="*80)
    print("DONE")
    print("="*80)


if __name__ == '__main__':
    main()
