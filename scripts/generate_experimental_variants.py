#!/usr/bin/env python3
"""
Generate Experimental Variants (Duration × SNR)

Takes base 1000ms clips and generates variants with:
- Different durations: 20, 40, 60, 80, 100, 200, 500, 1000ms
- Different SNR levels: -10, -5, 0, +5, +10, +20 dB
- Padding to 2000ms with low-amplitude noise (centered audio)

This creates a factorial design: 8 durations × 6 SNRs = 48 variants per base clip

Usage:
    python scripts/generate_experimental_variants.py \
        --input_base data/processed/base_1000ms \
        --output_dir data/processed/experimental_variants \
        --durations 20 40 60 80 100 200 500 1000 \
        --snr_levels -10 -5 0 5 10 20 \
        --padding_duration 2000 \
        --noise_amplitude 0.0001

Key Features:
- Centered audio in 2000ms container
- Low-amplitude noise to fill silence (prevents giving duration cues)
- SNR computed over ENTIRE 2000ms container (not just effective segment)
- Preserves metadata traceability (base clip ID → variant ID)

References:
- Psychoacoustic testing requires consistent temporal context
- Padding prevents model from "cheating" by detecting silence edges
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def trim_audio_to_duration(audio: np.ndarray, target_duration_ms: int, sr: int) -> np.ndarray:
    """
    Trim audio to target duration (centered extraction).

    Args:
        audio: Audio array (assumed to be 1000ms from base clips)
        target_duration_ms: Target duration in milliseconds
        sr: Sampling rate

    Returns:
        Trimmed audio
    """
    target_samples = int(target_duration_ms * sr / 1000)
    current_samples = len(audio)

    if target_samples >= current_samples:
        # No trimming needed (this shouldn't happen for base 1000ms clips)
        return audio

    # Center extraction
    start_idx = (current_samples - target_samples) // 2
    end_idx = start_idx + target_samples

    return audio[start_idx:end_idx]


def pad_audio_to_container(audio: np.ndarray, container_duration_ms: int, sr: int,
                          noise_amplitude: float = 0.0001) -> np.ndarray:
    """
    Pad audio to container duration with centered placement and low-amplitude noise.

    Example for 200ms audio in 2000ms container:
        [900ms noise] + [200ms audio] + [900ms noise] = 2000ms total

    Args:
        audio: Audio array
        container_duration_ms: Target container duration in milliseconds
        sr: Sampling rate
        noise_amplitude: RMS amplitude of padding noise (default: 0.0001)

    Returns:
        Padded audio (container_duration_ms long)
    """
    container_samples = int(container_duration_ms * sr / 1000)
    audio_samples = len(audio)

    if audio_samples >= container_samples:
        # Already long enough, just truncate
        return audio[:container_samples]

    # Calculate padding lengths (centered)
    total_padding = container_samples - audio_samples
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding

    # Generate low-amplitude noise for padding
    left_noise = np.random.normal(0, noise_amplitude, left_padding).astype(np.float32)
    right_noise = np.random.normal(0, noise_amplitude, right_padding).astype(np.float32)

    # Concatenate: [left_noise] + [audio] + [right_noise]
    padded = np.concatenate([left_noise, audio, right_noise])

    return padded


def add_noise_at_snr(audio: np.ndarray, snr_db: float, noise_amplitude: float = 0.0001) -> np.ndarray:
    """
    Add white noise to audio at specified SNR.

    CRITICAL: SNR is computed over the ENTIRE audio (including padding noise),
    not just the effective segment. This ensures consistent SNR measurement.

    Args:
        audio: Audio array (already padded to 2000ms)
        snr_db: Target SNR in dB
        noise_amplitude: Base noise amplitude (for reference)

    Returns:
        Audio with added noise at target SNR
    """
    # Compute RMS of entire signal (including existing padding noise)
    signal_rms = np.sqrt(np.mean(audio ** 2))

    if signal_rms < 1e-6:
        # Signal too quiet, just return as is
        return audio

    # Calculate noise RMS needed for target SNR
    # SNR_dB = 20 * log10(signal_RMS / noise_RMS)
    # => noise_RMS = signal_RMS / 10^(SNR_dB / 20)
    noise_rms = signal_rms / (10 ** (snr_db / 20))

    # Generate white noise
    noise = np.random.normal(0, noise_rms, len(audio)).astype(np.float32)

    # Add noise to signal
    noisy_audio = audio + noise

    # Safety clip to prevent overflow
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

    return noisy_audio


def create_variant(audio: np.ndarray, sr: int, duration_ms: int, snr_db: float,
                  padding_duration_ms: int, noise_amplitude: float) -> tuple[np.ndarray, float]:
    """
    Create a single variant with specified duration and SNR.

    Pipeline:
    1. Trim audio to target duration (centered)
    2. Pad to container duration with low-amplitude noise (centered)
    3. Add noise at target SNR (computed over entire container)

    Args:
        audio: Base audio (1000ms)
        sr: Sampling rate
        duration_ms: Target duration for the audio segment
        snr_db: Target SNR in dB
        padding_duration_ms: Container duration (e.g., 2000ms)
        noise_amplitude: Amplitude of padding noise

    Returns:
        (variant_audio, variant_rms)
    """
    # Step 1: Trim to target duration
    trimmed = trim_audio_to_duration(audio, duration_ms, sr)

    # Step 2: Pad to container with noise
    padded = pad_audio_to_container(trimmed, padding_duration_ms, sr, noise_amplitude)

    # Step 3: Add noise at target SNR
    noisy = add_noise_at_snr(padded, snr_db, noise_amplitude)

    # Compute final RMS
    variant_rms = np.sqrt(np.mean(noisy ** 2))

    return noisy, variant_rms


def generate_variants_for_split(base_csv: Path, output_dir: Path, split_name: str,
                                durations_ms: list, snr_levels_db: list,
                                padding_duration_ms: int, noise_amplitude: float,
                                target_sr: int) -> pd.DataFrame:
    """
    Generate all variants for a single split (train/dev/test).

    Args:
        base_csv: Path to base metadata CSV (e.g., train_base.csv)
        output_dir: Output directory for variants
        split_name: 'train', 'dev', or 'test'
        durations_ms: List of durations to generate
        snr_levels_db: List of SNR levels to generate
        padding_duration_ms: Container duration
        noise_amplitude: Padding noise amplitude
        target_sr: Target sampling rate

    Returns:
        DataFrame with variant metadata
    """
    # Load base metadata
    base_df = pd.read_csv(base_csv)

    print(f"\n{split_name.upper()} split:")
    print(f"  Base clips: {len(base_df)}")
    print(f"  Variants per clip: {len(durations_ms)} × {len(snr_levels_db)} = {len(durations_ms) * len(snr_levels_db)}")
    print(f"  Total variants: {len(base_df) * len(durations_ms) * len(snr_levels_db)}")

    # Create output audio directory
    audio_dir = output_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Generate variants
    variants = []

    for _, row in tqdm(base_df.iterrows(), total=len(base_df), desc=f"Processing {split_name}"):
        # Load base audio
        base_audio, sr = sf.read(row['audio_path'])

        # Resample if needed
        if sr != target_sr:
            import librosa
            base_audio = librosa.resample(base_audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        base_clip_id = row['clip_id']

        # Generate all duration × SNR combinations
        for duration_ms in durations_ms:
            for snr_db in snr_levels_db:
                # Create variant
                variant_audio, variant_rms = create_variant(
                    base_audio, sr, duration_ms, snr_db,
                    padding_duration_ms, noise_amplitude
                )

                # Create variant ID
                variant_id = f"{base_clip_id}_dur{duration_ms}ms_snr{snr_db:+d}dB"

                # Save variant audio
                output_path = audio_dir / f"{variant_id}.wav"
                sf.write(output_path, variant_audio, sr)

                # Store metadata
                variants.append({
                    'clip_id': base_clip_id,
                    'variant_id': variant_id,
                    'duration_ms': duration_ms,
                    'snr_db': snr_db,
                    'audio_path': str(output_path.relative_to(output_dir.parent.parent)),
                    'ground_truth': row['ground_truth'],
                    'dataset': row['dataset'],
                    'group_id': row['group_id'],
                    'sr': sr,
                    'rms': variant_rms,
                    'container_duration_ms': padding_duration_ms,
                    'noise_amplitude': noise_amplitude,
                    'normalization': row.get('normalization', 'peak')
                })

    # Create DataFrame
    variants_df = pd.DataFrame(variants)

    # Save metadata CSV
    metadata_path = output_dir / f"{split_name}_metadata.csv"
    variants_df.to_csv(metadata_path, index=False)

    print(f"  ✓ Saved {len(variants_df)} variants")
    print(f"    Audio: {audio_dir}")
    print(f"    Metadata: {metadata_path}")

    return variants_df


def main():
    parser = argparse.ArgumentParser(description="Generate experimental variants (duration × SNR)")
    parser.add_argument("--input_base", type=Path, required=True,
                       help="Input directory with base clips (contains train_base.csv, etc.)")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for variants")
    parser.add_argument("--durations", type=int, nargs='+',
                       default=[20, 40, 60, 80, 100, 200, 500, 1000],
                       help="Durations in milliseconds")
    parser.add_argument("--snr_levels", type=int, nargs='+',
                       default=[-10, -5, 0, 5, 10, 20],
                       help="SNR levels in dB")
    parser.add_argument("--padding_duration", type=int, default=2000,
                       help="Container duration for padding (default: 2000ms)")
    parser.add_argument("--noise_amplitude", type=float, default=0.0001,
                       help="RMS amplitude of padding noise (default: 0.0001)")
    parser.add_argument("--target_sr", type=int, default=16000,
                       help="Target sampling rate (default: 16000)")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATE EXPERIMENTAL VARIANTS (DURATION × SNR)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input: {args.input_base}")
    print(f"  Output: {args.output_dir}")
    print(f"  Durations: {args.durations} ms")
    print(f"  SNR levels: {args.snr_levels} dB")
    print(f"  Padding container: {args.padding_duration}ms")
    print(f"  Noise amplitude: {args.noise_amplitude}")
    print(f"  Target SR: {args.target_sr}Hz")

    # Process each split
    for split_name in ['train', 'dev', 'test']:
        base_csv = args.input_base / f"{split_name}_base.csv"

        if not base_csv.exists():
            print(f"\n⚠️  Warning: {base_csv} not found, skipping {split_name}")
            continue

        print(f"\n{'=' * 80}")
        print(f"PROCESSING {split_name.upper()} SPLIT")
        print(f"{'=' * 80}")

        variants_df = generate_variants_for_split(
            base_csv,
            args.output_dir,
            split_name,
            args.durations,
            args.snr_levels,
            args.padding_duration,
            args.noise_amplitude,
            args.target_sr
        )

        # Print statistics
        print(f"\n  Statistics for {split_name}:")
        print(f"    Total variants: {len(variants_df)}")
        print(f"    Unique base clips: {variants_df['clip_id'].nunique()}")
        print(f"    Durations: {sorted(variants_df['duration_ms'].unique())}")
        print(f"    SNR levels: {sorted(variants_df['snr_db'].unique())}")
        print(f"    Ground truth distribution:")
        for gt, count in variants_df['ground_truth'].value_counts().items():
            print(f"      {gt}: {count}")

    print(f"\n{'=' * 80}")
    print(f"✓ EXPERIMENTAL VARIANTS GENERATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  train_metadata.csv: All train variants")
    print(f"  dev_metadata.csv: All dev variants")
    print(f"  test_metadata.csv: All test variants")
    print(f"\n✓ Factorial design: {len(args.durations)} durations × {len(args.snr_levels)} SNRs per base clip")


if __name__ == "__main__":
    main()
