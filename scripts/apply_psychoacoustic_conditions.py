#!/usr/bin/env python3
"""
Apply psychoacoustic conditions to 1000ms audio padded to 2000ms.

Applies:
1. SNR: White noise at different levels over 2000ms
2. Band-limiting: Filters over 2000ms
3. Reverb: RIR convolution over 2000ms
"""

import json
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import signal
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_noise_snr(audio: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """
    Add white noise at specified SNR level.

    Args:
        audio: Audio signal
        snr_db: Signal-to-noise ratio in dB
        seed: Random seed for reproducibility

    Returns:
        Audio with added noise
    """
    rng = np.random.RandomState(seed)

    # Compute signal power
    signal_power = np.mean(audio ** 2)

    # Compute noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate white noise
    noise = rng.normal(0, np.sqrt(noise_power), len(audio))

    # Add noise to signal
    noisy_audio = audio + noise

    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val * 0.95

    return noisy_audio


def apply_bandpass_filter(audio: np.ndarray, sr: int, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    """
    Apply bandpass filter.

    Args:
        audio: Audio signal
        sr: Sample rate
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        order: Filter order

    Returns:
        Filtered audio
    """
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth bandpass filter
    sos = signal.butter(order, [low, high], btype='band', output='sos')

    # Apply filter (zero-phase)
    filtered = signal.sosfiltfilt(sos, audio)

    return filtered


def apply_lowpass_filter(audio: np.ndarray, sr: int, cutoff: float, order: int = 5) -> np.ndarray:
    """Apply lowpass filter."""
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
    filtered = signal.sosfiltfilt(sos, audio)
    return filtered


def apply_highpass_filter(audio: np.ndarray, sr: int, cutoff: float, order: int = 5) -> np.ndarray:
    """Apply highpass filter."""
    nyquist = sr / 2
    normal_cutoff = cutoff / nyquist
    sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    filtered = signal.sosfiltfilt(sos, audio)
    return filtered


def convolve_rir(audio: np.ndarray, rir: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convolve audio with RIR.

    Args:
        audio: Audio signal
        rir: Room impulse response
        normalize: Whether to normalize output

    Returns:
        Convolved audio
    """
    # Convolve
    reverb_audio = np.convolve(audio, rir, mode='same')

    # Normalize to prevent clipping
    if normalize:
        max_val = np.max(np.abs(reverb_audio))
        if max_val > 0:
            reverb_audio = reverb_audio / max_val * 0.95

    return reverb_audio


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    input_manifest_path = project_root / "data/processed/psychoacoustic_test_input.jsonl"
    selected_rirs_path = project_root / "data/external/RIRS_NOISES/selected_rirs.json"
    output_base_dir = project_root / "data/processed/psychoacoustic_conditions"

    logger.info("="*80)
    logger.info("APPLYING PSYCHOACOUSTIC CONDITIONS")
    logger.info("="*80)

    # Load input manifest
    logger.info(f"\nLoading input manifest: {input_manifest_path}")
    input_df = pd.read_json(input_manifest_path, lines=True)
    logger.info(f"Total samples: {len(input_df)}")
    logger.info(f"Labels: {input_df.label.value_counts().to_dict()}")

    # Define conditions
    snr_levels = [20, 10, 5, 0, -5, -10]  # dB
    band_filters = {
        'telephony': (300, 3400),  # Telephony band
        'lowpass_3400': (None, 3400),  # Low-pass 3400 Hz
        'highpass_300': (300, None),  # High-pass 300 Hz
    }

    # Load RIR metadata
    logger.info(f"\nLoading RIR metadata: {selected_rirs_path}")
    with open(selected_rirs_path) as f:
        selected_rirs = json.load(f)

    # Load RIRs
    rir_data = {}
    for bin_name, rir_info in selected_rirs.items():
        rir_path = Path(rir_info['path'])
        rir, rir_sr = sf.read(rir_path)
        if rir.ndim > 1:
            rir = rir[:, 0]
        rir_data[bin_name] = {'audio': rir, 'sr': rir_sr, 't60': rir_info['t60']}

    all_variants = []

    # ========================================================================
    # 1. SNR CONDITIONS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("GENERATING SNR CONDITIONS")
    logger.info("="*80)

    snr_output_dir = output_base_dir / "snr" / "audio"
    snr_output_dir.mkdir(parents=True, exist_ok=True)

    for snr_db in snr_levels:
        logger.info(f"\n  SNR = {snr_db:+d} dB")

        for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc=f"  SNR {snr_db:+d}dB", leave=False):
            # Load audio (already padded to 2000ms)
            audio, sr = sf.read(row['audio_path'])
            if audio.ndim > 1:
                audio = audio[:, 0]

            # Apply noise
            noisy_audio = add_noise_snr(audio, snr_db, seed=42 + idx)

            # Save
            output_filename = f"{row['clip_id']}_snr{snr_db:+d}db.wav"
            output_path = snr_output_dir / output_filename
            sf.write(output_path, noisy_audio, sr)

            # Record variant
            all_variants.append({
                'clip_id': row['clip_id'],
                'original_path': row['original_path'],
                'audio_path': str(output_path),
                'label': row['label'],
                'duration_ms': row['duration_ms'],
                'condition_type': 'snr',
                'variant_type': f'snr_{snr_db:+d}db',
                'snr_db': snr_db,
                'sample_rate': sr
            })

    # ========================================================================
    # 2. BAND-LIMITING CONDITIONS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("GENERATING BAND-LIMITING CONDITIONS")
    logger.info("="*80)

    band_output_dir = output_base_dir / "band" / "audio"
    band_output_dir.mkdir(parents=True, exist_ok=True)

    for filter_name, (lowcut, highcut) in band_filters.items():
        logger.info(f"\n  Filter: {filter_name} (lowcut={lowcut}, highcut={highcut})")

        for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc=f"  {filter_name}", leave=False):
            # Load audio
            audio, sr = sf.read(row['audio_path'])
            if audio.ndim > 1:
                audio = audio[:, 0]

            # Apply filter
            if lowcut and highcut:
                filtered_audio = apply_bandpass_filter(audio, sr, lowcut, highcut)
            elif highcut:
                filtered_audio = apply_lowpass_filter(audio, sr, highcut)
            elif lowcut:
                filtered_audio = apply_highpass_filter(audio, sr, lowcut)

            # Save
            output_filename = f"{row['clip_id']}_{filter_name}.wav"
            output_path = band_output_dir / output_filename
            sf.write(output_path, filtered_audio, sr)

            # Record variant
            all_variants.append({
                'clip_id': row['clip_id'],
                'original_path': row['original_path'],
                'audio_path': str(output_path),
                'label': row['label'],
                'duration_ms': row['duration_ms'],
                'condition_type': 'band',
                'variant_type': f'band_{filter_name}',
                'filter_name': filter_name,
                'lowcut_hz': lowcut,
                'highcut_hz': highcut,
                'sample_rate': sr
            })

    # ========================================================================
    # 3. REVERB CONDITIONS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("GENERATING REVERB CONDITIONS")
    logger.info("="*80)

    reverb_output_dir = output_base_dir / "reverb" / "audio"
    reverb_output_dir.mkdir(parents=True, exist_ok=True)

    for bin_name, rir_info in rir_data.items():
        logger.info(f"\n  {bin_name}: T60={rir_info['t60']:.3f}s")

        for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc=f"  {bin_name}", leave=False):
            # Load audio
            audio, sr = sf.read(row['audio_path'])
            if audio.ndim > 1:
                audio = audio[:, 0]

            # Check RIR sample rate
            if rir_info['sr'] != sr:
                logger.warning(f"RIR sr ({rir_info['sr']}) != audio sr ({sr}), skipping")
                continue

            # Apply reverb
            reverb_audio = convolve_rir(audio, rir_info['audio'])

            # Save
            output_filename = f"{row['clip_id']}_{bin_name}.wav"
            output_path = reverb_output_dir / output_filename
            sf.write(output_path, reverb_audio, sr)

            # Record variant
            all_variants.append({
                'clip_id': row['clip_id'],
                'original_path': row['original_path'],
                'audio_path': str(output_path),
                'label': row['label'],
                'duration_ms': row['duration_ms'],
                'condition_type': 'reverb',
                'variant_type': f'reverb_{bin_name}',
                'rir_type': bin_name,
                't60': rir_info['t60'],
                'sample_rate': sr
            })

    # ========================================================================
    # SAVE MANIFESTS
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("SAVING MANIFESTS")
    logger.info("="*80)

    variants_df = pd.DataFrame(all_variants)

    # Save combined manifest
    combined_manifest_path = output_base_dir / "all_conditions_manifest.jsonl"
    variants_df.to_json(combined_manifest_path, orient='records', lines=True)
    logger.info(f"\nSaved combined manifest: {combined_manifest_path}")

    parquet_path = output_base_dir / "all_conditions_manifest.parquet"
    variants_df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved parquet: {parquet_path}")

    # Save per-condition manifests
    for condition in ['snr', 'band', 'reverb']:
        condition_df = variants_df[variants_df['condition_type'] == condition]
        condition_path = output_base_dir / condition / f"{condition}_manifest.parquet"
        condition_df.to_parquet(condition_path, index=False)
        logger.info(f"Saved {condition} manifest: {condition_path}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"\nTotal variants generated: {len(variants_df)}")
    logger.info(f"\nBreakdown by condition type:")
    for cond, count in variants_df['condition_type'].value_counts().items():
        logger.info(f"  {cond}: {count}")

    logger.info(f"\nBreakdown by variant type:")
    for variant, count in variants_df['variant_type'].value_counts().sort_index().items():
        logger.info(f"  {variant}: {count}")

    logger.info("\n" + "="*80)
    logger.info("DONE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
