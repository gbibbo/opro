#!/usr/bin/env python3
"""
Select 3 representative RIRs from the RIRS_NOISES dataset.

Selects one RIR per T60 bin (low, medium, high) based on:
1. T60 value close to bin median
2. Clean signal (minimal background noise)
3. Balanced frequency response (no extreme resonances)
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_rir_metadata(metadata_path: Path) -> List[Dict]:
    """Load RIR metadata from JSON."""
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)

    # Convert dict to list of dicts with normalized field names
    # T60 -> t60, sr -> sample_rate, duration_sec -> duration
    metadata = []
    for relative_path, info in metadata_dict.items():
        metadata.append({
            'relative_path': relative_path,  # Original path from JSON key
            'path': info['path'],  # Full path from JSON value
            't60': info['T60'],
            'sample_rate': info['sr'],
            'duration': info['duration_sec']
        })

    return metadata


def compute_rir_quality_metrics(rir_path: Path) -> Dict[str, float]:
    """
    Compute quality metrics for an RIR.

    Returns:
        snr: Signal-to-noise ratio (higher is better)
        spectral_flatness: Measure of frequency balance (closer to 1 is better)
    """
    # Load RIR
    rir, sr = sf.read(rir_path)

    # If stereo, take first channel
    if rir.ndim > 1:
        rir = rir[:, 0]

    # Compute SNR (energy in first 50% vs last 50%)
    midpoint = len(rir) // 2
    signal_energy = np.sum(rir[:midpoint] ** 2)
    noise_energy = np.sum(rir[midpoint:] ** 2)

    if noise_energy > 0:
        snr = 10 * np.log10(signal_energy / noise_energy)
    else:
        snr = 100.0  # Perfect SNR

    # Compute spectral flatness (geometric mean / arithmetic mean of power spectrum)
    spectrum = np.abs(np.fft.rfft(rir))
    power_spectrum = spectrum ** 2

    # Avoid log(0)
    power_spectrum = power_spectrum[power_spectrum > 0]

    geometric_mean = np.exp(np.mean(np.log(power_spectrum)))
    arithmetic_mean = np.mean(power_spectrum)

    spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0

    return {
        'snr': snr,
        'spectral_flatness': spectral_flatness
    }


def select_best_rir_for_bin(
    candidates: List[Dict],
    target_t60: float,
    rir_base_path: Path
) -> Tuple[Dict, Dict]:
    """
    Select the best RIR for a given T60 bin.

    Criteria:
    1. T60 closest to target
    2. High SNR (>10 dB)
    3. Spectral flatness (>0.1)

    Returns:
        best_rir: Metadata of selected RIR
        quality_metrics: Quality metrics of selected RIR
    """
    logger.info(f"\nEvaluating {len(candidates)} candidates for T60 ~ {target_t60:.2f}s")

    best_rir = None
    best_score = -np.inf
    best_metrics = None

    for rir_meta in candidates:
        # The path in JSON is already absolute from project root
        rir_path = Path(rir_meta['path'])

        # Skip if file doesn't exist
        if not rir_path.exists():
            continue

        # Compute quality metrics
        try:
            metrics = compute_rir_quality_metrics(rir_path)
        except Exception as e:
            logger.warning(f"  Failed to process {rir_path.name}: {e}")
            continue

        # Compute score (weighted combination)
        t60_diff = abs(rir_meta['t60'] - target_t60)

        # Normalize components to [0, 1]
        t60_score = np.exp(-t60_diff)  # Closer to target = higher score
        snr_score = 1.0 / (1.0 + np.exp(-0.1 * (metrics['snr'] - 10)))  # Sigmoid centered at 10 dB
        flatness_score = metrics['spectral_flatness']

        # Weighted sum
        score = (
            0.5 * t60_score +       # T60 proximity is most important
            0.3 * snr_score +       # Clean signal
            0.2 * flatness_score    # Balanced spectrum
        )

        if score > best_score:
            best_score = score
            best_rir = rir_meta
            best_metrics = metrics

    if best_rir:
        logger.info(f"  Selected: {best_rir['path']}")
        logger.info(f"    T60: {best_rir['t60']:.3f}s (target: {target_t60:.2f}s)")
        logger.info(f"    SNR: {best_metrics['snr']:.1f} dB")
        logger.info(f"    Spectral flatness: {best_metrics['spectral_flatness']:.3f}")
        logger.info(f"    Score: {best_score:.3f}")

    return best_rir, best_metrics


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    metadata_path = project_root / "data/external/RIRS_NOISES/rir_metadata.json"
    rir_base_path = project_root / "data/external/RIRS_NOISES"
    output_path = project_root / "data/external/RIRS_NOISES/selected_rirs.json"

    logger.info("="*80)
    logger.info("REPRESENTATIVE RIR SELECTION")
    logger.info("="*80)

    # Load metadata
    logger.info(f"\nLoading metadata from: {metadata_path}")
    metadata = load_rir_metadata(metadata_path)
    logger.info(f"Total RIRs: {len(metadata)}")

    # Define T60 bins
    bins = {
        'low_reverb': {
            'range': (0.0, 0.5),
            'target': 0.3
        },
        'medium_reverb': {
            'range': (0.5, 1.5),
            'target': 1.0
        },
        'high_reverb': {
            'range': (1.5, np.inf),
            'target': 2.5
        }
    }

    # Select one RIR per bin
    selected_rirs = {}

    for bin_name, bin_config in bins.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Bin: {bin_name}")
        logger.info(f"  T60 range: {bin_config['range'][0]:.1f}s - {bin_config['range'][1]:.1f}s")
        logger.info(f"  Target T60: {bin_config['target']:.1f}s")

        # Filter candidates
        candidates = [
            rir for rir in metadata
            if bin_config['range'][0] <= rir['t60'] < bin_config['range'][1]
        ]

        if not candidates:
            logger.warning(f"  No candidates found for {bin_name}!")
            continue

        # Select best RIR
        best_rir, metrics = select_best_rir_for_bin(
            candidates,
            bin_config['target'],
            rir_base_path
        )

        if best_rir:
            selected_rirs[bin_name] = {
                'path': best_rir['path'],
                't60': best_rir['t60'],
                'sample_rate': best_rir['sample_rate'],
                'duration': best_rir['duration'],
                'snr': metrics['snr'],
                'spectral_flatness': metrics['spectral_flatness']
            }

    # Save selected RIRs
    logger.info(f"\n{'='*80}")
    logger.info("FINAL SELECTION")
    logger.info("="*80)

    for bin_name, rir_info in selected_rirs.items():
        logger.info(f"\n{bin_name}:")
        logger.info(f"  Path: {rir_info['path']}")
        logger.info(f"  T60: {rir_info['t60']:.3f}s")
        logger.info(f"  SNR: {rir_info['snr']:.1f} dB")
        logger.info(f"  Spectral flatness: {rir_info['spectral_flatness']:.3f}")

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(selected_rirs, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"Selected RIRs saved to: {output_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
