#!/usr/bin/env python3
"""
Apply the 3 selected RIRs to audio samples.

Generates reverb variants using the RIRs from selected_rirs.json
"""

import json
import pandas as pd
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convolve_rir(audio: np.ndarray, rir: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convolve audio with RIR.

    Args:
        audio: Audio signal
        rir: Room impulse response
        normalize: Whether to normalize output to prevent clipping

    Returns:
        Convolved audio
    """
    # Convolve
    reverb_audio = np.convolve(audio, rir, mode='same')

    # Normalize to prevent clipping
    if normalize:
        max_val = np.max(np.abs(reverb_audio))
        if max_val > 0:
            reverb_audio = reverb_audio / max_val * 0.95  # Leave some headroom

    return reverb_audio


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    selected_rirs_path = project_root / "data/external/RIRS_NOISES/selected_rirs.json"
    input_manifest_path = project_root / "data/processed/reverb_test_input.jsonl"
    output_dir = project_root / "data/processed/reverb_variants"
    output_audio_dir = output_dir / "audio"
    output_manifest_path = output_dir / "conditions_manifest.jsonl"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("APPLYING SELECTED RIRs")
    logger.info("="*80)

    # Load selected RIRs
    logger.info(f"\nLoading selected RIRs from: {selected_rirs_path}")
    with open(selected_rirs_path) as f:
        selected_rirs = json.load(f)

    # Load RIR audio files
    rir_data = {}
    for bin_name, rir_info in selected_rirs.items():
        rir_path = Path(rir_info['path'])
        logger.info(f"  {bin_name}: T60={rir_info['t60']:.3f}s")
        logger.info(f"    Loading: {rir_path}")

        rir, sr = sf.read(rir_path)

        # If stereo, take first channel
        if rir.ndim > 1:
            rir = rir[:, 0]

        rir_data[bin_name] = {
            't60': rir_info['t60'],
            'audio': rir,
            'sr': sr
        }

    # Load input manifest
    logger.info(f"\nLoading input manifest: {input_manifest_path}")
    input_df = pd.read_json(input_manifest_path, lines=True)
    logger.info(f"Total samples: {len(input_df)}")

    # Generate variants
    logger.info(f"\nGenerating reverb variants...")
    variants = []

    for _, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing"):
        # Load original audio
        audio_path = Path(row['audio_path'])
        audio, sr = sf.read(audio_path)

        # If stereo, take first channel
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Apply each RIR
        for bin_name, rir_info in rir_data.items():
            # Resample RIR if needed
            rir = rir_info['audio']
            if rir_info['sr'] != sr:
                logger.warning(f"RIR sample rate ({rir_info['sr']}) != audio sample rate ({sr})")
                # For now, skip resampling - assume they match
                continue

            # Convolve
            reverb_audio = convolve_rir(audio, rir, normalize=True)

            # Save output
            output_filename = f"{row['clip_id']}_{bin_name}.wav"
            output_path = output_audio_dir / output_filename

            sf.write(output_path, reverb_audio, sr)

            # Add to variants
            variants.append({
                'clip_id': row['clip_id'],
                'original_path': row['audio_path'],
                'audio_path': str(output_path),
                'label': row['label'],
                'duration_ms': row['duration_ms'],
                'variant_type': f"reverb_{bin_name}",
                't60': rir_info['t60'],
                'sample_rate': sr
            })

    # Save manifest
    logger.info(f"\nSaving manifest to: {output_manifest_path}")
    variants_df = pd.DataFrame(variants)
    variants_df.to_json(output_manifest_path, orient='records', lines=True)

    # Also save as parquet
    parquet_path = output_dir / "conditions_manifest.parquet"
    variants_df.to_parquet(parquet_path, index=False)

    logger.info(f"\nGenerated {len(variants)} reverb variants:")
    logger.info(f"  Breakdown by variant type:")
    for variant_type, count in variants_df['variant_type'].value_counts().items():
        logger.info(f"    {variant_type}: {count}")

    logger.info(f"\n{'='*80}")
    logger.info("DONE")
    logger.info("="*80)


if __name__ == '__main__':
    main()
