#!/usr/bin/env python3
"""
Build psychoacoustic condition variants for QSM evaluation.

Takes padded audio files (2000 ms containers) and generates variants with:
  - SNR sweep (white noise at different SNR levels)
  - Band-limited filtering (telephony 300-3400 Hz, LP, HP)
  - Reverberation (RIR convolution with T60 sweep)

Produces a complete matrix: dur × SNR × band × RIR + metadata.

Usage:
    python scripts/build_conditions.py \\
        --input_manifest data/processed/qsm_dev_padded.jsonl \\
        --output_dir data/processed/conditions/ \\
        --snr_levels -10,-5,0,5,10,20 \\
        --band_filters none,telephony,lp3400,hp300 \\
        --rir_root data/external/RIRS_NOISES \\
        --rir_t60_bins 0.0-0.4,0.4-0.8,0.8-1.5 \\
        --n_workers 4
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import soundfile as sf
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from qsm.audio import (
    mix_at_snr,
    apply_bandpass,
    apply_lowpass,
    apply_highpass,
    load_rir_database,
    apply_rir,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_snr_levels(snr_str: str) -> List[float]:
    """Parse comma-separated SNR levels."""
    if snr_str == "none":
        return []
    return [float(x) for x in snr_str.split(",")]


def parse_band_filters(band_str: str) -> List[str]:
    """Parse comma-separated band filter types."""
    valid = {"none", "telephony", "lp3400", "hp300"}
    filters = band_str.split(",")
    for f in filters:
        if f not in valid:
            raise ValueError(f"Invalid band filter: {f}. Must be one of {valid}")
    return filters


def parse_t60_bins(t60_str: str) -> List[Tuple[float, float]]:
    """Parse T60 bin ranges (e.g., '0.0-0.4,0.4-0.8,0.8-1.5')."""
    if t60_str == "none":
        return []
    bins = []
    for bin_str in t60_str.split(","):
        low, high = bin_str.split("-")
        bins.append((float(low), float(high)))
    return bins


def apply_band_filter(audio: np.ndarray, sr: int, filter_type: str) -> np.ndarray:
    """Apply band filter by type."""
    if filter_type == "none":
        return audio
    elif filter_type == "telephony":
        return apply_bandpass(audio, sr, lowcut=300.0, highcut=3400.0)
    elif filter_type == "lp3400":
        return apply_lowpass(audio, sr, highcut=3400.0)
    elif filter_type == "hp300":
        return apply_highpass(audio, sr, lowcut=300.0)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def process_single_file(
    item: dict,
    output_dir: Path,
    snr_levels: List[float],
    band_filters: List[str],
    rir_db: Optional[object],
    t60_bins: List[Tuple[float, float]],
    sr: int = 16000,
    seed_base: int = 42,
) -> List[dict]:
    """
    Process a single audio file and generate all condition variants.

    Args:
        item: Manifest item with {"audio_path", "duration_ms", "label", ...}
        output_dir: Output directory for variants
        snr_levels: List of SNR values to apply
        band_filters: List of band filter types
        rir_db: RIR database instance (or None if no RIR)
        t60_bins: List of (T60_min, T60_max) tuples
        sr: Sample rate
        seed_base: Base seed for reproducibility

    Returns:
        List of metadata dicts for all generated variants
    """
    audio_path = Path(item["audio_path"])
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return []

    # Load audio
    audio, file_sr = sf.read(audio_path, dtype="float32")
    if file_sr != sr:
        logger.warning(f"Sample rate mismatch: {audio_path} ({file_sr} Hz)")
        return []

    # Metadata
    clip_id = item.get("clip_id", audio_path.stem)
    duration_ms = item.get("duration_ms")
    label = item.get("label")

    # Base metadata
    base_meta = {
        "clip_id": clip_id,
        "original_path": str(audio_path),
        "duration_ms": duration_ms,
        "label": label,
    }

    variants = []

    # 1. Generate base variants (no RIR, no noise, no filter)
    base_audio = audio.copy()

    # 2. SNR sweep
    for snr_db in snr_levels:
        seed_snr = seed_base + hash((clip_id, snr_db)) % 10000
        noisy_audio, noise_meta = mix_at_snr(
            base_audio,
            snr_db=snr_db,
            sr=sr,
            padding_ms=2000,
            effective_dur_ms=duration_ms,
            seed=seed_snr,
        )

        # Save noisy variant (no filter, no RIR)
        output_name = f"{clip_id}_snr{snr_db:+.0f}.wav"
        output_path = output_dir / "snr" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, noisy_audio, sr)

        variants.append({
            **base_meta,
            "variant_type": "snr",
            "snr_db": snr_db,
            "band_filter": "none",
            "rir_id": None,
            "T60": None,
            "audio_path": str(output_path),
            **noise_meta,
        })

    # 3. Band-limited filters (no noise, no RIR)
    for band_filter in band_filters:
        if band_filter == "none":
            continue  # Skip, already have base

        filtered_audio = apply_band_filter(base_audio, sr, band_filter)

        output_name = f"{clip_id}_band{band_filter}.wav"
        output_path = output_dir / "band" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, filtered_audio, sr)

        variants.append({
            **base_meta,
            "variant_type": "band",
            "snr_db": None,
            "band_filter": band_filter,
            "rir_id": None,
            "T60": None,
            "audio_path": str(output_path),
        })

    # 4. RIR sweep (no noise, no filter)
    if rir_db is not None and len(t60_bins) > 0:
        for t60_min, t60_max in t60_bins:
            # Get RIRs in this T60 bin
            rir_ids = rir_db.get_by_t60(t60_min, t60_max)
            if len(rir_ids) == 0:
                logger.warning(f"No RIRs found for T60 [{t60_min}, {t60_max}]")
                continue

            # Pick one RIR per bin (deterministic)
            seed_rir = seed_base + hash((clip_id, t60_min)) % len(rir_ids)
            rir_id = rir_ids[seed_rir]
            rir_audio = rir_db.get_rir(rir_id, sr=sr)
            T60 = rir_db.rirs[rir_id].get("T60")

            reverb_audio = apply_rir(base_audio, rir_audio, normalize=True)

            t60_label = f"T60_{t60_min:.1f}-{t60_max:.1f}"
            output_name = f"{clip_id}_rir_{t60_label}.wav"
            output_path = output_dir / "rir" / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, reverb_audio, sr)

            variants.append({
                **base_meta,
                "variant_type": "rir",
                "snr_db": None,
                "band_filter": "none",
                "rir_id": rir_id,
                "T60": T60,
                "T60_bin": t60_label,
                "audio_path": str(output_path),
            })

    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_manifest", type=Path, required=True, help="Input JSONL manifest (padded 2000ms)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for variants")
    parser.add_argument("--snr_levels", type=str, default="-10,-5,0,5,10,20", help="Comma-separated SNR levels (dB)")
    parser.add_argument("--band_filters", type=str, default="none,telephony,lp3400,hp300", help="Comma-separated band filters")
    parser.add_argument("--rir_root", type=Path, default=None, help="Root directory of RIR dataset (OpenSLR SLR28)")
    parser.add_argument("--rir_metadata", type=Path, default=None, help="Optional RIR metadata JSON with T60")
    parser.add_argument("--rir_t60_bins", type=str, default="none", help="T60 bins (e.g., '0.0-0.4,0.4-0.8,0.8-1.5')")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (Hz)")
    args = parser.parse_args()

    # Parse parameters
    snr_levels = parse_snr_levels(args.snr_levels)
    band_filters = parse_band_filters(args.band_filters)
    t60_bins = parse_t60_bins(args.rir_t60_bins)

    logger.info(f"SNR levels: {snr_levels}")
    logger.info(f"Band filters: {band_filters}")
    logger.info(f"T60 bins: {t60_bins}")

    # Load RIR database if provided
    rir_db = None
    if args.rir_root is not None and args.rir_root.exists():
        logger.info(f"Loading RIR database from {args.rir_root}")
        rir_db = load_rir_database(args.rir_root, args.rir_metadata)
        logger.info(f"Loaded {len(rir_db.list_all())} RIRs")
    else:
        if len(t60_bins) > 0:
            logger.warning("T60 bins specified but no RIR root provided; skipping RIR processing")

    # Load input manifest
    with open(args.input_manifest, "r") as f:
        items = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(items)} items from {args.input_manifest}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files
    all_variants = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = []
        for item in items:
            future = executor.submit(
                process_single_file,
                item,
                args.output_dir,
                snr_levels,
                band_filters,
                rir_db,
                t60_bins,
                args.sr,
                args.seed,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            variants = future.result()
            all_variants.extend(variants)

    # Save metadata
    output_jsonl = args.output_dir / "conditions_manifest.jsonl"
    with open(output_jsonl, "w") as f:
        for variant in all_variants:
            f.write(json.dumps(variant) + "\n")

    output_parquet = args.output_dir / "conditions_manifest.parquet"
    df = pd.DataFrame(all_variants)
    df.to_parquet(output_parquet, index=False)

    logger.info(f"Generated {len(all_variants)} condition variants")
    logger.info(f"Saved manifest to {output_jsonl}")
    logger.info(f"Saved parquet to {output_parquet}")


if __name__ == "__main__":
    main()
