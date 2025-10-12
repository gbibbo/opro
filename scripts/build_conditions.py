#!/usr/bin/env python3
"""
Build psychoacoustic condition variants for QSM evaluation.

From each 1000ms audio segment (padded to 2000ms), generates 20 variants:
  - 8 duration variants: 20, 40, 60, 80, 100, 200, 500, 1000 ms (each padded to 2000ms)
  - 6 SNR variants: -10, -5, 0, +5, +10, +20 dB (applied to full 1000ms)
  - 3 band filters: telephony, hp300, lp3400 (applied to full 1000ms)
  - 3 reverb bins: T60 0.0-0.4, 0.4-0.8, 0.8-1.5 s (applied to full 1000ms)

Total: 8 + 6 + 3 + 3 = 20 variants per clip (ADDITION, not multiplication)

Usage:
    python scripts/build_conditions.py \
        --input_manifest data/processed/qsm_dev_1000ms_only.jsonl \
        --output_dir data/processed/conditions/ \
        --durations 20,40,60,80,100,200,500,1000 \
        --snr_levels -10,-5,0,5,10,20 \
        --band_filters telephony,lp3400,hp300 \
        --rir_root data/external/RIRS_NOISES/RIRS_NOISES \
        --rir_metadata data/external/RIRS_NOISES/rir_metadata.json \
        --rir_t60_bins 0.0-0.4,0.4-0.8,0.8-1.5 \
        --n_workers 4
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import soundfile as sf
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path for qsm imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.audio import (
    extract_segment_center,
    pad_audio_center,
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


def parse_durations(dur_str: str) -> List[int]:
    """Parse comma-separated duration values in milliseconds."""
    if dur_str == "none":
        return []
    return [int(x) for x in dur_str.split(",")]


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


# Note: extract_segment_center() and pad_audio_center() are now imported from qsm.audio.slicing


def process_single_file(
    item: dict,
    output_dir: Path,
    durations_ms: List[int],
    snr_levels: List[float],
    band_filters: List[str],
    rir_db: Optional[object],
    t60_bins: List[Tuple[float, float]],
    sr: int = 16000,
    seed_base: int = 42,
) -> List[dict]:
    """
    Process a single 1000ms audio file and generate all 20 condition variants.

    Args:
        item: Manifest item with {"audio_path", "duration_ms", "label", ...}
        output_dir: Output directory for variants
        durations_ms: List of target durations (e.g., [20, 40, 60, 80, 100, 200, 500, 1000])
        snr_levels: List of SNR values to apply (e.g., [-10, -5, 0, 5, 10, 20])
        band_filters: List of band filter types (e.g., ["telephony", "lp3400", "hp300"])
        rir_db: RIR database instance (or None if no RIR)
        t60_bins: List of (T60_min, T60_max) tuples (e.g., [(0.0, 0.4), (0.4, 0.8), (0.8, 1.5)])
        sr: Sample rate
        seed_base: Base seed for reproducibility

    Returns:
        List of metadata dicts for all generated variants (20 per file)
    """
    audio_path = Path(item["audio_path"])
    if not audio_path.exists():
        logger.warning(f"Audio file not found: {audio_path}")
        return []

    # Load audio (should be 1000ms padded to 2000ms, or raw 1000ms)
    audio, file_sr = sf.read(audio_path, dtype="float32")
    if file_sr != sr:
        logger.warning(f"Sample rate mismatch: {audio_path} ({file_sr} Hz)")
        return []

    # Metadata
    clip_id = item.get("clip_id", audio_path.stem)
    duration_ms_original = item.get("duration_ms", 1000.0)  # Should be 1000ms
    label = item.get("label")

    # Base metadata
    base_meta = {
        "clip_id": clip_id,
        "original_path": str(audio_path),
        "duration_ms": duration_ms_original,
        "label": label,
    }

    variants = []

    # ========================================================================
    # SECTION 1: DURATION VARIANTS (8 variants)
    # Extract segments of different durations from the 1000ms, then pad to 2000ms
    # ========================================================================
    for dur_ms in durations_ms:
        # Extract duration segment from center of 1000ms
        segment = extract_segment_center(audio, dur_ms, sr)

        # Pad to 2000ms with low-amplitude noise
        target_samples_2000ms = int(2000 * sr / 1000.0)
        seed_dur = seed_base + hash((clip_id, dur_ms)) % 10000
        padded_segment = pad_audio_center(segment, target_samples_2000ms, sr, noise_amplitude=0.0001, seed=seed_dur)

        # Save duration variant
        output_name = f"{clip_id}_dur{dur_ms}ms.wav"
        output_path = output_dir / "duration" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, padded_segment, sr)

        variants.append({
            **base_meta,
            "variant_type": "duration",
            "duration_ms": dur_ms,  # Override with actual segment duration
            "snr_db": None,
            "band_filter": None,
            "rir_id": None,
            "T60": None,
            "audio_path": str(output_path),
        })

    # ========================================================================
    # SECTION 2: SNR VARIANTS (6 variants)
    # Apply noise to the FULL 1000ms segment (padded to 2000ms)
    # ========================================================================
    # Ensure audio is 2000ms padded
    target_samples_2000ms = int(2000 * sr / 1000.0)
    if len(audio) < target_samples_2000ms:
        audio_padded = pad_audio_center(audio, 2000, sr, noise_amplitude=0.0001, seed=seed_base + hash(clip_id) % 10000)
    else:
        audio_padded = audio[:target_samples_2000ms]

    for snr_db in snr_levels:
        seed_snr = seed_base + hash((clip_id, snr_db)) % 10000
        noisy_audio, noise_meta = mix_at_snr(
            audio_padded,
            snr_db=snr_db,
            sr=sr,
            padding_ms=2000,
            effective_dur_ms=duration_ms_original,
            seed=seed_snr,
        )

        # Save noisy variant
        output_name = f"{clip_id}_snr{snr_db:+.0f}db.wav"
        output_path = output_dir / "snr" / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, noisy_audio, sr)

        variants.append({
            **base_meta,
            "variant_type": "snr",
            "snr_db": snr_db,
            "band_filter": None,
            "rir_id": None,
            "T60": None,
            "audio_path": str(output_path),
            **noise_meta,
        })

    # ========================================================================
    # SECTION 3: BAND FILTER VARIANTS (3 variants)
    # Apply filters to the FULL 1000ms segment (padded to 2000ms)
    # ========================================================================
    for band_filter in band_filters:
        if band_filter == "none":
            continue  # Skip

        filtered_audio = apply_band_filter(audio_padded, sr, band_filter)

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

    # ========================================================================
    # SECTION 4: RIR VARIANTS (3 variants)
    # Apply reverb to the FULL 1000ms segment (padded to 2000ms)
    # ========================================================================
    if rir_db is not None and len(t60_bins) > 0:
        for t60_min, t60_max in t60_bins:
            # Get RIRs in this T60 bin
            rir_ids = rir_db.get_by_t60(t60_min, t60_max)
            if len(rir_ids) == 0:
                logger.warning(f"No RIRs found for T60 [{t60_min}, {t60_max}]")
                continue

            # Pick one RIR per bin (deterministic)
            seed_rir = (seed_base + hash((clip_id, t60_min))) % len(rir_ids)
            rir_id = rir_ids[seed_rir]
            rir_audio = rir_db.get_rir(rir_id, sr=sr)
            T60 = rir_db.rirs[rir_id].get("T60")

            reverb_audio = apply_rir(audio_padded, rir_audio, normalize=True)

            t60_label = f"T60_{t60_min:.1f}-{t60_max:.1f}"
            output_name = f"{clip_id}_rir_{t60_label}.wav"
            output_path = output_dir / "rir" / output_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, reverb_audio, sr)

            variants.append({
                **base_meta,
                "variant_type": "rir",
                "snr_db": None,
                "band_filter": None,
                "rir_id": rir_id,
                "T60": T60,
                "T60_bin": t60_label,
                "audio_path": str(output_path),
            })

    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_manifest", type=Path, required=True, help="Input JSONL manifest (1000ms segments)")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for variants")
    parser.add_argument("--durations", type=str, default="20,40,60,80,100,200,500,1000", help="Comma-separated durations (ms)")
    parser.add_argument("--snr_levels", type=str, default="-10,-5,0,5,10,20", help="Comma-separated SNR levels (dB)")
    parser.add_argument("--band_filters", type=str, default="telephony,lp3400,hp300", help="Comma-separated band filters")
    parser.add_argument("--rir_root", type=Path, default=None, help="Root directory of RIR dataset (OpenSLR SLR28)")
    parser.add_argument("--rir_metadata", type=Path, default=None, help="Optional RIR metadata JSON with T60")
    parser.add_argument("--rir_t60_bins", type=str, default="0.0-0.4,0.4-0.8,0.8-1.5", help="T60 bins (e.g., '0.0-0.4,0.4-0.8,0.8-1.5')")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (Hz)")
    args = parser.parse_args()

    # Parse parameters
    durations_ms = parse_durations(args.durations)
    snr_levels = parse_snr_levels(args.snr_levels)
    band_filters = parse_band_filters(args.band_filters)
    t60_bins = parse_t60_bins(args.rir_t60_bins)

    logger.info("=" * 80)
    logger.info("BUILDING PSYCHOACOUSTIC CONDITION VARIANTS")
    logger.info("=" * 80)
    logger.info(f"\nConfiguration:")
    logger.info(f"  Durations: {durations_ms} ms ({len(durations_ms)} variants)")
    logger.info(f"  SNR levels: {snr_levels} dB ({len(snr_levels)} variants)")
    logger.info(f"  Band filters: {band_filters} ({len(band_filters)} variants)")
    logger.info(f"  T60 bins: {t60_bins} ({len(t60_bins)} variants)")
    logger.info(f"\n  TOTAL variants per clip: {len(durations_ms) + len(snr_levels) + len(band_filters) + len(t60_bins)}")
    logger.info(f"  (8 durations + 6 SNR + 3 band + 3 RIR = 20 variants)")

    # Load RIR database if provided
    rir_db = None
    if args.rir_root is not None and args.rir_root.exists():
        logger.info(f"\nLoading RIR database from {args.rir_root}")
        rir_db = load_rir_database(args.rir_root, args.rir_metadata)
        logger.info(f"Loaded {len(rir_db.list_all())} RIRs")
    else:
        if len(t60_bins) > 0:
            logger.warning("T60 bins specified but no RIR root provided; skipping RIR processing")

    # Load input manifest
    with open(args.input_manifest, "r") as f:
        items = [json.loads(line) for line in f]
    logger.info(f"\nLoaded {len(items)} items from {args.input_manifest}")

    # Verify all are 1000ms
    non_1000ms = [item for item in items if item.get("duration_ms") != 1000.0]
    if len(non_1000ms) > 0:
        logger.warning(f"Found {len(non_1000ms)} items that are not 1000ms - they will be processed but may give unexpected results")

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
                durations_ms,
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

    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nGenerated {len(all_variants)} condition variants from {len(items)} clips")
    logger.info(f"Average variants per clip: {len(all_variants) / len(items):.1f}")
    logger.info(f"\nBreakdown by variant type:")
    for variant_type in df["variant_type"].unique():
        count = len(df[df["variant_type"] == variant_type])
        logger.info(f"  {variant_type:12s}: {count:5d} variants")

    logger.info(f"\nSaved manifest to {output_jsonl}")
    logger.info(f"Saved parquet to {output_parquet}")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
