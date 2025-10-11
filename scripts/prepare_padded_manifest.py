#!/usr/bin/env python3
"""
Prepare padded manifest from existing segments for condition generation.

Reads existing audio segments, pads them to 2000ms, and creates a manifest
for build_conditions.py.

Usage:
    python scripts/prepare_padded_manifest.py \
        --segments_root data/segments \
        --output_dir data/processed/padded \
        --output_manifest data/processed/qsm_dev_padded.jsonl \
        --padding_ms 2000 \
        --n_workers 8
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
import soundfile as sf
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_label_from_path(audio_path: Path) -> str:
    """Infer label from path structure."""
    parts = audio_path.parts
    if "esc50" in parts:
        return "NON-SPEECH"
    elif "ava_speech" in parts or "voxconverse" in parts:
        return "SPEECH"
    else:
        logger.warning(f"Could not infer label from path: {audio_path}")
        return "UNKNOWN"


def parse_duration_from_filename(filename: str) -> float:
    """Extract duration from filename (e.g., 'clip_100ms_001.wav' -> 100.0)."""
    import re
    match = re.search(r"_(\d+)ms_", filename)
    if match:
        return float(match.group(1))
    return None


def pad_audio(
    audio: np.ndarray,
    sr: int,
    target_duration_ms: int = 2000,
    noise_level: float = 0.0001,
    seed: int = 42,
) -> np.ndarray:
    """
    Pad audio to target duration with low-level noise.

    Args:
        audio: Input audio array
        sr: Sample rate
        target_duration_ms: Target duration in ms
        noise_level: Padding noise amplitude
        seed: Random seed

    Returns:
        Padded audio array
    """
    target_samples = int(sr * target_duration_ms / 1000.0)
    current_samples = len(audio)

    if current_samples >= target_samples:
        # Truncate if longer
        return audio[:target_samples]

    # Create noise padding
    rng = np.random.default_rng(seed)
    padding = rng.uniform(-noise_level, noise_level, target_samples).astype(audio.dtype)

    # Center the audio in the padding
    start_idx = (target_samples - current_samples) // 2
    padding[start_idx : start_idx + current_samples] = audio

    return padding


def process_audio_file(
    audio_path: Path,
    output_dir: Path,
    padding_ms: int,
    sr_target: int,
    noise_level: float,
    seed_base: int,
) -> Dict:
    """
    Process a single audio file: pad and save.

    Args:
        audio_path: Path to input audio
        output_dir: Output directory for padded audio
        padding_ms: Target padding duration (ms)
        sr_target: Target sample rate
        noise_level: Padding noise level
        seed_base: Base random seed

    Returns:
        Metadata dict
    """
    try:
        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")

        # Resample if needed
        if sr != sr_target:
            from scipy import signal as sp_signal
            num_samples = int(len(audio) * sr_target / sr)
            audio = sp_signal.resample(audio, num_samples).astype(np.float32)
            sr = sr_target

        # Get metadata
        duration_ms = parse_duration_from_filename(audio_path.name)
        if duration_ms is None:
            duration_ms = len(audio) / sr * 1000.0

        label = get_label_from_path(audio_path)
        clip_id = audio_path.stem

        # Pad audio
        seed = seed_base + hash(clip_id) % 10000
        padded = pad_audio(audio, sr, padding_ms, noise_level, seed)

        # Save padded audio
        output_subdir = output_dir / label.lower().replace("-", "_")
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_path = output_subdir / f"{clip_id}_padded.wav"
        sf.write(output_path, padded, sr)

        return {
            "clip_id": clip_id,
            "audio_path": str(output_path.resolve()),
            "original_path": str(audio_path.resolve()),
            "duration_ms": duration_ms,
            "label": label,
            "padding_ms": padding_ms,
            "sample_rate": sr,
        }

    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--segments_root",
        type=Path,
        default=Path("data/segments"),
        help="Root directory of audio segments",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/padded"),
        help="Output directory for padded audio",
    )
    parser.add_argument(
        "--output_manifest",
        type=Path,
        default=Path("data/processed/qsm_dev_padded.jsonl"),
        help="Output manifest JSONL path",
    )
    parser.add_argument(
        "--padding_ms",
        type=int,
        default=2000,
        help="Target padding duration (ms)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.0001,
        help="Padding noise amplitude",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Target sample rate (Hz)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--exclude_backup",
        action="store_true",
        help="Exclude backup directories",
    )
    args = parser.parse_args()

    # Find all audio files
    logger.info(f"Scanning {args.segments_root} for audio files...")
    audio_files = []
    for ext in ["*.wav", "*.flac", "*.mp3"]:
        audio_files.extend(args.segments_root.rglob(ext))

    # Exclude backup directories if requested
    if args.exclude_backup:
        audio_files = [
            f for f in audio_files
            if "backup" not in str(f).lower()
        ]

    logger.info(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        logger.error("No audio files found!")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process all files
    results = []
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = []
        for audio_file in audio_files:
            future = executor.submit(
                process_audio_file,
                audio_file,
                args.output_dir,
                args.padding_ms,
                args.sr,
                args.noise_level,
                args.seed,
            )
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc="Padding"):
            result = future.result()
            if result is not None:
                results.append(result)

    # Save manifest
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_manifest, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    # Save parquet
    output_parquet = args.output_manifest.with_suffix(".parquet")
    df = pd.DataFrame(results)
    df.to_parquet(output_parquet, index=False)

    # Summary
    logger.info(f"Processed {len(results)} files")
    logger.info(f"Manifest saved to {args.output_manifest}")
    logger.info(f"Parquet saved to {output_parquet}")

    # Label distribution
    label_counts = df["label"].value_counts()
    logger.info("Label distribution:")
    for label, count in label_counts.items():
        logger.info(f"  {label}: {count}")


if __name__ == "__main__":
    main()
