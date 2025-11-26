#!/usr/bin/env python3
"""
Generate All Condition Variants (Band Filters + RIR Reverb)

Generates band-filtered and reverberant versions of base 1000ms clips.

Band Filters:
- telephony: bandpass 300-3400 Hz (ITU-T standard)
- lp3400: lowpass 3400 Hz (no high-pass)
- hp300: highpass 300 Hz (no low-pass)

Reverb (RIR):
- T60 0.0-0.4s: Low reverb (small rooms)
- T60 0.4-0.8s: Medium reverb (office/classroom)
- T60 0.8-1.5s: High reverb (large rooms/halls)

Usage:
    python scripts/generate_conditions.py \
        --input_manifest data/processed/clean_clips/clean_metadata.csv \
        --output_dir data/processed/conditions_final \
        --rir_root data/external/rirs_noises

Note: RIR dataset from OpenSLR SLR28 (download separately if not available)
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

from src.qsm.audio.filters import apply_bandpass, apply_lowpass, apply_highpass


def normalize_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio to target RMS."""
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-8:
        return audio * (target_rms / rms)
    return audio


def generate_band_variants(
    base_clips_df: pd.DataFrame,
    output_dir: Path,
    sr: int = 16000,
) -> list[dict]:
    """
    Generate band-filtered variants of base clips.

    Args:
        base_clips_df: DataFrame with audio_path, clip_id, label columns
        output_dir: Output directory for audio files
        sr: Sample rate

    Returns:
        List of metadata dicts for generated variants
    """
    band_dir = output_dir / "band"
    band_dir.mkdir(parents=True, exist_ok=True)

    band_configs = {
        "telephony": {"filter_fn": apply_bandpass, "kwargs": {"lowcut": 300, "highcut": 3400}},
        "lp3400": {"filter_fn": apply_lowpass, "kwargs": {"highcut": 3400}},
        "hp300": {"filter_fn": apply_highpass, "kwargs": {"lowcut": 300}},
    }

    results = []

    for _, row in tqdm(base_clips_df.iterrows(), total=len(base_clips_df), desc="Band filters"):
        audio_path = row["audio_path"]
        clip_id = row["clip_id"]
        label = row.get("label", row.get("ground_truth", "UNKNOWN"))

        # Load audio
        try:
            audio, file_sr = sf.read(audio_path, dtype="float32")
            if file_sr != sr:
                # Simple resample
                from scipy import signal
                num_samples = int(len(audio) * sr / file_sr)
                audio = signal.resample(audio, num_samples)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        # Apply each filter
        for band_name, config in band_configs.items():
            filtered = config["filter_fn"](audio, sr, **config["kwargs"])
            filtered = normalize_rms(filtered, target_rms=0.1)

            # Save
            out_filename = f"{clip_id}_band{band_name}.wav"
            out_path = band_dir / out_filename
            sf.write(out_path, filtered, sr)

            results.append({
                "clip_id": clip_id,
                "original_path": str(audio_path),
                "duration_ms": 1000,  # Base clips are 1000ms
                "label": label if label != "NON-SPEECH" else "NONSPEECH",
                "variant_type": "band",
                "snr_db": None,
                "band_filter": band_name,
                "rir_id": None,
                "T60": None,
                "T60_bin": None,
                "audio_path": str(out_path),
                "ground_truth": label if label != "NON-SPEECH" else "NONSPEECH",
            })

    return results


def generate_rir_variants_simple(
    base_clips_df: pd.DataFrame,
    output_dir: Path,
    sr: int = 16000,
) -> list[dict]:
    """
    Generate reverberant variants using synthetic RIRs (no external dataset needed).

    Creates synthetic RIRs for different T60 bins:
    - T60 0.0-0.4s: Short reverb
    - T60 0.4-0.8s: Medium reverb
    - T60 0.8-1.5s: Long reverb

    Args:
        base_clips_df: DataFrame with audio_path, clip_id, label columns
        output_dir: Output directory for audio files
        sr: Sample rate

    Returns:
        List of metadata dicts for generated variants
    """
    rir_dir = output_dir / "rir"
    rir_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic RIR generation using exponential decay
    def create_synthetic_rir(t60: float, sr: int = 16000, duration: float = 1.0) -> np.ndarray:
        """Create synthetic RIR with specified T60."""
        n_samples = int(duration * sr)
        t = np.arange(n_samples) / sr

        # Exponential decay based on T60 (time for 60dB decay)
        if t60 > 0.01:
            decay_rate = 3 * np.log(10) / t60  # ln(1000)/T60
            envelope = np.exp(-decay_rate * t)
        else:
            envelope = np.zeros(n_samples)
            envelope[0] = 1.0

        # Random noise modulated by envelope
        np.random.seed(int(t60 * 1000))  # Reproducible
        noise = np.random.randn(n_samples)
        rir = noise * envelope

        # Normalize
        rir = rir / (np.max(np.abs(rir)) + 1e-8)

        return rir.astype(np.float32)

    # T60 configurations
    t60_configs = [
        {"bin": "T60_0.0-0.4", "t60": 0.2},
        {"bin": "T60_0.4-0.8", "t60": 0.6},
        {"bin": "T60_0.8-1.5", "t60": 1.1},
    ]

    results = []

    for _, row in tqdm(base_clips_df.iterrows(), total=len(base_clips_df), desc="Reverb (RIR)"):
        audio_path = row["audio_path"]
        clip_id = row["clip_id"]
        label = row.get("label", row.get("ground_truth", "UNKNOWN"))

        # Load audio
        try:
            audio, file_sr = sf.read(audio_path, dtype="float32")
            if file_sr != sr:
                from scipy import signal
                num_samples = int(len(audio) * sr / file_sr)
                audio = signal.resample(audio, num_samples)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        # Apply each RIR
        for t60_cfg in t60_configs:
            t60_bin = t60_cfg["bin"]
            t60_value = t60_cfg["t60"]

            # Create synthetic RIR
            rir = create_synthetic_rir(t60_value, sr)

            # Convolve
            from scipy import signal
            reverb = signal.fftconvolve(audio, rir, mode="full")[:len(audio)]

            # Normalize to preserve energy
            rms_orig = np.sqrt(np.mean(audio**2))
            rms_reverb = np.sqrt(np.mean(reverb**2))
            if rms_reverb > 1e-8:
                reverb = reverb * (rms_orig / rms_reverb)

            reverb = normalize_rms(reverb, target_rms=0.1)

            # Save
            out_filename = f"{clip_id}_rir_{t60_bin}.wav"
            out_path = rir_dir / out_filename
            sf.write(out_path, reverb.astype(np.float32), sr)

            results.append({
                "clip_id": clip_id,
                "original_path": str(audio_path),
                "duration_ms": 1000,
                "label": label if label != "NON-SPEECH" else "NONSPEECH",
                "variant_type": "rir",
                "snr_db": None,
                "band_filter": None,
                "rir_id": f"synthetic_{t60_bin}",
                "T60": t60_value,
                "T60_bin": t60_bin,
                "audio_path": str(out_path),
                "ground_truth": label if label != "NON-SPEECH" else "NONSPEECH",
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate band and RIR condition variants")
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=Path("data/processed/clean_clips/clean_metadata.csv"),
        help="Input manifest with base 1000ms clips",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed/conditions_final"),
        help="Output directory for generated variants",
    )
    parser.add_argument(
        "--max_clips",
        type=int,
        default=0,
        help="Maximum clips to process (0 = all)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate",
    )

    args = parser.parse_args()

    print("="*60)
    print("GENERATE CONDITION VARIANTS (BAND + RIR)")
    print("="*60)

    # Load base clips
    print(f"\nLoading base clips from: {args.input_manifest}")

    if args.input_manifest.suffix == ".csv":
        base_df = pd.read_csv(args.input_manifest)
    else:
        base_df = pd.read_parquet(args.input_manifest)

    # Ensure required columns
    if "clip_id" not in base_df.columns:
        # Generate clip_id from audio_path
        base_df["clip_id"] = base_df["audio_path"].apply(
            lambda x: Path(x).stem.replace("_1000ms", "")
        )

    print(f"Loaded {len(base_df)} base clips")

    # Filter by max_clips if set
    if args.max_clips > 0:
        base_df = base_df.head(args.max_clips)
        print(f"Using first {args.max_clips} clips")

    # Resolve audio paths
    def resolve_path(p):
        p = Path(p)
        if p.is_file():
            return str(p)
        # Try with data/ prefix
        candidate = Path("data") / p
        if candidate.is_file():
            return str(candidate)
        # Try from project root
        candidate = project_root / p
        if candidate.is_file():
            return str(candidate)
        return str(p)

    base_df["audio_path"] = base_df["audio_path"].apply(resolve_path)

    # Verify files exist
    exists = base_df["audio_path"].apply(lambda x: Path(x).is_file())
    print(f"Files found: {exists.mean():.1%} ({exists.sum()}/{len(base_df)})")

    if exists.mean() < 0.5:
        print("\nERROR: Most audio files not found!")
        print("First 5 missing:")
        for p in base_df[~exists]["audio_path"].head(5):
            print(f"  - {p}")
        return 1

    base_df = base_df[exists].copy()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Generate band variants
    print("\n--- Generating Band Filter Variants ---")
    band_results = generate_band_variants(base_df, args.output_dir, args.sr)
    all_results.extend(band_results)
    print(f"Generated {len(band_results)} band variants")

    # Generate RIR variants
    print("\n--- Generating RIR Reverb Variants ---")
    rir_results = generate_rir_variants_simple(base_df, args.output_dir, args.sr)
    all_results.extend(rir_results)
    print(f"Generated {len(rir_results)} RIR variants")

    # Save manifest
    results_df = pd.DataFrame(all_results)
    manifest_path = args.output_dir / "band_rir_manifest.parquet"
    results_df.to_parquet(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path}")

    # Also save CSV for easier inspection
    csv_path = args.output_dir / "band_rir_manifest.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total variants generated: {len(results_df)}")
    print(f"\nBy variant type:")
    print(results_df["variant_type"].value_counts().to_string())
    print(f"\nBy band_filter:")
    print(results_df[results_df["variant_type"]=="band"]["band_filter"].value_counts().to_string())
    print(f"\nBy T60_bin:")
    print(results_df[results_df["variant_type"]=="rir"]["T60_bin"].value_counts().to_string())
    print(f"\nBy label:")
    print(results_df["ground_truth"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
