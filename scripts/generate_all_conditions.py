#!/usr/bin/env python3
"""
Generate ALL Condition Variants (Duration, SNR, Band Filters, RIR Reverb)

Generates all condition variants from base 1000ms clips:

Duration variants:
- 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ms

SNR variants (with white noise):
- -10, -5, 0, 5, 10, 15, 20 dB

Band Filters:
- telephony: bandpass 300-3400 Hz (ITU-T standard)
- lp3400: lowpass 3400 Hz (no high-pass)
- hp300: highpass 300 Hz (no low-pass)

Reverb (RIR):
- T60 0.0-0.4s: Low reverb (small rooms)
- T60 0.4-0.8s: Medium reverb (office/classroom)
- T60 0.8-1.5s: High reverb (large rooms/halls)

Usage:
    python scripts/generate_all_conditions.py \
        --input_manifest data/processed/base_1000ms/test_base.csv \
        --output_dir data/processed/conditions_final \
        --sr 16000
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


def generate_duration_variants(
    base_clips_df: pd.DataFrame,
    output_dir: Path,
    sr: int = 16000,
) -> list[dict]:
    """
    Generate duration variants by truncating base clips.

    Durations: 20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ms
    """
    dur_dir = output_dir / "duration"
    dur_dir.mkdir(parents=True, exist_ok=True)

    durations_ms = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    results = []

    for _, row in tqdm(base_clips_df.iterrows(), total=len(base_clips_df), desc="Duration variants"):
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

        # Generate each duration variant
        for dur_ms in durations_ms:
            n_samples = int(dur_ms * sr / 1000)

            if n_samples > len(audio):
                # Pad with zeros if needed
                truncated = np.pad(audio, (0, n_samples - len(audio)), mode='constant')
            else:
                # Take from center for better representation
                start = (len(audio) - n_samples) // 2
                truncated = audio[start:start + n_samples]

            truncated = normalize_rms(truncated, target_rms=0.1)

            # Save
            out_filename = f"{clip_id}_dur{dur_ms}ms.wav"
            out_path = dur_dir / out_filename
            sf.write(out_path, truncated.astype(np.float32), sr)

            results.append({
                "clip_id": clip_id,
                "original_path": str(audio_path),
                "duration_ms": dur_ms,
                "label": label if label != "NON-SPEECH" else "NONSPEECH",
                "variant_type": "duration",
                "snr_db": None,
                "band_filter": None,
                "rir_id": None,
                "T60": None,
                "T60_bin": None,
                "audio_path": str(out_path),
                "ground_truth": label if label != "NON-SPEECH" else "NONSPEECH",
            })

    return results


def generate_snr_variants(
    base_clips_df: pd.DataFrame,
    output_dir: Path,
    sr: int = 16000,
) -> list[dict]:
    """
    Generate SNR variants by adding white noise.

    SNR levels: -10, -5, 0, 5, 10, 15, 20 dB
    """
    snr_dir = output_dir / "snr"
    snr_dir.mkdir(parents=True, exist_ok=True)

    snr_levels = [-10, -5, 0, 5, 10, 15, 20]
    results = []

    np.random.seed(42)  # Reproducibility

    for _, row in tqdm(base_clips_df.iterrows(), total=len(base_clips_df), desc="SNR variants"):
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

        # Calculate signal power
        signal_power = np.mean(audio**2)

        # Generate each SNR variant
        for snr_db in snr_levels:
            # Calculate required noise power
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear

            # Generate white noise
            noise = np.random.randn(len(audio)) * np.sqrt(noise_power)

            # Mix
            noisy = audio + noise
            noisy = normalize_rms(noisy, target_rms=0.1)

            # Save
            out_filename = f"{clip_id}_snr{snr_db:+d}dB.wav"
            out_path = snr_dir / out_filename
            sf.write(out_path, noisy.astype(np.float32), sr)

            results.append({
                "clip_id": clip_id,
                "original_path": str(audio_path),
                "duration_ms": 1000,  # Base clips are 1000ms
                "label": label if label != "NON-SPEECH" else "NONSPEECH",
                "variant_type": "snr",
                "snr_db": snr_db,
                "band_filter": None,
                "rir_id": None,
                "T60": None,
                "T60_bin": None,
                "audio_path": str(out_path),
                "ground_truth": label if label != "NON-SPEECH" else "NONSPEECH",
            })

    return results


def generate_band_variants(
    base_clips_df: pd.DataFrame,
    output_dir: Path,
    sr: int = 16000,
) -> list[dict]:
    """
    Generate band-filtered variants of base clips.
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
                "duration_ms": 1000,
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


def generate_rir_variants(
    base_clips_df: pd.DataFrame,
    output_dir: Path,
    sr: int = 16000,
) -> list[dict]:
    """
    Generate reverberant variants using synthetic RIRs.
    """
    rir_dir = output_dir / "rir"
    rir_dir.mkdir(parents=True, exist_ok=True)

    def create_synthetic_rir(t60: float, sr: int = 16000, duration: float = 1.0) -> np.ndarray:
        """Create synthetic RIR with specified T60."""
        n_samples = int(duration * sr)
        t = np.arange(n_samples) / sr

        if t60 > 0.01:
            decay_rate = 3 * np.log(10) / t60
            envelope = np.exp(-decay_rate * t)
        else:
            envelope = np.zeros(n_samples)
            envelope[0] = 1.0

        np.random.seed(int(t60 * 1000))
        noise = np.random.randn(n_samples)
        rir = noise * envelope
        rir = rir / (np.max(np.abs(rir)) + 1e-8)

        return rir.astype(np.float32)

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

        try:
            audio, file_sr = sf.read(audio_path, dtype="float32")
            if file_sr != sr:
                from scipy import signal
                num_samples = int(len(audio) * sr / file_sr)
                audio = signal.resample(audio, num_samples)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue

        for t60_cfg in t60_configs:
            t60_bin = t60_cfg["bin"]
            t60_value = t60_cfg["t60"]

            rir = create_synthetic_rir(t60_value, sr)

            from scipy import signal
            reverb = signal.fftconvolve(audio, rir, mode="full")[:len(audio)]

            rms_orig = np.sqrt(np.mean(audio**2))
            rms_reverb = np.sqrt(np.mean(reverb**2))
            if rms_reverb > 1e-8:
                reverb = reverb * (rms_orig / rms_reverb)

            reverb = normalize_rms(reverb, target_rms=0.1)

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


def resolve_path(p, project_root):
    """Resolve audio path trying multiple locations."""
    p_str = str(p).replace("\\", "/")
    p = Path(p_str)
    if p.is_file():
        return str(p)
    # Try with data/ prefix (for paths starting with "processed/")
    if p_str.startswith("processed/"):
        candidate = Path("data") / p_str
        if candidate.is_file():
            return str(candidate)
    # Try with data/ prefix
    candidate = Path("data") / p
    if candidate.is_file():
        return str(candidate)
    # Try from project root
    candidate = project_root / p
    if candidate.is_file():
        return str(candidate)
    return str(p)


def main():
    parser = argparse.ArgumentParser(description="Generate ALL condition variants")
    parser.add_argument(
        "--input_manifest",
        type=Path,
        default=Path("data/processed/base_1000ms/test_base.csv"),
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
    parser.add_argument(
        "--skip_duration",
        action="store_true",
        help="Skip duration variant generation",
    )
    parser.add_argument(
        "--skip_snr",
        action="store_true",
        help="Skip SNR variant generation",
    )
    parser.add_argument(
        "--skip_band",
        action="store_true",
        help="Skip band filter variant generation",
    )
    parser.add_argument(
        "--skip_rir",
        action="store_true",
        help="Skip RIR variant generation",
    )

    args = parser.parse_args()

    print("="*60)
    print("GENERATE ALL CONDITION VARIANTS")
    print("="*60)

    # Load base clips
    print(f"\nLoading base clips from: {args.input_manifest}")

    if args.input_manifest.suffix == ".csv":
        base_df = pd.read_csv(args.input_manifest)
    else:
        base_df = pd.read_parquet(args.input_manifest)

    # Ensure required columns
    if "clip_id" not in base_df.columns:
        base_df["clip_id"] = base_df["audio_path"].apply(
            lambda x: Path(x).stem.replace("_1000ms", "")
        )

    if "ground_truth" in base_df.columns and "label" not in base_df.columns:
        base_df["label"] = base_df["ground_truth"]

    print(f"Loaded {len(base_df)} base clips")

    if args.max_clips > 0:
        base_df = base_df.head(args.max_clips)
        print(f"Using first {args.max_clips} clips")

    # Resolve audio paths
    base_df["audio_path"] = base_df["audio_path"].apply(lambda p: resolve_path(p, project_root))

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

    # Generate duration variants
    if not args.skip_duration:
        print("\n--- Generating Duration Variants ---")
        dur_results = generate_duration_variants(base_df, args.output_dir, args.sr)
        all_results.extend(dur_results)
        print(f"Generated {len(dur_results)} duration variants")

    # Generate SNR variants
    if not args.skip_snr:
        print("\n--- Generating SNR Variants ---")
        snr_results = generate_snr_variants(base_df, args.output_dir, args.sr)
        all_results.extend(snr_results)
        print(f"Generated {len(snr_results)} SNR variants")

    # Generate band variants
    if not args.skip_band:
        print("\n--- Generating Band Filter Variants ---")
        band_results = generate_band_variants(base_df, args.output_dir, args.sr)
        all_results.extend(band_results)
        print(f"Generated {len(band_results)} band variants")

    # Generate RIR variants
    if not args.skip_rir:
        print("\n--- Generating RIR Reverb Variants ---")
        rir_results = generate_rir_variants(base_df, args.output_dir, args.sr)
        all_results.extend(rir_results)
        print(f"Generated {len(rir_results)} RIR variants")

    # Save manifest
    results_df = pd.DataFrame(all_results)
    manifest_path = args.output_dir / "all_conditions_manifest.parquet"
    results_df.to_parquet(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path}")

    csv_path = args.output_dir / "all_conditions_manifest.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total variants generated: {len(results_df)}")
    print(f"\nBy variant type:")
    print(results_df["variant_type"].value_counts().to_string())
    if "duration" in results_df["variant_type"].values:
        print(f"\nBy duration_ms:")
        print(results_df[results_df["variant_type"]=="duration"]["duration_ms"].value_counts().sort_index().to_string())
    if "snr" in results_df["variant_type"].values:
        print(f"\nBy snr_db:")
        print(results_df[results_df["variant_type"]=="snr"]["snr_db"].value_counts().sort_index().to_string())
    if "band" in results_df["variant_type"].values:
        print(f"\nBy band_filter:")
        print(results_df[results_df["variant_type"]=="band"]["band_filter"].value_counts().to_string())
    if "rir" in results_df["variant_type"].values:
        print(f"\nBy T60_bin:")
        print(results_df[results_df["variant_type"]=="rir"]["T60_bin"].value_counts().to_string())
    print(f"\nBy label:")
    print(results_df["ground_truth"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
