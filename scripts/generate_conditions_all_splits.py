#!/usr/bin/env python3
"""
Generate ALL Condition Variants from ALL Data Splits (train + dev + test)

This generates a much larger evaluation set for better statistical power.
Combines train_base.csv, dev_base.csv, and test_base.csv.

Expected output: ~150+ base clips × 27 conditions = ~4000+ samples
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


def load_all_manifests(base_dir: Path) -> pd.DataFrame:
    """Load and combine all manifest files (train, dev, test)."""
    manifests = []

    for split in ["train", "dev", "test"]:
        manifest_path = base_dir / f"{split}_base.csv"
        if manifest_path.exists():
            df = pd.read_csv(manifest_path)
            df["split"] = split
            manifests.append(df)
            print(f"  {split}: {len(df)} clips")

    if not manifests:
        raise FileNotFoundError(f"No manifest files found in {base_dir}")

    combined = pd.concat(manifests, ignore_index=True)
    print(f"  Total: {len(combined)} clips")
    return combined


def resolve_path(p, project_root):
    """Resolve audio path trying multiple locations."""
    p_str = str(p).replace("\\", "/")
    p = Path(p_str)
    if p.is_file():
        return str(p)
    if p_str.startswith("processed/"):
        candidate = Path("data") / p_str
        if candidate.is_file():
            return str(candidate)
    candidate = Path("data") / p
    if candidate.is_file():
        return str(candidate)
    candidate = project_root / p
    if candidate.is_file():
        return str(candidate)
    return str(p)


def generate_duration_variants(base_clips_df, output_dir, sr=16000):
    """Generate duration variants."""
    dur_dir = output_dir / "duration"
    dur_dir.mkdir(parents=True, exist_ok=True)

    durations_ms = [20, 40, 60, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    results = []

    for _, row in tqdm(base_clips_df.iterrows(), total=len(base_clips_df), desc="Duration"):
        audio_path = row["audio_path"]
        clip_id = row["clip_id"]
        label = row.get("label", row.get("ground_truth", "UNKNOWN"))
        split = row.get("split", "unknown")

        try:
            audio, file_sr = sf.read(audio_path, dtype="float32")
            if file_sr != sr:
                from scipy import signal
                num_samples = int(len(audio) * sr / file_sr)
                audio = signal.resample(audio, num_samples)
        except Exception as e:
            continue

        for dur_ms in durations_ms:
            n_samples = int(dur_ms * sr / 1000)
            if n_samples > len(audio):
                truncated = np.pad(audio, (0, n_samples - len(audio)), mode='constant')
            else:
                start = (len(audio) - n_samples) // 2
                truncated = audio[start:start + n_samples]

            truncated = normalize_rms(truncated, target_rms=0.1)
            out_filename = f"{clip_id}_dur{dur_ms}ms.wav"
            out_path = dur_dir / out_filename
            sf.write(out_path, truncated.astype(np.float32), sr)

            results.append({
                "clip_id": clip_id,
                "split": split,
                "duration_ms": dur_ms,
                "variant_type": "duration",
                "snr_db": None,
                "band_filter": None,
                "T60_bin": None,
                "audio_path": str(out_path),
                "ground_truth": label if label != "NON-SPEECH" else "NONSPEECH",
            })

    return results


def generate_snr_variants(base_clips_df, output_dir, sr=16000):
    """Generate SNR variants."""
    snr_dir = output_dir / "snr"
    snr_dir.mkdir(parents=True, exist_ok=True)

    snr_levels = [-10, -5, 0, 5, 10, 15, 20]
    results = []
    np.random.seed(42)

    for _, row in tqdm(base_clips_df.iterrows(), total=len(base_clips_df), desc="SNR"):
        audio_path = row["audio_path"]
        clip_id = row["clip_id"]
        label = row.get("label", row.get("ground_truth", "UNKNOWN"))
        split = row.get("split", "unknown")

        try:
            audio, file_sr = sf.read(audio_path, dtype="float32")
            if file_sr != sr:
                from scipy import signal
                num_samples = int(len(audio) * sr / file_sr)
                audio = signal.resample(audio, num_samples)
        except Exception as e:
            continue

        signal_power = np.mean(audio**2)

        for snr_db in snr_levels:
            snr_linear = 10 ** (snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
            noisy = audio + noise
            noisy = normalize_rms(noisy, target_rms=0.1)

            out_filename = f"{clip_id}_snr{snr_db:+d}dB.wav"
            out_path = snr_dir / out_filename
            sf.write(out_path, noisy.astype(np.float32), sr)

            results.append({
                "clip_id": clip_id,
                "split": split,
                "duration_ms": 1000,
                "variant_type": "snr",
                "snr_db": snr_db,
                "band_filter": None,
                "T60_bin": None,
                "audio_path": str(out_path),
                "ground_truth": label if label != "NON-SPEECH" else "NONSPEECH",
            })

    return results


def generate_band_variants(base_clips_df, output_dir, sr=16000):
    """Generate band filter variants."""
    band_dir = output_dir / "band"
    band_dir.mkdir(parents=True, exist_ok=True)

    band_configs = {
        "telephony": {"filter_fn": apply_bandpass, "kwargs": {"lowcut": 300, "highcut": 3400}},
        "lp3400": {"filter_fn": apply_lowpass, "kwargs": {"highcut": 3400}},
        "hp300": {"filter_fn": apply_highpass, "kwargs": {"lowcut": 300}},
    }

    results = []

    for _, row in tqdm(base_clips_df.iterrows(), total=len(base_clips_df), desc="Band"):
        audio_path = row["audio_path"]
        clip_id = row["clip_id"]
        label = row.get("label", row.get("ground_truth", "UNKNOWN"))
        split = row.get("split", "unknown")

        try:
            audio, file_sr = sf.read(audio_path, dtype="float32")
            if file_sr != sr:
                from scipy import signal
                num_samples = int(len(audio) * sr / file_sr)
                audio = signal.resample(audio, num_samples)
        except Exception as e:
            continue

        for band_name, config in band_configs.items():
            filtered = config["filter_fn"](audio, sr, **config["kwargs"])
            filtered = normalize_rms(filtered, target_rms=0.1)

            out_filename = f"{clip_id}_band{band_name}.wav"
            out_path = band_dir / out_filename
            sf.write(out_path, filtered, sr)

            results.append({
                "clip_id": clip_id,
                "split": split,
                "duration_ms": 1000,
                "variant_type": "band",
                "snr_db": None,
                "band_filter": band_name,
                "T60_bin": None,
                "audio_path": str(out_path),
                "ground_truth": label if label != "NON-SPEECH" else "NONSPEECH",
            })

    return results


def generate_rir_variants(base_clips_df, output_dir, sr=16000):
    """Generate RIR reverb variants."""
    rir_dir = output_dir / "rir"
    rir_dir.mkdir(parents=True, exist_ok=True)

    def create_synthetic_rir(t60, sr=16000, duration=1.0):
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

    for _, row in tqdm(base_clips_df.iterrows(), total=len(base_clips_df), desc="RIR"):
        audio_path = row["audio_path"]
        clip_id = row["clip_id"]
        label = row.get("label", row.get("ground_truth", "UNKNOWN"))
        split = row.get("split", "unknown")

        try:
            audio, file_sr = sf.read(audio_path, dtype="float32")
            if file_sr != sr:
                from scipy import signal
                num_samples = int(len(audio) * sr / file_sr)
                audio = signal.resample(audio, num_samples)
        except Exception as e:
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
                "split": split,
                "duration_ms": 1000,
                "variant_type": "rir",
                "snr_db": None,
                "band_filter": None,
                "T60_bin": t60_bin,
                "audio_path": str(out_path),
                "ground_truth": label if label != "NON-SPEECH" else "NONSPEECH",
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=Path("data/processed/base_1000ms"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/conditions_all_splits"))
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    print("="*60)
    print("GENERATE CONDITIONS FROM ALL SPLITS")
    print("="*60)

    # Load all manifests
    print("\nLoading manifests...")
    base_df = load_all_manifests(args.base_dir)

    # Ensure columns
    if "clip_id" not in base_df.columns:
        base_df["clip_id"] = base_df["audio_path"].apply(
            lambda x: Path(x).stem.replace("_1000ms", "")
        )
    if "ground_truth" in base_df.columns and "label" not in base_df.columns:
        base_df["label"] = base_df["ground_truth"]

    # Resolve paths
    base_df["audio_path"] = base_df["audio_path"].apply(lambda p: resolve_path(p, project_root))

    # Verify
    exists = base_df["audio_path"].apply(lambda x: Path(x).is_file())
    print(f"\nFiles found: {exists.mean():.1%} ({exists.sum()}/{len(base_df)})")

    if exists.mean() < 0.5:
        print("ERROR: Most files not found!")
        return 1

    base_df = base_df[exists].copy()

    # Clean output
    import shutil
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all variants
    all_results = []

    print("\n--- Duration Variants ---")
    all_results.extend(generate_duration_variants(base_df, args.output_dir, args.sr))

    print("\n--- SNR Variants ---")
    all_results.extend(generate_snr_variants(base_df, args.output_dir, args.sr))

    print("\n--- Band Variants ---")
    all_results.extend(generate_band_variants(base_df, args.output_dir, args.sr))

    print("\n--- RIR Variants ---")
    all_results.extend(generate_rir_variants(base_df, args.output_dir, args.sr))

    # Save
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(args.output_dir / "all_conditions_manifest.parquet", index=False)
    results_df.to_csv(args.output_dir / "all_conditions_manifest.csv", index=False)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(results_df)}")
    print(f"\nBy split:")
    print(results_df["split"].value_counts().to_string())
    print(f"\nBy variant type:")
    print(results_df["variant_type"].value_counts().to_string())
    print(f"\nBy label:")
    print(results_df["ground_truth"].value_counts().to_string())

    # Samples per condition
    print(f"\nSamples per duration level: {len(results_df[results_df['variant_type']=='duration']) // 14}")
    print(f"Samples per SNR level: {len(results_df[results_df['variant_type']=='snr']) // 7}")
    print(f"Samples per band filter: {len(results_df[results_df['variant_type']=='band']) // 3}")
    print(f"Samples per T60 bin: {len(results_df[results_df['variant_type']=='rir']) // 3}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
