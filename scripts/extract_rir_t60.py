#!/usr/bin/env python3
"""
Extract T60 (reverberation time) from RIR database.

T60 is defined as the time it takes for the sound energy to decay by 60 dB
after the direct sound arrives.

Uses Schroeder integration method:
1. Find direct sound peak
2. Compute backward-integrated energy decay curve
3. Fit linear regression to decay (typically -5 to -25 dB range)
4. Extrapolate to -60 dB to get T60
"""

import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
import json
from tqdm import tqdm
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def estimate_t60(rir: np.ndarray, sr: int, method: str = "edt") -> float:
    """
    Estimate T60 from RIR using Schroeder integration.

    Args:
        rir: Room impulse response (samples,)
        sr: Sample rate
        method: 'edt' (Early Decay Time, 0 to -10 dB) or 't30' (-5 to -35 dB, extrapolated)

    Returns:
        T60 in seconds
    """
    # Ensure mono
    if rir.ndim > 1:
        rir = rir.mean(axis=1)

    # Square and reverse for backward integration
    rir_squared = rir ** 2

    # Schroeder backward integration
    energy = np.cumsum(rir_squared[::-1])[::-1]

    # Convert to dB
    energy_db = 10 * np.log10(energy / energy.max() + 1e-10)

    # Find direct sound peak
    peak_idx = np.argmax(np.abs(rir))

    # Only consider decay after peak
    decay_db = energy_db[peak_idx:]
    time_axis = np.arange(len(decay_db)) / sr

    if method == "edt":
        # Early Decay Time: fit 0 to -10 dB
        start_db = 0
        end_db = -10
    else:  # t30
        # T30: fit -5 to -35 dB (extrapolate to -60)
        start_db = -5
        end_db = -35

    # Find indices for fitting range
    start_idx = np.argmax(decay_db <= start_db)
    end_idx = np.argmax(decay_db <= end_db)

    if start_idx >= end_idx or end_idx == 0:
        # Decay too fast or not enough data
        return 0.0

    # Linear regression on decay
    t_fit = time_axis[start_idx:end_idx]
    db_fit = decay_db[start_idx:end_idx]

    if len(t_fit) < 2:
        return 0.0

    # Fit line: db = a*t + b
    poly = np.polyfit(t_fit, db_fit, 1)
    slope = poly[0]  # dB/sec

    if slope >= 0:
        # Non-decaying
        return 0.0

    if method == "edt":
        # EDT: time to decay 10 dB, extrapolate to 60 dB
        t60 = 60.0 / abs(slope) * (10.0 / 60.0) * 6
    else:  # t30
        # T30: time to decay 30 dB, extrapolate to 60 dB
        t60 = 60.0 / abs(slope) * (30.0 / 60.0) * 2

    return t60


def process_rir_database(
    rir_root: Path,
    output_json: Path,
    method: str = "t30",
    max_files: int = None,
):
    """
    Process all RIRs in database and extract T60.

    Args:
        rir_root: Root directory of RIR dataset (RIRS_NOISES)
        output_json: Output JSON path for metadata
        method: T60 estimation method ('edt' or 't30')
        max_files: Maximum number of files to process (for testing)
    """
    metadata = {}

    # Find all WAV files
    rir_files = []
    for pattern in ["simulated_rirs/**/*.wav", "real_rirs_isotropic_noises/*.wav"]:
        rir_files.extend(rir_root.glob(pattern))

    if max_files:
        rir_files = rir_files[:max_files]

    print(f"Processing {len(rir_files)} RIR files...")

    for rir_path in tqdm(rir_files, desc="Extracting T60"):
        try:
            # Load RIR
            rir, sr = sf.read(rir_path, dtype="float32")

            # Estimate T60
            t60 = estimate_t60(rir, sr, method=method)

            # Create relative path as ID
            rir_id = str(rir_path.relative_to(rir_root))

            # Store metadata
            metadata[rir_id] = {
                "path": str(rir_path),
                "T60": float(t60),
                "sr": int(sr),
                "duration_sec": len(rir) / sr,
                "method": method,
            }

        except Exception as e:
            print(f"\nError processing {rir_path}: {e}")
            continue

    # Save metadata
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to: {output_json}")

    # Print statistics
    t60_values = [meta["T60"] for meta in metadata.values() if meta["T60"] > 0]
    if t60_values:
        print(f"\nT60 Statistics:")
        print(f"  Count: {len(t60_values)}")
        print(f"  Mean: {np.mean(t60_values):.3f} s")
        print(f"  Median: {np.median(t60_values):.3f} s")
        print(f"  Min: {np.min(t60_values):.3f} s")
        print(f"  Max: {np.max(t60_values):.3f} s")
        print(f"  Std: {np.std(t60_values):.3f} s")

        # Histogram
        print(f"\nT60 Distribution:")
        bins = [0, 0.3, 0.6, 1.0, 1.5, 2.0, 100]
        labels = ["<0.3s", "0.3-0.6s", "0.6-1.0s", "1.0-1.5s", "1.5-2.0s", ">2.0s"]
        hist, _ = np.histogram(t60_values, bins=bins)
        for label, count in zip(labels, hist):
            print(f"  {label:12s}: {count:5d} ({count/len(t60_values)*100:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Extract T60 from RIR database")
    parser.add_argument(
        "--rir_root",
        type=Path,
        default=Path("data/external/RIRS_NOISES/RIRS_NOISES"),
        help="Root directory of RIR dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/RIRS_NOISES/rir_metadata.json"),
        help="Output JSON for metadata",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["edt", "t30"],
        default="t30",
        help="T60 estimation method",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum files to process (for testing)",
    )
    args = parser.parse_args()

    if not args.rir_root.exists():
        print(f"ERROR: RIR root not found: {args.rir_root}")
        print("\nPlease download the dataset first:")
        print("  python scripts/download_rirs.py")
        sys.exit(1)

    process_rir_database(
        rir_root=args.rir_root,
        output_json=args.output,
        method=args.method,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
