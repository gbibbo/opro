#!/usr/bin/env python3
"""
Generate Extended Test Set for Robust Evaluation

Creates 150-200 balanced clips covering:
- Durations: 200ms, 400ms, 600ms, 800ms, 1000ms
- SNRs: 0dB, 5dB, 10dB, 15dB, 20dB
- Classes: SPEECH (50%), NONSPEECH (50%)
- Sources: VoxConverse (speech), MUSAN/ESC-50 (nonspeech)

This provides statistical power to distinguish models with p<0.05.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random
import argparse


def load_source_datasets(
    voxconverse_csv: Path,
    musan_csv: Path,
    min_duration_ms: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load source datasets and filter by minimum duration."""

    # Load VoxConverse (speech)
    vox_df = pd.read_csv(voxconverse_csv)
    vox_df = vox_df[vox_df["duration_ms"] >= min_duration_ms].copy()

    # Load MUSAN/ESC-50 (nonspeech)
    musan_df = pd.read_csv(musan_csv)
    musan_df = musan_df[musan_df["duration_ms"] >= min_duration_ms].copy()

    print(f"Loaded {len(vox_df)} VoxConverse clips (≥{min_duration_ms}ms)")
    print(f"Loaded {len(musan_df)} MUSAN clips (≥{min_duration_ms}ms)")

    return vox_df, musan_df


def generate_factorial_design(
    n_samples_per_class: int = 100,
) -> List[Dict]:
    """
    Generate factorial design: duration × SNR × class

    Target: 200 total samples (100 SPEECH, 100 NONSPEECH)

    Factorial:
    - 5 durations × 5 SNRs = 25 conditions
    - 4 samples per condition per class = 100 samples per class
    """

    durations_ms = [200, 400, 600, 800, 1000]
    snrs_db = [0, 5, 10, 15, 20]
    classes = ["SPEECH", "NONSPEECH"]

    # Samples per condition per class
    samples_per_condition = n_samples_per_class // (len(durations_ms) * len(snrs_db))

    print(f"\nFactorial Design:")
    print(f"  Durations: {durations_ms}")
    print(f"  SNRs: {snrs_db}")
    print(f"  Samples per condition per class: {samples_per_condition}")
    print(f"  Total samples: {len(durations_ms) * len(snrs_db) * samples_per_condition * len(classes)}")

    design = []
    for cls in classes:
        for duration in durations_ms:
            for snr in snrs_db:
                for i in range(samples_per_condition):
                    design.append({
                        "class": cls,
                        "duration_ms": duration,
                        "snr_db": snr,
                        "sample_id": i,
                    })

    return design


def sample_clips_from_design(
    design: List[Dict],
    vox_df: pd.DataFrame,
    musan_df: pd.DataFrame,
    output_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample clips according to factorial design.

    For each design point:
    1. Randomly sample source clip (VoxConverse or MUSAN)
    2. Randomly select starting position (ensuring min duration)
    3. Extract clip with specified duration
    4. Add noise at specified SNR
    5. Apply peak normalization
    """

    random.seed(seed)
    np.random.seed(seed)

    metadata = []

    for i, spec in enumerate(design):
        cls = spec["class"]
        duration = spec["duration_ms"]
        snr = spec["snr_db"]
        sample_id = spec["sample_id"]

        # Sample source clip
        if cls == "SPEECH":
            source_df = vox_df
            source_type = "voxconverse"
        else:
            source_df = musan_df
            source_type = "musan"

        # Random source clip
        source_row = source_df.sample(n=1, random_state=seed + i).iloc[0]
        source_path = source_row["audio_path"]
        source_duration = source_row["duration_ms"]

        # Random starting position
        max_start = source_duration - duration
        if max_start > 0:
            start_ms = random.randint(0, max_start)
        else:
            start_ms = 0

        # Generate output filename
        output_filename = (
            f"{source_type}_{Path(source_path).stem}_"
            f"{duration}ms_snr{snr:+d}dB_s{sample_id}.wav"
        )

        metadata.append({
            "audio_path": str(output_dir / output_filename),
            "label": cls,
            "duration_ms": duration,
            "snr_db": snr,
            "source_type": source_type,
            "source_path": source_path,
            "start_ms": start_ms,
            "sample_id": sample_id,
        })

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{len(design)} sample specs...")

    return pd.DataFrame(metadata)


def main():
    parser = argparse.ArgumentParser(
        description="Generate extended test set with factorial design"
    )
    parser.add_argument(
        "--voxconverse_csv",
        type=str,
        default="data/raw/voxconverse_metadata.csv",
        help="Path to VoxConverse metadata CSV",
    )
    parser.add_argument(
        "--musan_csv",
        type=str,
        default="data/raw/musan_metadata.csv",
        help="Path to MUSAN metadata CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/extended_test_clips",
        help="Output directory for extended test clips",
    )
    parser.add_argument(
        "--n_samples_per_class",
        type=int,
        default=100,
        help="Number of samples per class (SPEECH/NONSPEECH)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vox_csv = project_root / args.voxconverse_csv
    musan_csv = project_root / args.musan_csv
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EXTENDED TEST SET GENERATION")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Target samples per class: {args.n_samples_per_class}")
    print(f"Random seed: {args.seed}")
    print()

    # Load source datasets
    print("Loading source datasets...")
    vox_df, musan_df = load_source_datasets(vox_csv, musan_csv)

    # Generate factorial design
    print("\nGenerating factorial design...")
    design = generate_factorial_design(n_samples_per_class=args.n_samples_per_class)

    # Sample clips
    print(f"\nSampling {len(design)} clips according to design...")
    metadata_df = sample_clips_from_design(
        design, vox_df, musan_df, output_dir, seed=args.seed
    )

    # Save metadata
    metadata_csv = output_dir / "extended_test_metadata.csv"
    metadata_df.to_csv(metadata_csv, index=False)
    print(f"\n✓ Metadata saved to: {metadata_csv}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(metadata_df)}")
    print(f"  SPEECH:    {len(metadata_df[metadata_df['label'] == 'SPEECH'])}")
    print(f"  NONSPEECH: {len(metadata_df[metadata_df['label'] == 'NONSPEECH'])}")
    print("\nBreakdown by duration:")
    print(metadata_df.groupby(["label", "duration_ms"]).size())
    print("\nBreakdown by SNR:")
    print(metadata_df.groupby(["label", "snr_db"]).size())
    print("\nNext steps:")
    print(f"  1. Run audio processing script to generate clips:")
    print(f"     python scripts/process_extended_test_clips.py")
    print(f"  2. Evaluate models on extended test set:")
    print(f"     python scripts/test_with_logit_scoring.py --test_csv {metadata_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
