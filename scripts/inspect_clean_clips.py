"""Inspect what's actually in the 'clean' clips."""

import sys
from pathlib import Path
import pandas as pd
import soundfile as sf
import numpy as np

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

def main():
    print("=" * 80)
    print("INSPECTING CLEAN CLIPS")
    print("=" * 80)

    train_csv = project_root / "data" / "processed" / "clean_clips" / "train_metadata.csv"
    df = pd.read_csv(train_csv)

    # Check a few SPEECH samples
    speech_samples = df[df['ground_truth'] == 'SPEECH'].head(3)

    print("\n📊 SPEECH Samples Analysis:")
    print("-" * 80)

    for idx, row in speech_samples.iterrows():
        audio_path = project_root / row['audio_path']

        if not audio_path.exists():
            print(f"\n[SKIP] {audio_path.name} - not found")
            continue

        audio, sr = sf.read(audio_path)

        rms = np.sqrt(np.mean(audio ** 2))
        duration_s = len(audio) / sr
        max_amp = np.abs(audio).max()

        # Calculate what % of the audio is "active" (above threshold)
        threshold = 0.01
        active_samples = np.sum(np.abs(audio) > threshold)
        active_pct = 100 * active_samples / len(audio)

        print(f"\n{audio_path.name}")
        print(f"  Original variant: {row['original_variant']}")
        print(f"  Duration: {duration_s:.3f}s ({len(audio)} samples)")
        print(f"  RMS: {rms:.6f}")
        print(f"  Max amplitude: {max_amp:.6f}")
        print(f"  Active audio (>{threshold}): {active_pct:.1f}%")

        if rms < 0.005:
            print(f"  ⚠️  Very low RMS - might be too quiet")
        if active_pct < 30:
            print(f"  ⚠️  Less than 30% active audio - mostly silence?")

    # Compare with NONSPEECH
    print("\n" + "=" * 80)
    print("\n📊 NONSPEECH Samples Analysis (for comparison):")
    print("-" * 80)

    nonspeech_samples = df[df['ground_truth'] == 'NONSPEECH'].head(3)

    for idx, row in nonspeech_samples.iterrows():
        audio_path = project_root / row['audio_path']

        if not audio_path.exists():
            continue

        audio, sr = sf.read(audio_path)

        rms = np.sqrt(np.mean(audio ** 2))
        duration_s = len(audio) / sr
        max_amp = np.abs(audio).max()

        threshold = 0.01
        active_samples = np.sum(np.abs(audio) > threshold)
        active_pct = 100 * active_samples / len(audio)

        print(f"\n{audio_path.name}")
        print(f"  Original variant: {row['original_variant']}")
        print(f"  Duration: {duration_s:.3f}s")
        print(f"  RMS: {rms:.6f}")
        print(f"  Max amplitude: {max_amp:.6f}")
        print(f"  Active audio (>{threshold}): {active_pct:.1f}%")

    print("\n" + "=" * 80)
    print("\nConclusion:")
    print("  If SPEECH and NONSPEECH have similar RMS/activity levels,")
    print("  the model can't distinguish them based on these features alone.")
    print("=" * 80)

if __name__ == "__main__":
    main()
