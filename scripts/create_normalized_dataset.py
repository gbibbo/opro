"""Create dataset with NORMALIZED audio (proper RMS levels)."""

import sys
from pathlib import Path
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

def normalize_audio_peak(audio: np.ndarray, target_peak: float = 0.9, headroom_db: float = 3.0) -> np.ndarray:
    """Normalize audio by peak level, preserving relative energy differences (SNR).

    This is better than RMS normalization because it doesn't equalize the energy
    of all clips - clips with more noise will still have lower effective signal
    compared to clean clips, preserving SNR as a discriminative feature.

    Args:
        audio: Input audio array
        target_peak: Target peak level after normalization (default: 0.9 to avoid clipping)
        headroom_db: Additional headroom in dB to prevent clipping (default: 3.0)

    Returns:
        Peak-normalized audio (preserves SNR)
    """
    # Find current peak
    current_peak = np.abs(audio).max()

    if current_peak < 1e-6:  # Too quiet, skip
        return audio

    # Calculate gain to reach target peak with headroom
    headroom_factor = 10 ** (-headroom_db / 20.0)
    gain = (target_peak * headroom_factor) / current_peak

    # Apply gain
    normalized = audio * gain

    # Safety clip (should rarely trigger with proper headroom)
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized


def normalize_audio_rms(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normalize audio to target RMS level (LEGACY - destroys SNR as a feature).

    Args:
        audio: Input audio array
        target_rms: Target RMS level (default: 0.1, which is reasonable for speech)

    Returns:
        Normalized audio
    """
    current_rms = np.sqrt(np.mean(audio ** 2))

    if current_rms < 1e-6:  # Too quiet, skip
        return audio

    # Calculate gain needed
    gain = target_rms / current_rms

    # Apply gain with clipping protection
    normalized = audio * gain

    # Clip to [-1, 1] range
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized


def main():
    print("=" * 80)
    print("NORMALIZED DATASET CREATION (Peak-based, preserves SNR)")
    print("=" * 80)

    # Use clean clips as input
    clean_dir = project_root / "data" / "processed" / "clean_clips"
    output_dir = project_root / "data" / "processed" / "normalized_clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = clean_dir / "clean_metadata.csv"

    if not metadata_path.exists():
        print(f"\n[ERROR] Clean metadata not found: {metadata_path}")
        print("   Run 'python scripts/create_clean_dataset.py' first!")
        return

    print(f"\nInput:  {clean_dir}")
    print(f"Output: {output_dir}")

    # Load metadata
    df = pd.read_csv(metadata_path)
    print(f"\nTotal clean samples: {len(df)}")

    # Normalize all clips
    print(f"\nNormalizing audio clips...")

    new_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        input_path = project_root / row['audio_path']

        if not input_path.exists():
            print(f"\n[WARNING] File not found: {input_path.name}")
            continue

        # Load audio
        audio, sr = sf.read(input_path)

        # Normalize by peak (preserves SNR)
        audio_normalized = normalize_audio_peak(audio, target_peak=0.9, headroom_db=3.0)

        # Save
        output_filename = input_path.stem.replace('_clean_', '_normalized_') + '.wav'
        output_path = output_dir / output_filename

        sf.write(output_path, audio_normalized, sr)

        # Calculate new RMS
        new_rms = np.sqrt(np.mean(audio_normalized ** 2))

        # Create new record
        new_record = {
            'clip_id': row['clip_id'],
            'original_variant': row['original_variant'],
            'duration_ms': row['duration_ms'],
            'snr_db': row['snr_db'],
            'audio_path': f"data/processed/normalized_clips/{output_filename}",
            'ground_truth': row['ground_truth'],
            'dataset': row['dataset'],
            'rms': new_rms,
        }
        new_records.append(new_record)

    # Create new metadata
    df_normalized = pd.DataFrame(new_records)

    print(f"\n✓ Normalized {len(df_normalized)} clips")

    # Split into train/test (80/20)
    print(f"\nSplitting into train/test (80/20)...")
    df_train = df_normalized.sample(frac=0.8, random_state=42)
    df_test = df_normalized.drop(df_train.index)

    print(f"  Train: {len(df_train)} samples")
    print(f"  Test:  {len(df_test)} samples")

    # Save metadata
    train_csv = output_dir / "train_metadata.csv"
    test_csv = output_dir / "test_metadata.csv"
    full_csv = output_dir / "normalized_metadata.csv"

    df_normalized.to_csv(full_csv, index=False)
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"\n✓ Metadata saved:")
    print(f"  Full:  {full_csv}")
    print(f"  Train: {train_csv}")
    print(f"  Test:  {test_csv}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    speech_train = (df_train['ground_truth'] == 'SPEECH').sum()
    nonspeech_train = (df_train['ground_truth'] == 'NONSPEECH').sum()
    speech_test = (df_test['ground_truth'] == 'SPEECH').sum()
    nonspeech_test = (df_test['ground_truth'] == 'NONSPEECH').sum()

    print(f"Total normalized clips: {len(df_normalized)}")
    print(f"Train samples:          {len(df_train)}")
    print(f"  SPEECH:    {speech_train}")
    print(f"  NONSPEECH: {nonspeech_train}")
    print(f"Test samples:           {len(df_test)}")
    print(f"  SPEECH:    {speech_test}")
    print(f"  NONSPEECH: {nonspeech_test}")

    # Check RMS distribution
    avg_rms = df_normalized['rms'].mean()
    min_rms = df_normalized['rms'].min()
    max_rms = df_normalized['rms'].max()

    print(f"\nRMS levels after normalization:")
    print(f"  Mean: {avg_rms:.6f}")
    print(f"  Min:  {min_rms:.6f}")
    print(f"  Max:  {max_rms:.6f}")

    print("\n✓ Normalized dataset ready for fine-tuning!")
    print("\nNext step:")
    print("  Update finetune_qwen_audio.py to use normalized_clips instead of clean_clips")
    print("=" * 80)


if __name__ == "__main__":
    main()
