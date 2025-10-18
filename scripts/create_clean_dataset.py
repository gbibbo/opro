"""Extract clean clips (center portion) from SNR/duration crossed files.

This removes the noise padding and keeps only the actual speech/nonspeech content.
For files that are 2000ms with 1000ms of content in the center:
- Original: [500ms noise][1000ms content][500ms noise]
- Cleaned: [1000ms content]
"""

import sys
from pathlib import Path
import pandas as pd
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

def extract_center_portion(audio_path: Path, target_duration_ms: int, output_path: Path):
    """Extract the center portion of an audio file.

    Args:
        audio_path: Path to input audio file
        target_duration_ms: Duration of the center portion to extract (in ms)
        output_path: Path to save the extracted audio
    """
    # Load audio
    audio, sr = sf.read(audio_path)

    # Calculate samples
    total_samples = len(audio)
    target_samples = int(target_duration_ms * sr / 1000)

    if total_samples <= target_samples:
        # Already shorter than target, just copy
        sf.write(output_path, audio, sr)
        return audio, sr

    # Extract center
    start_sample = (total_samples - target_samples) // 2
    end_sample = start_sample + target_samples

    audio_center = audio[start_sample:end_sample]

    # Save
    sf.write(output_path, audio_center, sr)

    return audio_center, sr


def main():
    print("=" * 80)
    print("CLEAN DATASET CREATION")
    print("=" * 80)

    # Paths
    input_dir = project_root / "data" / "processed" / "snr_duration_crossed"
    output_dir = project_root / "data" / "processed" / "clean_clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = input_dir / "metadata.csv"

    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load metadata
    print(f"\nLoading metadata from {metadata_path.name}...")
    df = pd.read_csv(metadata_path)
    print(f"  Total samples: {len(df)}")

    # Filter: Use more relaxed criteria to get more training data
    # Lower SNR threshold and minimum duration
    df_filtered = df[
        (df['snr_db'] >= 0) &  # Accept even low SNR (we'll normalize anyway)
        (df['duration_ms'] >= 200)  # Accept shorter clips
    ].copy()

    print(f"\nFiltered to {len(df_filtered)} samples (SNR >= 0dB, duration >= 200ms)")

    # Check class distribution
    speech_count = (df_filtered['ground_truth'] == 'SPEECH').sum()
    nonspeech_count = (df_filtered['ground_truth'].str.upper().isin(['NONSPEECH', 'NON-SPEECH'])).sum()

    print(f"\nClass distribution:")
    print(f"  SPEECH:    {speech_count}")
    print(f"  NONSPEECH: {nonspeech_count}")

    # Balance classes (take min of both)
    min_count = min(speech_count, nonspeech_count)
    print(f"\nBalancing to {min_count} samples per class...")

    df_speech = df_filtered[df_filtered['ground_truth'] == 'SPEECH'].sample(n=min_count, random_state=42)
    df_nonspeech = df_filtered[df_filtered['ground_truth'].str.upper().isin(['NONSPEECH', 'NON-SPEECH'])].sample(n=min_count, random_state=42)

    df_balanced = pd.concat([df_speech, df_nonspeech])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    print(f"  Balanced dataset: {len(df_balanced)} samples")

    # Process each file
    print(f"\nExtracting clean clips...")

    new_records = []

    for idx, row in tqdm(df_balanced.iterrows(), total=len(df_balanced)):
        input_path = project_root / row['audio_path']

        if not input_path.exists():
            print(f"\n[WARNING] File not found: {input_path.name}")
            continue

        # Create output filename
        output_filename = f"{row['clip_id']}_clean_{row['duration_ms']}ms_snr{row['snr_db']:+.0f}dB.wav"
        output_path = output_dir / output_filename

        try:
            # Extract center portion
            audio_clean, sr = extract_center_portion(
                input_path,
                target_duration_ms=int(row['duration_ms']),
                output_path=output_path
            )

            # Calculate RMS of cleaned audio
            rms_clean = np.sqrt(np.mean(audio_clean ** 2))

            # Create new record
            new_record = {
                'clip_id': row['clip_id'],
                'original_variant': row['variant_name'],
                'duration_ms': row['duration_ms'],
                'snr_db': row['snr_db'],
                'audio_path': f"data/processed/clean_clips/{output_filename}",
                'ground_truth': row['ground_truth'].upper(),
                'dataset': row['dataset'],
                'rms': rms_clean,
            }
            new_records.append(new_record)

        except Exception as e:
            print(f"\n[ERROR] Failed to process {input_path.name}: {e}")
            continue

    # Create new metadata
    df_clean = pd.DataFrame(new_records)

    print(f"\n✓ Successfully extracted {len(df_clean)} clean clips")

    # Split into train/test (80/20)
    print(f"\nSplitting into train/test (80/20)...")
    df_train = df_clean.sample(frac=0.8, random_state=42)
    df_test = df_clean.drop(df_train.index)

    print(f"  Train: {len(df_train)} samples")
    print(f"  Test:  {len(df_test)} samples")

    # Save metadata
    train_csv = output_dir / "train_metadata.csv"
    test_csv = output_dir / "test_metadata.csv"
    full_csv = output_dir / "clean_metadata.csv"

    df_clean.to_csv(full_csv, index=False)
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
    print(f"Total clean clips:     {len(df_clean)}")
    print(f"Train samples:         {len(df_train)}")
    print(f"Test samples:          {len(df_test)}")
    print(f"\nTrain class balance:")
    print(f"  SPEECH:    {(df_train['ground_truth'] == 'SPEECH').sum()}")
    print(f"  NONSPEECH: {(df_train['ground_truth'] == 'NONSPEECH').sum()}")
    print(f"\nTest class balance:")
    print(f"  SPEECH:    {(df_test['ground_truth'] == 'SPEECH').sum()}")
    print(f"  NONSPEECH: {(df_test['ground_truth'] == 'NONSPEECH').sum()}")
    print("\n✓ Dataset ready for fine-tuning!")
    print("=" * 80)


if __name__ == "__main__":
    main()
