#!/usr/bin/env python3
"""
Prepare Base Clips with GroupShuffleSplit (Zero-Leakage)

Extracts 1000ms clips from VoxConverse (SPEECH) and ESC-50 (NONSPEECH)
and splits them into train/dev/test using GroupShuffleSplit to ensure
no speaker/sound leakage between splits.

Usage:
    python scripts/prepare_base_clips.py \
        --voxconverse_dir data/raw/voxconverse/dev \
        --esc50_dir data/raw/esc50/audio \
        --output_dir data/processed/base_1000ms \
        --duration 1000 \
        --train_size 64 \
        --dev_size 72 \
        --test_size 24 \
        --seed 42

Features:
- GroupShuffleSplit by speaker_id (VoxConverse) and clip_id (ESC-50)
- Zero-leakage guarantee: no group overlap between splits
- Balanced SPEECH/NONSPEECH in each split
- Metadata CSV with group_id, source_dataset, sr, duration_ms
- Peak normalization to preserve SNR

References:
- GroupShuffleSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def normalize_audio_peak(audio: np.ndarray, target_peak: float = 0.9, headroom_db: float = 3.0) -> np.ndarray:
    """
    Normalize audio by peak level, preserving relative energy differences (SNR).

    This is critical: peak normalization preserves the natural SNR characteristics
    of the audio, unlike RMS normalization which would equalize all clips.

    Args:
        audio: Input audio array
        target_peak: Target peak level (default: 0.9 to avoid clipping)
        headroom_db: Additional headroom in dB (default: 3.0)

    Returns:
        Peak-normalized audio
    """
    current_peak = np.abs(audio).max()

    if current_peak < 1e-6:
        return audio

    headroom_factor = 10 ** (-headroom_db / 20.0)
    gain = (target_peak * headroom_factor) / current_peak
    normalized = audio * gain
    normalized = np.clip(normalized, -1.0, 1.0)

    return normalized


def extract_clip_from_audio(audio_path: Path, duration_ms: int, target_sr: int = 16000,
                            normalize: bool = True) -> tuple[np.ndarray, int]:
    """
    Extract a random clip of specified duration from audio file.

    Args:
        audio_path: Path to audio file
        duration_ms: Desired clip duration in milliseconds
        target_sr: Target sampling rate (default: 16000 for Qwen2-Audio)
        normalize: Apply peak normalization

    Returns:
        (audio_clip, sampling_rate)
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Calculate samples needed
    samples_needed = int(duration_ms * target_sr / 1000)

    if len(audio) < samples_needed:
        # Pad if too short
        pad_length = samples_needed - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')

    # Extract random clip
    if len(audio) > samples_needed:
        max_start = len(audio) - samples_needed
        start_idx = np.random.randint(0, max_start + 1)
        audio = audio[start_idx:start_idx + samples_needed]

    # Normalize
    if normalize:
        audio = normalize_audio_peak(audio)

    return audio, target_sr


def extract_voxconverse_clips(voxconverse_dir: Path, duration_ms: int, n_clips: int,
                              target_sr: int, seed: int) -> pd.DataFrame:
    """
    Extract clips from VoxConverse dataset.

    Args:
        voxconverse_dir: Path to VoxConverse dev directory
        duration_ms: Clip duration in ms
        n_clips: Number of clips to extract
        target_sr: Target sampling rate
        seed: Random seed

    Returns:
        DataFrame with columns: clip_id, audio_path, ground_truth, dataset, speaker_id, duration_ms, sr
    """
    np.random.seed(seed)

    # Find all VoxConverse audio files
    audio_files = list(voxconverse_dir.glob("**/*.wav"))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {voxconverse_dir}")

    print(f"\nFound {len(audio_files)} VoxConverse audio files")

    # Sample files
    if len(audio_files) < n_clips:
        print(f"Warning: Only {len(audio_files)} files available, requested {n_clips}")
        n_clips = len(audio_files)

    selected_files = np.random.choice(audio_files, size=n_clips, replace=False)

    clips = []
    for idx, audio_file in enumerate(tqdm(selected_files, desc="Extracting VoxConverse clips")):
        # Extract speaker_id from filename (format: speakerID_utterance.wav)
        speaker_id = audio_file.stem.split('_')[0]

        clip_id = f"voxconverse_{speaker_id}_{idx:03d}_{duration_ms}ms"

        clips.append({
            'clip_id': clip_id,
            'audio_file': audio_file,
            'ground_truth': 'SPEECH',
            'dataset': 'voxconverse',
            'speaker_id': speaker_id,
            'duration_ms': duration_ms,
            'sr': target_sr
        })

    return pd.DataFrame(clips)


def extract_esc50_clips(esc50_dir: Path, duration_ms: int, n_clips: int,
                       target_sr: int, seed: int) -> pd.DataFrame:
    """
    Extract clips from ESC-50 dataset.

    Args:
        esc50_dir: Path to ESC-50 audio directory
        duration_ms: Clip duration in ms
        n_clips: Number of clips to extract
        target_sr: Target sampling rate
        seed: Random seed

    Returns:
        DataFrame with columns: clip_id, audio_path, ground_truth, dataset, clip_id, duration_ms, sr
    """
    np.random.seed(seed + 1)  # Different seed than VoxConverse

    # Find all ESC-50 audio files
    audio_files = list(esc50_dir.glob("*.wav"))

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {esc50_dir}")

    print(f"\nFound {len(audio_files)} ESC-50 audio files")

    # Sample files
    if len(audio_files) < n_clips:
        print(f"Warning: Only {len(audio_files)} files available, requested {n_clips}")
        n_clips = len(audio_files)

    selected_files = np.random.choice(audio_files, size=n_clips, replace=False)

    clips = []
    for idx, audio_file in enumerate(tqdm(selected_files, desc="Extracting ESC-50 clips")):
        # ESC-50 filename format: fold-class-clip.wav
        # Use the entire filename as clip_id to ensure uniqueness
        clip_identifier = audio_file.stem

        clip_id = f"esc50_{clip_identifier}_{idx:03d}_{duration_ms}ms"

        clips.append({
            'clip_id': clip_id,
            'audio_file': audio_file,
            'ground_truth': 'NONSPEECH',
            'dataset': 'esc50',
            'group_id': clip_identifier,  # Use full stem as group identifier
            'duration_ms': duration_ms,
            'sr': target_sr
        })

    return pd.DataFrame(clips)


def group_shuffle_split(df: pd.DataFrame, train_size: int, dev_size: int, test_size: int,
                       group_col: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data using GroupShuffleSplit to ensure no group leakage.

    Args:
        df: DataFrame with clips
        train_size: Number of samples for train
        dev_size: Number of samples for dev
        test_size: Number of samples for test
        group_col: Column name for grouping (e.g., 'speaker_id' or 'group_id')
        seed: Random seed

    Returns:
        (train_df, dev_df, test_df)
    """
    total_size = train_size + dev_size + test_size

    if len(df) < total_size:
        raise ValueError(f"Not enough samples: have {len(df)}, need {total_size}")

    # Get unique groups
    groups = df[group_col].values

    # First split: train vs (dev + test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()
    temp_groups = temp_df[group_col].values

    # Second split: dev vs test
    gss2 = GroupShuffleSplit(n_splits=1, train_size=dev_size, random_state=seed + 1)
    dev_idx, test_idx = next(gss2.split(temp_df, groups=temp_groups))

    dev_df = temp_df.iloc[dev_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # Verify no group leakage
    train_groups = set(train_df[group_col].unique())
    dev_groups = set(dev_df[group_col].unique())
    test_groups = set(test_df[group_col].unique())

    assert len(train_groups & dev_groups) == 0, "Group leakage between train and dev!"
    assert len(train_groups & test_groups) == 0, "Group leakage between train and test!"
    assert len(dev_groups & test_groups) == 0, "Group leakage between dev and test!"

    print(f"\n✓ Zero-leakage verified:")
    print(f"  Train groups: {len(train_groups)}")
    print(f"  Dev groups: {len(dev_groups)}")
    print(f"  Test groups: {len(test_groups)}")
    print(f"  No overlap confirmed")

    return train_df, dev_df, test_df


def save_clips_and_metadata(df: pd.DataFrame, output_dir: Path, split_name: str,
                           duration_ms: int, target_sr: int):
    """
    Save audio clips to disk and create metadata CSV.

    Args:
        df: DataFrame with clip information
        output_dir: Output directory
        split_name: 'train', 'dev', or 'test'
        duration_ms: Clip duration in ms
        target_sr: Target sampling rate
    """
    # Create audio directory
    audio_dir = output_dir / "audio" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Process each clip
    output_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Saving {split_name} clips"):
        # Extract clip
        audio, sr = extract_clip_from_audio(
            row['audio_file'],
            duration_ms,
            target_sr=target_sr,
            normalize=True
        )

        # Save audio
        output_path = audio_dir / f"{row['clip_id']}.wav"
        sf.write(output_path, audio, sr)

        # Compute RMS for metadata
        rms = np.sqrt(np.mean(audio ** 2))

        # Prepare metadata
        output_data.append({
            'clip_id': row['clip_id'],
            'audio_path': str(output_path.relative_to(output_dir.parent.parent)),  # Relative to project root
            'ground_truth': row['ground_truth'],
            'dataset': row['dataset'],
            'group_id': row.get('speaker_id', row.get('group_id')),
            'duration_ms': duration_ms,
            'sr': sr,
            'rms': rms,
            'normalization': 'peak'
        })

    # Save metadata CSV
    metadata_df = pd.DataFrame(output_data)
    metadata_path = output_dir / f"{split_name}_base.csv"
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\n✓ Saved {len(metadata_df)} {split_name} clips")
    print(f"  Audio: {audio_dir}")
    print(f"  Metadata: {metadata_path}")

    # Print class distribution
    class_counts = metadata_df['ground_truth'].value_counts()
    print(f"  Distribution: {dict(class_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare base clips with GroupShuffleSplit")
    parser.add_argument("--voxconverse_dir", type=Path, required=True,
                       help="Path to VoxConverse dev directory")
    parser.add_argument("--esc50_dir", type=Path, required=True,
                       help="Path to ESC-50 audio directory")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for base clips")
    parser.add_argument("--duration", type=int, default=1000,
                       help="Clip duration in milliseconds (default: 1000)")
    parser.add_argument("--train_size", type=int, default=64,
                       help="Number of train samples (default: 64)")
    parser.add_argument("--dev_size", type=int, default=72,
                       help="Number of dev samples (default: 72)")
    parser.add_argument("--test_size", type=int, default=24,
                       help="Number of test samples (default: 24)")
    parser.add_argument("--target_sr", type=int, default=16000,
                       help="Target sampling rate (default: 16000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--limit_per_split", type=int, default=None,
                       help="Limit samples per split for smoke testing")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPARE BASE CLIPS WITH GROUPSHUFFLESPLIT")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  VoxConverse: {args.voxconverse_dir}")
    print(f"  ESC-50: {args.esc50_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Duration: {args.duration}ms")
    print(f"  Target SR: {args.target_sr}Hz")
    print(f"  Splits: train={args.train_size}, dev={args.dev_size}, test={args.test_size}")
    print(f"  Seed: {args.seed}")

    # Smoke test mode
    if args.limit_per_split:
        print(f"\n⚠️  SMOKE TEST MODE: Limiting to {args.limit_per_split} samples per split")
        args.train_size = args.limit_per_split
        args.dev_size = args.limit_per_split
        args.test_size = args.limit_per_split

    # Calculate total samples needed per dataset
    total_needed = args.train_size + args.dev_size + args.test_size
    speech_needed = total_needed // 2
    nonspeech_needed = total_needed - speech_needed

    # Extract VoxConverse clips (SPEECH)
    print(f"\n{'=' * 80}")
    print(f"EXTRACTING VOXCONVERSE CLIPS (SPEECH)")
    print(f"{'=' * 80}")
    speech_df = extract_voxconverse_clips(
        args.voxconverse_dir,
        args.duration,
        speech_needed,
        args.target_sr,
        args.seed
    )

    # Extract ESC-50 clips (NONSPEECH)
    print(f"\n{'=' * 80}")
    print(f"EXTRACTING ESC-50 CLIPS (NONSPEECH)")
    print(f"{'=' * 80}")
    nonspeech_df = extract_esc50_clips(
        args.esc50_dir,
        args.duration,
        nonspeech_needed,
        args.target_sr,
        args.seed
    )

    # Split SPEECH with GroupShuffleSplit
    print(f"\n{'=' * 80}")
    print(f"SPLITTING SPEECH SAMPLES (GroupShuffleSplit by speaker)")
    print(f"{'=' * 80}")
    speech_train, speech_dev, speech_test = group_shuffle_split(
        speech_df,
        args.train_size // 2,
        args.dev_size // 2,
        args.test_size // 2,
        group_col='speaker_id',
        seed=args.seed
    )

    # Split NONSPEECH with GroupShuffleSplit
    print(f"\n{'=' * 80}")
    print(f"SPLITTING NONSPEECH SAMPLES (GroupShuffleSplit by clip)")
    print(f"{'=' * 80}")
    nonspeech_train, nonspeech_dev, nonspeech_test = group_shuffle_split(
        nonspeech_df,
        args.train_size // 2,
        args.dev_size // 2,
        args.test_size // 2,
        group_col='group_id',
        seed=args.seed
    )

    # Combine SPEECH and NONSPEECH for each split
    train_df = pd.concat([speech_train, nonspeech_train], ignore_index=True).sample(frac=1, random_state=args.seed)
    dev_df = pd.concat([speech_dev, nonspeech_dev], ignore_index=True).sample(frac=1, random_state=args.seed)
    test_df = pd.concat([speech_test, nonspeech_test], ignore_index=True).sample(frac=1, random_state=args.seed)

    # Save clips and metadata
    print(f"\n{'=' * 80}")
    print(f"SAVING CLIPS AND METADATA")
    print(f"{'=' * 80}")

    save_clips_and_metadata(train_df, args.output_dir, 'train', args.duration, args.target_sr)
    save_clips_and_metadata(dev_df, args.output_dir, 'dev', args.duration, args.target_sr)
    save_clips_and_metadata(test_df, args.output_dir, 'test', args.duration, args.target_sr)

    print(f"\n{'=' * 80}")
    print(f"✓ BASE CLIPS PREPARATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  train_base.csv: {len(train_df)} samples")
    print(f"  dev_base.csv: {len(dev_df)} samples")
    print(f"  test_base.csv: {len(test_df)} samples")
    print(f"\n✓ Zero-leakage verified (no group overlap between splits)")


if __name__ == "__main__":
    main()
