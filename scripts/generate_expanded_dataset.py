"""Generate expanded training dataset using ALL available audio data.

This script creates a much larger training dataset by:
1. Using ALL VoxConverse clips (SPEECH) - 216 clips
2. Using balanced ESC50 clips (NONSPEECH) - 216 clips
3. Applying data augmentation: 8 durations × 6 SNRs = 48 variations per clip
4. Total: 432 base clips × 48 = 20,736 balanced samples

This is ~6x more data than the original experimental_variants dataset.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Configuration
DURATIONS_MS = [20, 40, 60, 80, 100, 200, 500, 1000]
SNRS_DB = [-10, -5, 0, 5, 10, 20]
TARGET_SR = 16000
CONTAINER_DURATION_MS = 2000  # Pad/contain in 2 second clips


def load_and_resample(audio_path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio and resample to target sample rate."""
    audio, sr = sf.read(audio_path)

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak level."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def extract_segment(audio: np.ndarray, duration_ms: int, sr: int = TARGET_SR) -> np.ndarray:
    """Extract a segment of specified duration from audio."""
    duration_samples = int(duration_ms * sr / 1000)

    if len(audio) <= duration_samples:
        # Pad if too short
        return np.pad(audio, (0, duration_samples - len(audio)), mode='constant')

    # Random start position
    max_start = len(audio) - duration_samples
    start = random.randint(0, max_start)
    return audio[start:start + duration_samples]


def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise at specified SNR."""
    signal_power = np.mean(audio ** 2)

    if signal_power == 0:
        return audio

    # Calculate noise power for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate noise
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)

    return audio + noise.astype(np.float32)


def pad_to_container(audio: np.ndarray, container_ms: int = CONTAINER_DURATION_MS,
                     sr: int = TARGET_SR) -> np.ndarray:
    """Pad audio to container duration, centering the content."""
    container_samples = int(container_ms * sr / 1000)

    if len(audio) >= container_samples:
        return audio[:container_samples]

    # Center the audio in the container
    pad_total = container_samples - len(audio)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return np.pad(audio, (pad_left, pad_right), mode='constant')


def process_single_clip(args):
    """Process a single base clip with all augmentations."""
    clip_id, audio_path, ground_truth, dataset, output_dir, split = args

    results = []

    try:
        # Load audio
        audio = load_and_resample(audio_path)
        audio = peak_normalize(audio)

        # Generate all variations
        for duration_ms in DURATIONS_MS:
            for snr_db in SNRS_DB:
                # Extract segment
                segment = extract_segment(audio, duration_ms)

                # Add noise
                noisy = add_noise(segment, snr_db)

                # Pad to container
                final = pad_to_container(noisy)

                # Clip to prevent overflow
                final = np.clip(final, -1.0, 1.0)

                # Generate variant ID
                snr_str = f"+{snr_db}" if snr_db >= 0 else str(snr_db)
                variant_id = f"{clip_id}_dur{duration_ms}ms_snr{snr_str}dB"

                # Output path
                out_path = output_dir / "audio" / split / f"{variant_id}.wav"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Save audio
                sf.write(out_path, final, TARGET_SR)

                # Calculate RMS
                rms = float(np.sqrt(np.mean(final ** 2)))

                results.append({
                    'clip_id': clip_id,
                    'variant_id': variant_id,
                    'duration_ms': duration_ms,
                    'snr_db': snr_db,
                    'audio_path': f"processed/expanded_dataset/audio/{split}/{variant_id}.wav",
                    'ground_truth': ground_truth,
                    'dataset': dataset,
                    'sr': TARGET_SR,
                    'rms': rms,
                    'container_duration_ms': CONTAINER_DURATION_MS,
                    'normalization': 'peak',
                })

    except Exception as e:
        print(f"Error processing {clip_id}: {e}")
        return []

    return results


def discover_audio_files(data_dir: Path):
    """Discover all available audio files and categorize them."""
    speech_files = []
    nonspeech_files = []

    # VoxConverse (SPEECH)
    vox_dir = data_dir / "raw" / "voxconverse"
    if vox_dir.exists():
        for wav in vox_dir.rglob("*.wav"):
            clip_id = wav.stem
            speech_files.append({
                'clip_id': f"voxconverse_{clip_id}",
                'path': wav,
                'dataset': 'voxconverse'
            })

    # ESC-50 (NONSPEECH - environmental sounds)
    esc_dir = data_dir / "raw" / "esc50" / "audio"
    if not esc_dir.exists():
        esc_dir = data_dir / "raw" / "ESC-50" / "audio"

    if esc_dir.exists():
        for wav in esc_dir.rglob("*.wav"):
            clip_id = wav.stem
            nonspeech_files.append({
                'clip_id': f"esc50_{clip_id}",
                'path': wav,
                'dataset': 'esc50'
            })

    print(f"Found {len(speech_files)} SPEECH files (VoxConverse)")
    print(f"Found {len(nonspeech_files)} NONSPEECH files (ESC50)")

    return speech_files, nonspeech_files


def main():
    parser = argparse.ArgumentParser(description="Generate expanded training dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed/expanded_dataset",
                        help="Output directory")
    parser.add_argument("--max_clips_per_class", type=int, default=None,
                        help="Max clips per class (for testing)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--dev_ratio", type=float, default=0.2, help="Dev split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING EXPANDED TRAINING DATASET")
    print("=" * 80)
    print(f"Durations: {DURATIONS_MS} ms")
    print(f"SNRs: {SNRS_DB} dB")
    print(f"Variations per clip: {len(DURATIONS_MS) * len(SNRS_DB)}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Discover files
    speech_files, nonspeech_files = discover_audio_files(data_dir)

    if not speech_files or not nonspeech_files:
        print("ERROR: Could not find audio files!")
        return

    # Balance classes
    n_clips = min(len(speech_files), len(nonspeech_files))
    if args.max_clips_per_class:
        n_clips = min(n_clips, args.max_clips_per_class)

    print(f"\nUsing {n_clips} clips per class (balanced)")

    # Sample balanced clips
    random.shuffle(speech_files)
    random.shuffle(nonspeech_files)
    speech_files = speech_files[:n_clips]
    nonspeech_files = nonspeech_files[:n_clips]

    # Split into train/dev/test
    n_train = int(n_clips * args.train_ratio)
    n_dev = int(n_clips * args.dev_ratio)
    n_test = n_clips - n_train - n_dev

    print(f"Split: {n_train} train, {n_dev} dev, {n_test} test (per class)")

    # Create splits
    splits = {
        'train': {
            'speech': speech_files[:n_train],
            'nonspeech': nonspeech_files[:n_train]
        },
        'dev': {
            'speech': speech_files[n_train:n_train+n_dev],
            'nonspeech': nonspeech_files[n_train:n_train+n_dev]
        },
        'test': {
            'speech': speech_files[n_train+n_dev:],
            'nonspeech': nonspeech_files[n_train+n_dev:]
        }
    }

    # Process each split
    all_results = {'train': [], 'dev': [], 'test': []}

    for split_name, split_data in splits.items():
        print(f"\n{'=' * 40}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'=' * 40}")

        # Prepare tasks
        tasks = []

        for file_info in split_data['speech']:
            tasks.append((
                file_info['clip_id'],
                file_info['path'],
                'SPEECH',
                file_info['dataset'],
                output_dir,
                split_name
            ))

        for file_info in split_data['nonspeech']:
            tasks.append((
                file_info['clip_id'],
                file_info['path'],
                'NONSPEECH',
                file_info['dataset'],
                output_dir,
                split_name
            ))

        # Process in parallel
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_clip, task): task for task in tasks}

            for future in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Generating {split_name}"):
                result = future.result()
                results.extend(result)

        all_results[split_name] = results

        # Save manifest for this split
        df = pd.DataFrame(results)
        manifest_path = output_dir / f"{split_name}_metadata.csv"
        df.to_csv(manifest_path, index=False)

        print(f"  Saved {len(df)} samples to {manifest_path}")
        print(f"    SPEECH: {(df['ground_truth'] == 'SPEECH').sum()}")
        print(f"    NONSPEECH: {(df['ground_truth'] == 'NONSPEECH').sum()}")

    # Summary
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)

    total = 0
    for split_name, results in all_results.items():
        n = len(results)
        total += n
        print(f"  {split_name}: {n} samples")

    print(f"  TOTAL: {total} samples")
    print(f"\nOutput directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
