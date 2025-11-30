#!/usr/bin/env python3
"""Generate base 1000ms clips with Silero VAD verification for SPEECH samples.

This script:
1. Extracts random 1000ms segments from VoxConverse (SPEECH)
2. Verifies each segment has >80% speech using Silero VAD
3. Only keeps clips that pass verification
4. ESC-50 (NONSPEECH) clips don't need VAD verification

Output: data/processed/base_1000ms_verified/
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
from tqdm import tqdm
import random
import argparse
from scipy import signal

# Configuration
TARGET_SR = 16000
CLIP_DURATION_MS = 1000
SPEECH_THRESHOLD = 0.8  # Require >80% speech

# Load Silero VAD globally
print("Loading Silero VAD model...")
vad_model, vad_utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
(get_speech_timestamps, _, _, _, _) = vad_utils
print("Silero VAD loaded.")


def check_speech_ratio(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    """Check speech ratio using Silero VAD."""
    if sr != 16000:
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))

    wav = torch.tensor(audio, dtype=torch.float32)
    duration_samples = len(wav)

    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=16000)
    speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)

    return speech_samples / duration_samples if duration_samples > 0 else 0


def load_and_resample(audio_path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio and resample to target sample rate."""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak level."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return audio


def extract_verified_segment(audio: np.ndarray, duration_ms: int = CLIP_DURATION_MS,
                              sr: int = TARGET_SR, max_attempts: int = 50) -> tuple:
    """Extract a segment that passes Silero VAD verification.

    Returns:
        (segment, speech_ratio) or (None, 0) if no valid segment found
    """
    duration_samples = int(duration_ms * sr / 1000)

    if len(audio) < duration_samples:
        return None, 0

    max_start = len(audio) - duration_samples

    for _ in range(max_attempts):
        start = random.randint(0, max_start)
        segment = audio[start:start + duration_samples]

        speech_ratio = check_speech_ratio(segment, sr)

        if speech_ratio >= SPEECH_THRESHOLD:
            return segment, speech_ratio

    return None, 0


def extract_segment_nonspeech(audio: np.ndarray, duration_ms: int = CLIP_DURATION_MS,
                               sr: int = TARGET_SR) -> np.ndarray:
    """Extract a segment from NONSPEECH audio (no VAD verification needed)."""
    duration_samples = int(duration_ms * sr / 1000)

    if len(audio) <= duration_samples:
        return np.pad(audio, (0, duration_samples - len(audio)), mode='constant')

    max_start = len(audio) - duration_samples
    start = random.randint(0, max_start)
    return audio[start:start + duration_samples]


def discover_voxconverse_files(data_dir: Path) -> list:
    """Find all VoxConverse wav files."""
    vox_dir = data_dir / "raw" / "voxconverse"
    files = []

    for wav in vox_dir.rglob("*.wav"):
        if wav.stat().st_size > 10000:  # Skip tiny files
            files.append({
                'path': wav,
                'name': wav.stem,
            })

    print(f"Found {len(files)} VoxConverse files")
    return files


def discover_esc50_files(data_dir: Path) -> list:
    """Find all ESC-50 wav files."""
    esc_dir = data_dir / "raw" / "esc50" / "audio"
    if not esc_dir.exists():
        esc_dir = data_dir / "raw" / "ESC-50" / "audio"

    files = []
    for wav in esc_dir.rglob("*.wav"):
        files.append({
            'path': wav,
            'name': wav.stem,
        })

    print(f"Found {len(files)} ESC-50 files")
    return files


def main():
    parser = argparse.ArgumentParser(description="Generate verified base clips")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data/processed/base_1000ms_verified")
    parser.add_argument("--clips_per_source", type=int, default=3,
                        help="Number of clips to extract per source file")
    parser.add_argument("--target_speech_clips", type=int, default=200,
                        help="Target number of SPEECH clips total")
    parser.add_argument("--target_nonspeech_clips", type=int, default=200,
                        help="Target number of NONSPEECH clips total")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING VERIFIED BASE CLIPS")
    print("=" * 80)
    print(f"Speech threshold: {SPEECH_THRESHOLD*100:.0f}%")
    print(f"Target SPEECH clips: {args.target_speech_clips}")
    print(f"Target NONSPEECH clips: {args.target_nonspeech_clips}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Discover source files
    vox_files = discover_voxconverse_files(data_dir)
    esc_files = discover_esc50_files(data_dir)

    if not vox_files:
        print("ERROR: No VoxConverse files found!")
        return
    if not esc_files:
        print("ERROR: No ESC-50 files found!")
        return

    # Shuffle source files
    random.shuffle(vox_files)
    random.shuffle(esc_files)

    # ==========================================================================
    # Extract SPEECH clips with VAD verification
    # ==========================================================================
    print(f"\n{'='*40}")
    print("Extracting SPEECH clips with VAD verification...")
    print(f"{'='*40}")

    speech_clips = []
    pbar = tqdm(total=args.target_speech_clips, desc="SPEECH clips")

    for vox in vox_files:
        if len(speech_clips) >= args.target_speech_clips:
            break

        try:
            audio = load_and_resample(vox['path'])
            audio = peak_normalize(audio)

            # Try to extract multiple clips from this source
            clips_from_source = 0
            for i in range(args.clips_per_source):
                if len(speech_clips) >= args.target_speech_clips:
                    break

                segment, speech_ratio = extract_verified_segment(audio)

                if segment is not None:
                    clip_id = f"voxconverse_{vox['name']}_{clips_from_source:03d}_1000ms"
                    speech_clips.append({
                        'clip_id': clip_id,
                        'audio': segment,
                        'speech_ratio': speech_ratio,
                        'source': vox['name'],
                    })
                    clips_from_source += 1
                    pbar.update(1)

        except Exception as e:
            print(f"Error processing {vox['path']}: {e}")

    pbar.close()
    print(f"Extracted {len(speech_clips)} verified SPEECH clips")

    # ==========================================================================
    # Extract NONSPEECH clips (no VAD verification needed)
    # ==========================================================================
    print(f"\n{'='*40}")
    print("Extracting NONSPEECH clips...")
    print(f"{'='*40}")

    nonspeech_clips = []

    for esc in tqdm(esc_files[:args.target_nonspeech_clips], desc="NONSPEECH clips"):
        try:
            audio = load_and_resample(esc['path'])
            audio = peak_normalize(audio)
            segment = extract_segment_nonspeech(audio)

            clip_id = f"esc50_{esc['name']}_1000ms"
            nonspeech_clips.append({
                'clip_id': clip_id,
                'audio': segment,
                'source': esc['name'],
            })

        except Exception as e:
            print(f"Error processing {esc['path']}: {e}")

    print(f"Extracted {len(nonspeech_clips)} NONSPEECH clips")

    # ==========================================================================
    # Split into train/dev/test
    # ==========================================================================
    print(f"\n{'='*40}")
    print("Splitting into train/dev/test...")
    print(f"{'='*40}")

    random.shuffle(speech_clips)
    random.shuffle(nonspeech_clips)

    n_speech = len(speech_clips)
    n_nonspeech = len(nonspeech_clips)

    # Balance classes
    n_clips = min(n_speech, n_nonspeech)
    speech_clips = speech_clips[:n_clips]
    nonspeech_clips = nonspeech_clips[:n_clips]

    n_train = int(n_clips * args.train_ratio)
    n_dev = int(n_clips * args.dev_ratio)
    n_test = n_clips - n_train - n_dev

    print(f"Balanced: {n_clips} clips per class")
    print(f"Split: {n_train} train, {n_dev} dev, {n_test} test (per class)")

    splits = {
        'train': {
            'speech': speech_clips[:n_train],
            'nonspeech': nonspeech_clips[:n_train],
        },
        'dev': {
            'speech': speech_clips[n_train:n_train+n_dev],
            'nonspeech': nonspeech_clips[n_train:n_train+n_dev],
        },
        'test': {
            'speech': speech_clips[n_train+n_dev:],
            'nonspeech': nonspeech_clips[n_train+n_dev:],
        },
    }

    # ==========================================================================
    # Save clips and create CSVs
    # ==========================================================================
    for split_name, split_data in splits.items():
        print(f"\nSaving {split_name} split...")

        audio_dir = output_dir / "audio" / split_name
        audio_dir.mkdir(parents=True, exist_ok=True)

        rows = []

        # Save SPEECH clips
        for clip in split_data['speech']:
            audio_path = audio_dir / f"{clip['clip_id']}.wav"
            sf.write(audio_path, clip['audio'], TARGET_SR)

            rows.append({
                'clip_id': clip['clip_id'],
                'audio_path': f"processed/base_1000ms_verified/audio/{split_name}/{clip['clip_id']}.wav",
                'ground_truth': 'SPEECH',
                'dataset': 'voxconverse',
                'duration_ms': 1000,
                'sr': TARGET_SR,
                'speech_ratio': clip.get('speech_ratio', None),
            })

        # Save NONSPEECH clips
        for clip in split_data['nonspeech']:
            audio_path = audio_dir / f"{clip['clip_id']}.wav"
            sf.write(audio_path, clip['audio'], TARGET_SR)

            rows.append({
                'clip_id': clip['clip_id'],
                'audio_path': f"processed/base_1000ms_verified/audio/{split_name}/{clip['clip_id']}.wav",
                'ground_truth': 'NONSPEECH',
                'dataset': 'esc50',
                'duration_ms': 1000,
                'sr': TARGET_SR,
                'speech_ratio': None,
            })

        # Save CSV
        df = pd.DataFrame(rows)
        csv_path = output_dir / f"{split_name}_base.csv"
        df.to_csv(csv_path, index=False)

        n_speech = len(split_data['speech'])
        n_nonspeech = len(split_data['nonspeech'])
        print(f"  {split_name}: {len(rows)} clips ({n_speech} SPEECH, {n_nonspeech} NONSPEECH)")
        print(f"  Saved to: {csv_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"SPEECH clips verified with Silero VAD (>{SPEECH_THRESHOLD*100:.0f}% speech)")
    print("=" * 80)


if __name__ == "__main__":
    main()
