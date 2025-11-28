"""Generate expanded training dataset with ALL 4 psychoacoustic conditions.

This script creates a comprehensive training dataset by applying:
1. Duration variations (8 levels: 20-1000ms) - clean, no filter, no reverb
2. SNR variations (6 levels: -10 to +20dB) - 1000ms, no filter, no reverb
3. Band filter variations (3 types: telephony, lp3400, hp300) - 1000ms, clean, no reverb
4. Reverb/RIR variations (3 T60 levels: 0.2s, 0.6s, 1.1s) - 1000ms, clean, no filter

Each condition varies ONE dimension while keeping others at "normal" (baseline).
Normal = 1000ms, clean (no added noise), no filter, no reverb.

Total: 8 + 6 + 3 + 3 = 20 conditions per base clip.
Note: duration=1000ms is the "normal" for duration, so effectively 19 unique + 1 shared.
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
from scipy import signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qsm.audio.filters import apply_bandpass, apply_lowpass, apply_highpass

# Configuration
DURATIONS_MS = [20, 40, 60, 80, 100, 200, 500, 1000]
SNRS_DB = [-10, -5, 0, 5, 10, 20]
BAND_FILTERS = {
    "telephony": {"fn": apply_bandpass, "kwargs": {"lowcut": 300, "highcut": 3400}},
    "lp3400": {"fn": apply_lowpass, "kwargs": {"highcut": 3400}},
    "hp300": {"fn": apply_highpass, "kwargs": {"lowcut": 300}},
}
T60_CONFIGS = [
    {"name": "T60_0.2", "t60": 0.2, "bin": "T60_0.0-0.4"},
    {"name": "T60_0.6", "t60": 0.6, "bin": "T60_0.4-0.8"},
    {"name": "T60_1.1", "t60": 1.1, "bin": "T60_0.8-1.5"},
]
TARGET_SR = 16000
CONTAINER_DURATION_MS = 2000  # Pad/contain in 2 second clips


def create_synthetic_rir(t60: float, sr: int = TARGET_SR, duration: float = 1.0) -> np.ndarray:
    """Create synthetic RIR with specified T60 (reverberation time).

    Uses exponential decay envelope with white noise.

    Args:
        t60: Reverberation time in seconds (time for 60dB decay)
        sr: Sample rate
        duration: RIR duration in seconds

    Returns:
        Synthetic RIR array
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    if t60 > 0.01:
        # Exponential decay: e^(-decay_rate * t) where decay_rate = ln(1000)/T60
        decay_rate = 3 * np.log(10) / t60
        envelope = np.exp(-decay_rate * t)
    else:
        # Near-zero T60: just direct sound (impulse)
        envelope = np.zeros(n_samples)
        envelope[0] = 1.0

    # Reproducible random noise based on T60
    rng = np.random.RandomState(int(t60 * 1000))
    noise = rng.randn(n_samples)

    rir = noise * envelope
    rir = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir.astype(np.float32)


def apply_rir(audio: np.ndarray, rir: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Apply RIR to audio via FFT convolution.

    Args:
        audio: Input audio
        rir: Room impulse response
        normalize: Whether to normalize output to preserve RMS

    Returns:
        Reverberant audio (same length as input)
    """
    reverb = signal.fftconvolve(audio, rir, mode="full")[:len(audio)]

    if normalize:
        rms_orig = np.sqrt(np.mean(audio**2))
        rms_reverb = np.sqrt(np.mean(reverb**2))
        if rms_reverb > 1e-8:
            reverb = reverb * (rms_orig / rms_reverb)

    return reverb.astype(np.float32)


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

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)

    return (audio + noise).astype(np.float32)


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
    """Process a single base clip with all 4 condition types (20 independent conditions).

    Each condition varies ONE dimension while keeping others at "normal":
    - Normal = 1000ms, clean (no added noise), no filter, no reverb
    """
    clip_id, audio_path, ground_truth, dataset, output_dir, split = args

    results = []

    try:
        # Load and normalize audio
        audio = load_and_resample(audio_path)
        audio = peak_normalize(audio)

        # =================================================================
        # 1. DURATION variants (8 levels) - clean, no filter, no reverb
        # =================================================================
        for duration_ms in DURATIONS_MS:
            segment = extract_segment(audio, duration_ms)
            # NO noise added - clean audio
            final = pad_to_container(segment)
            final = np.clip(final, -1.0, 1.0)

            variant_id = f"{clip_id}_dur{duration_ms}ms"

            out_path = output_dir / "audio" / split / f"{variant_id}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, final, TARGET_SR)

            # duration=1000ms is the "normal" baseline
            is_normal = (duration_ms == 1000)

            results.append({
                'clip_id': clip_id,
                'variant_id': variant_id,
                'condition_type': 'duration',
                'duration_ms': duration_ms,
                'snr_db': None,  # clean
                'band_filter': None,  # none
                't60': None,  # none
                't60_bin': None,
                'is_normal': is_normal,
                'audio_path': f"audio/{split}/{variant_id}.wav",
                'ground_truth': ground_truth,
                'dataset': dataset,
            })

        # =================================================================
        # 2. SNR variants (6 levels) - 1000ms, no filter, no reverb
        # =================================================================
        for snr_db in SNRS_DB:
            segment = extract_segment(audio, 1000)  # Fixed at 1000ms
            noisy = add_noise(segment, snr_db)
            final = pad_to_container(noisy)
            final = np.clip(final, -1.0, 1.0)

            snr_str = f"+{snr_db}" if snr_db >= 0 else str(snr_db)
            variant_id = f"{clip_id}_snr{snr_str}dB"

            out_path = output_dir / "audio" / split / f"{variant_id}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, final, TARGET_SR)

            results.append({
                'clip_id': clip_id,
                'variant_id': variant_id,
                'condition_type': 'snr',
                'duration_ms': 1000,  # fixed
                'snr_db': snr_db,
                'band_filter': None,  # none
                't60': None,  # none
                't60_bin': None,
                'is_normal': False,
                'audio_path': f"audio/{split}/{variant_id}.wav",
                'ground_truth': ground_truth,
                'dataset': dataset,
            })

        # =================================================================
        # 3. BAND FILTER variants (3 types) - 1000ms, clean, no reverb
        # =================================================================
        for filter_name, filter_cfg in BAND_FILTERS.items():
            segment = extract_segment(audio, 1000)  # Fixed at 1000ms
            # NO noise added - clean audio
            filtered = filter_cfg["fn"](segment, TARGET_SR, **filter_cfg["kwargs"])
            final = pad_to_container(filtered)
            final = np.clip(final, -1.0, 1.0)

            variant_id = f"{clip_id}_filter{filter_name}"

            out_path = output_dir / "audio" / split / f"{variant_id}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, final, TARGET_SR)

            results.append({
                'clip_id': clip_id,
                'variant_id': variant_id,
                'condition_type': 'filter',
                'duration_ms': 1000,  # fixed
                'snr_db': None,  # clean
                'band_filter': filter_name,
                't60': None,  # none
                't60_bin': None,
                'is_normal': False,
                'audio_path': f"audio/{split}/{variant_id}.wav",
                'ground_truth': ground_truth,
                'dataset': dataset,
            })

        # =================================================================
        # 4. REVERB/RIR variants (3 levels) - 1000ms, clean, no filter
        # =================================================================
        for t60_cfg in T60_CONFIGS:
            segment = extract_segment(audio, 1000)  # Fixed at 1000ms
            # NO noise added - clean audio
            rir = create_synthetic_rir(t60_cfg["t60"], TARGET_SR)
            reverb = apply_rir(segment, rir)
            final = pad_to_container(reverb)
            final = np.clip(final, -1.0, 1.0)

            variant_id = f"{clip_id}_reverb{t60_cfg['name']}"

            out_path = output_dir / "audio" / split / f"{variant_id}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, final, TARGET_SR)

            results.append({
                'clip_id': clip_id,
                'variant_id': variant_id,
                'condition_type': 'reverb',
                'duration_ms': 1000,  # fixed
                'snr_db': None,  # clean
                'band_filter': None,  # none
                't60': t60_cfg["t60"],
                't60_bin': t60_cfg["bin"],
                'is_normal': False,
                'audio_path': f"audio/{split}/{variant_id}.wav",
                'ground_truth': ground_truth,
                'dataset': dataset,
            })

    except Exception as e:
        print(f"Error processing {clip_id}: {e}")
        return []

    return results


def load_base_clips_from_csv(data_dir: Path):
    """Load base clips from preprocessed CSV manifests.

    Uses the existing base_1000ms structure with train/dev/test splits.
    """
    base_dir = data_dir / "processed" / "base_1000ms"

    splits = {'train': [], 'dev': [], 'test': []}

    for split_name in ['train', 'dev', 'test']:
        csv_path = base_dir / f"{split_name}_base.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} clips from {csv_path}")

        for _, row in df.iterrows():
            # Build full path
            audio_path = data_dir / row['audio_path']
            if not audio_path.exists():
                # Try relative to processed dir
                audio_path = base_dir / row['audio_path'].replace('data/processed/base_1000ms/', '')

            splits[split_name].append({
                'clip_id': row['clip_id'],
                'path': audio_path,
                'ground_truth': row['ground_truth'],
                'dataset': row.get('dataset', 'unknown'),
            })

    return splits


def discover_audio_files(data_dir: Path):
    """Discover all available audio files and categorize them.

    First tries to use preprocessed base_1000ms clips, then falls back to raw files.
    """
    # Try preprocessed base clips first
    base_dir = data_dir / "processed" / "base_1000ms"
    if (base_dir / "train_base.csv").exists():
        print("Using preprocessed base_1000ms clips")
        return None, None  # Signal to use load_base_clips_from_csv instead

    # Fall back to discovering raw files
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
    parser = argparse.ArgumentParser(description="Generate expanded dataset with 4 conditions")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="data/processed/expanded_4conditions",
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
    print("GENERATING EXPANDED DATASET WITH 4 CONDITIONS (20 INDEPENDENT)")
    print("=" * 80)
    print(f"1. Duration: {DURATIONS_MS} ms ({len(DURATIONS_MS)} levels) - clean, no filter, no reverb")
    print(f"2. SNR: {SNRS_DB} dB ({len(SNRS_DB)} levels) - 1000ms, no filter, no reverb")
    print(f"3. Band Filter: {list(BAND_FILTERS.keys())} ({len(BAND_FILTERS)} types) - 1000ms, clean, no reverb")
    print(f"4. Reverb/T60: {[c['t60'] for c in T60_CONFIGS]} s ({len(T60_CONFIGS)} levels) - 1000ms, clean, no filter")
    n_dur = len(DURATIONS_MS)
    n_snr = len(SNRS_DB)
    n_band = len(BAND_FILTERS)
    n_rir = len(T60_CONFIGS)
    n_total = n_dur + n_snr + n_band + n_rir
    print(f"Total conditions per clip: {n_dur} + {n_snr} + {n_band} + {n_rir} = {n_total}")
    print(f"Note: duration=1000ms is the 'Normal' baseline (is_normal=True)")
    print(f"Output: {output_dir}")
    print("=" * 80)

    # Check for preprocessed base clips
    speech_files, nonspeech_files = discover_audio_files(data_dir)

    if speech_files is None:
        # Use preprocessed base_1000ms clips with predefined splits
        print("\nUsing preprocessed base_1000ms clips with existing splits")
        base_splits = load_base_clips_from_csv(data_dir)

        if not any(base_splits.values()):
            print("ERROR: Could not load base clips!")
            return

        # Convert to the format expected by processing
        splits = {}
        for split_name, clips in base_splits.items():
            splits[split_name] = []
            for clip_info in clips:
                splits[split_name].append({
                    'clip_id': clip_info['clip_id'],
                    'path': clip_info['path'],
                    'ground_truth': clip_info['ground_truth'],
                    'dataset': clip_info['dataset'],
                })

        for split_name, clip_list in splits.items():
            n_speech = sum(1 for c in clip_list if c['ground_truth'] == 'SPEECH')
            n_nonspeech = len(clip_list) - n_speech
            print(f"  {split_name}: {len(clip_list)} clips ({n_speech} speech, {n_nonspeech} nonspeech)")

    else:
        # Fall back to raw file discovery
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

        # Create splits (from raw files, need to add ground_truth)
        splits = {'train': [], 'dev': [], 'test': []}
        for f in speech_files[:n_train]:
            f['ground_truth'] = 'SPEECH'
            splits['train'].append(f)
        for f in nonspeech_files[:n_train]:
            f['ground_truth'] = 'NONSPEECH'
            splits['train'].append(f)
        for f in speech_files[n_train:n_train+n_dev]:
            f['ground_truth'] = 'SPEECH'
            splits['dev'].append(f)
        for f in nonspeech_files[n_train:n_train+n_dev]:
            f['ground_truth'] = 'NONSPEECH'
            splits['dev'].append(f)
        for f in speech_files[n_train+n_dev:]:
            f['ground_truth'] = 'SPEECH'
            splits['test'].append(f)
        for f in nonspeech_files[n_train+n_dev:]:
            f['ground_truth'] = 'NONSPEECH'
            splits['test'].append(f)

    # Process each split
    all_results = {'train': [], 'dev': [], 'test': []}

    for split_name, clip_list in splits.items():
        print(f"\n{'=' * 40}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'=' * 40}")

        # Prepare tasks
        tasks = []

        for file_info in clip_list:
            tasks.append((
                file_info['clip_id'],
                file_info['path'],
                file_info['ground_truth'],
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
        print(f"    By condition_type:")
        for ct in ['duration', 'snr', 'filter', 'reverb']:
            if ct in df['condition_type'].values:
                print(f"      {ct}: {(df['condition_type'] == ct).sum()}")

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
