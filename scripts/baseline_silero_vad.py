#!/usr/bin/env python3
"""
Silero VAD Baseline Evaluation

Evaluates Silero Voice Activity Detector on the same test set used for
fine-tuned models. Provides modern deep learning baseline for comparison.

Silero VAD: State-of-the-art deep learning VAD
- Pre-trained on diverse multi-language data
- PyTorch-based (no compilation required)
- Frame-level predictions with high accuracy
- Supports various sample rates

Installation:
    # Silero VAD loads from torch.hub automatically
    # No additional installation required beyond torch

Usage:
    python scripts/baseline_silero_vad.py \
        --test_csv data/processed/grouped_split/test_metadata.csv \
        --output_csv results/baselines/silero_vad_predictions.csv \
        --threshold 0.5

Reference:
    https://github.com/snakers4/silero-vad
"""

import argparse
import pandas as pd
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def load_silero_vad():
    """
    Load Silero VAD model from torch.hub.

    Returns:
        model: Silero VAD model
        utils: Utility functions (get_speech_timestamps, etc.)
    """
    print("Loading Silero VAD from torch.hub...")
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )

    return model, utils


def classify_audio_silero(audio_path, model, threshold=0.5, sample_rate=16000):
    """
    Classify audio using Silero VAD.

    Args:
        audio_path: Path to audio file
        model: Silero VAD model
        threshold: Threshold for speech detection (default: 0.5)
        sample_rate: Target sample rate (8000 or 16000)

    Returns:
        dict with prediction, confidence (mean VAD probability), and frame stats
    """
    # Load audio
    audio, sr = sf.read(audio_path)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != sample_rate:
        import resampy
        audio = resampy.resample(audio, sr, sample_rate)

    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio).float()

    # Silero VAD requires fixed chunk size: 512 samples for 16kHz, 256 for 8kHz
    chunk_size = 512 if sample_rate == 16000 else 256

    # Process audio in chunks
    speech_probs = []
    for i in range(0, len(audio_tensor), chunk_size):
        chunk = audio_tensor[i:i+chunk_size]

        # Pad last chunk if needed
        if len(chunk) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))

        # Get speech probability for this chunk
        with torch.no_grad():
            prob = model(chunk, sample_rate)
            speech_probs.append(float(prob))

    # Convert to numpy array
    speech_probs = np.array(speech_probs)

    # Compute mean probability
    mean_prob = float(np.mean(speech_probs))

    # Count frames above threshold
    n_speech_frames = int(np.sum(speech_probs > threshold))
    n_total_frames = len(speech_probs)

    # Decision: classify as SPEECH if mean probability > threshold
    prediction = 'SPEECH' if mean_prob > threshold else 'NONSPEECH'

    return {
        'prediction': prediction,
        'confidence': mean_prob,
        'n_speech_frames': n_speech_frames,
        'n_total_frames': n_total_frames,
        'max_prob': float(np.max(speech_probs)),
        'min_prob': float(np.min(speech_probs))
    }


def evaluate_silero_vad(test_csv, threshold=0.5, sample_rate=16000, filter_duration=None, filter_snr=None):
    """
    Evaluate Silero VAD on test set.

    Args:
        test_csv: Path to test metadata CSV
        threshold: Threshold for speech detection
        sample_rate: Sample rate (8000 or 16000 Hz)
        filter_duration: Filter by duration in ms (optional)
        filter_snr: Filter by SNR in dB (optional)

    Returns:
        DataFrame with predictions and results
    """
    # Load test data
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")

    # Apply filters if specified
    if filter_duration is not None:
        if 'duration_ms' in test_df.columns:
            orig_len = len(test_df)
            test_df = test_df[test_df['duration_ms'] == filter_duration]
            print(f"  Filtered by duration={filter_duration}ms: {orig_len} -> {len(test_df)}")
        else:
            print(f"  WARNING: filter_duration specified but 'duration_ms' column not found")

    if filter_snr is not None:
        if 'snr_db' in test_df.columns:
            orig_len = len(test_df)
            test_df = test_df[test_df['snr_db'] == filter_snr]
            print(f"  Filtered by SNR={filter_snr}dB: {orig_len} -> {len(test_df)}")
        else:
            print(f"  WARNING: filter_snr specified but 'snr_db' column not found")

    print(f"Final test samples: {len(test_df)}")

    # Determine label column
    label_col = 'ground_truth' if 'ground_truth' in test_df.columns else 'label'

    # Load Silero VAD
    model, utils = load_silero_vad()
    model.eval()
    print(f"Silero VAD loaded (threshold={threshold})")

    # Evaluate
    results = []
    correct = 0
    total = 0

    print(f"\nEvaluating with sample_rate={sample_rate}, threshold={threshold}...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        audio_path = row['audio_path']
        ground_truth = row[label_col]

        try:
            result = classify_audio_silero(audio_path, model, threshold, sample_rate)

            is_correct = (result['prediction'] == ground_truth)
            correct += is_correct
            total += 1

            results.append({
                'clip_id': row['clip_id'],
                'audio_path': audio_path,
                'ground_truth': ground_truth,
                'prediction': result['prediction'],
                'correct': is_correct,
                'confidence': result['confidence'],
                'n_speech_frames': result['n_speech_frames'],
                'n_total_frames': result['n_total_frames'],
                'max_prob': result['max_prob'],
                'min_prob': result['min_prob']
            })
        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            results.append({
                'clip_id': row['clip_id'],
                'audio_path': audio_path,
                'ground_truth': ground_truth,
                'prediction': 'ERROR',
                'correct': False,
                'confidence': 0.0,
                'n_speech_frames': 0,
                'n_total_frames': 0,
                'max_prob': 0.0,
                'min_prob': 0.0
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Silero VAD baseline evaluation')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test metadata CSV')
    parser.add_argument('--output_csv', type=str, default='results/baselines/silero_vad_predictions.csv',
                       help='Output CSV for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Speech detection threshold (default: 0.5)')
    parser.add_argument('--sample_rate', type=int, default=16000, choices=[8000, 16000],
                       help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--filter_duration', type=int, default=None,
                       help='Filter by duration in ms (e.g., 1000). If None, no filtering.')
    parser.add_argument('--filter_snr', type=float, default=None,
                       help='Filter by SNR in dB (e.g., 20). If None, no filtering.')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate
    results_df = evaluate_silero_vad(
        args.test_csv,
        threshold=args.threshold,
        sample_rate=args.sample_rate,
        filter_duration=args.filter_duration,
        filter_snr=args.filter_snr
    )

    # Calculate metrics
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    # Overall accuracy
    n_correct = results_df['correct'].sum()
    n_total = len(results_df)
    accuracy = n_correct / n_total * 100

    print(f"\nOverall Accuracy: {n_correct}/{n_total} = {accuracy:.1f}%")

    # Per-class accuracy
    for label in ['SPEECH', 'NONSPEECH']:
        label_df = results_df[results_df['ground_truth'] == label]
        if len(label_df) > 0:
            label_correct = label_df['correct'].sum()
            label_total = len(label_df)
            label_acc = label_correct / label_total * 100
            print(f"  {label}: {label_correct}/{label_total} = {label_acc:.1f}%")

    # Confidence statistics
    print(f"\nConfidence (mean VAD probability) statistics:")
    print(f"  Overall:  {results_df['confidence'].mean():.3f} ± {results_df['confidence'].std():.3f}")
    correct_df = results_df[results_df['correct'] == True]
    wrong_df = results_df[results_df['correct'] == False]
    if len(correct_df) > 0:
        print(f"  Correct:  {correct_df['confidence'].mean():.3f}")
    if len(wrong_df) > 0:
        print(f"  Wrong:    {wrong_df['confidence'].mean():.3f}")

    # Probability range
    print(f"\nProbability range:")
    print(f"  Max: {results_df['max_prob'].max():.3f}")
    print(f"  Min: {results_df['min_prob'].min():.3f}")

    # Errors
    n_errors = len(wrong_df)
    if n_errors > 0:
        print(f"\n{'='*70}")
        print(f"ERRORS ({n_errors} total)")
        print(f"{'='*70}\n")

        for idx, row in wrong_df.iterrows():
            print(f"Clip: {row['clip_id']}")
            print(f"  Ground truth: {row['ground_truth']}")
            print(f"  Prediction:   {row['prediction']}")
            print(f"  Confidence:   {row['confidence']:.3f} (range: [{row['min_prob']:.3f}, {row['max_prob']:.3f}])")
            print(f"  Speech frames: {row['n_speech_frames']}/{row['n_total_frames']}")
            print()

    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to: {args.output_csv}")
    print()


if __name__ == "__main__":
    main()
