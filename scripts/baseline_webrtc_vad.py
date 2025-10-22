#!/usr/bin/env python3
"""
WebRTC VAD Baseline Evaluation

Evaluates WebRTC Voice Activity Detector on the same test set used for
fine-tuned models. Provides classical baseline for comparison.

WebRTC VAD: Industry-standard VAD optimized for telephony speech.
- Three aggressiveness modes: 0 (quality), 1 (low bitrate), 2 (aggressive)
- Frame-based processing (10, 20, or 30 ms frames)
- Optimized for clean speech detection

Installation:
    pip install webrtcvad

Usage:
    python scripts/baseline_webrtc_vad.py \
        --test_csv data/processed/grouped_split/test_metadata.csv \
        --output_csv results/baselines/webrtc_vad_predictions.csv \
        --aggressiveness 1
"""

import argparse
import pandas as pd
import numpy as np
import webrtcvad
import soundfile as sf
from pathlib import Path
from tqdm import tqdm


def frame_generator(audio, sample_rate, frame_duration_ms=30):
    """
    Generate audio frames for VAD processing.

    Args:
        audio: Audio samples (numpy array)
        sample_rate: Sample rate in Hz
        frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)

    Yields:
        Audio frames as bytes
    """
    # Calculate frame size
    n_samples = int(sample_rate * (frame_duration_ms / 1000.0))

    # Pad audio to multiple of frame size
    remainder = len(audio) % n_samples
    if remainder != 0:
        audio = np.pad(audio, (0, n_samples - remainder), mode='constant')

    # Convert to int16 (WebRTC VAD requirement)
    if audio.dtype != np.int16:
        audio = (audio * 32767).astype(np.int16)

    # Generate frames
    for i in range(0, len(audio), n_samples):
        frame = audio[i:i+n_samples]
        yield frame.tobytes()


def classify_audio_webrtc(audio_path, vad, sample_rate=16000, frame_duration_ms=30):
    """
    Classify audio using WebRTC VAD.

    Args:
        audio_path: Path to audio file
        vad: WebRTC VAD instance
        sample_rate: Target sample rate (8000, 16000, 32000, or 48000)
        frame_duration_ms: Frame duration (10, 20, or 30 ms)

    Returns:
        dict with prediction, confidence (proportion of speech frames), and frame counts
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

    # Process frames
    n_speech_frames = 0
    n_total_frames = 0

    for frame in frame_generator(audio, sample_rate, frame_duration_ms):
        is_speech = vad.is_speech(frame, sample_rate)
        if is_speech:
            n_speech_frames += 1
        n_total_frames += 1

    # Compute proportion of speech frames
    speech_proportion = n_speech_frames / n_total_frames if n_total_frames > 0 else 0.0

    # Decision: classify as SPEECH if > 50% of frames contain speech
    prediction = 'SPEECH' if speech_proportion > 0.5 else 'NONSPEECH'

    return {
        'prediction': prediction,
        'confidence': speech_proportion,
        'n_speech_frames': n_speech_frames,
        'n_total_frames': n_total_frames
    }


def evaluate_webrtc_vad(test_csv, aggressiveness=1, sample_rate=16000, frame_duration_ms=30):
    """
    Evaluate WebRTC VAD on test set.

    Args:
        test_csv: Path to test metadata CSV
        aggressiveness: VAD aggressiveness (0=quality, 1=low_bitrate, 2=aggressive)
        sample_rate: Sample rate (8000, 16000, 32000, or 48000 Hz)
        frame_duration_ms: Frame duration (10, 20, or 30 ms)

    Returns:
        DataFrame with predictions and results
    """
    # Load test data
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")

    # Determine label column
    label_col = 'ground_truth' if 'ground_truth' in test_df.columns else 'label'

    # Initialize WebRTC VAD
    vad = webrtcvad.Vad(aggressiveness)
    print(f"WebRTC VAD initialized (aggressiveness={aggressiveness})")

    # Evaluate
    results = []
    correct = 0
    total = 0

    print(f"\nEvaluating with sample_rate={sample_rate}, frame_duration={frame_duration_ms}ms...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        audio_path = row['audio_path']
        ground_truth = row[label_col]

        try:
            result = classify_audio_webrtc(audio_path, vad, sample_rate, frame_duration_ms)

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
                'n_total_frames': result['n_total_frames']
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
                'n_total_frames': 0
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='WebRTC VAD baseline evaluation')
    parser.add_argument('--test_csv', type=str, required=True,
                       help='Path to test metadata CSV')
    parser.add_argument('--output_csv', type=str, default='results/baselines/webrtc_vad_predictions.csv',
                       help='Output CSV for predictions')
    parser.add_argument('--aggressiveness', type=int, default=1, choices=[0, 1, 2],
                       help='VAD aggressiveness: 0=quality, 1=low_bitrate, 2=aggressive (default: 1)')
    parser.add_argument('--sample_rate', type=int, default=16000, choices=[8000, 16000, 32000, 48000],
                       help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--frame_duration_ms', type=int, default=30, choices=[10, 20, 30],
                       help='Frame duration in ms (default: 30)')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate
    results_df = evaluate_webrtc_vad(
        args.test_csv,
        aggressiveness=args.aggressiveness,
        sample_rate=args.sample_rate,
        frame_duration_ms=args.frame_duration_ms
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
    print(f"\nConfidence (speech proportion) statistics:")
    print(f"  Overall:  {results_df['confidence'].mean():.3f} ± {results_df['confidence'].std():.3f}")
    correct_df = results_df[results_df['correct'] == True]
    wrong_df = results_df[results_df['correct'] == False]
    if len(correct_df) > 0:
        print(f"  Correct:  {correct_df['confidence'].mean():.3f}")
    if len(wrong_df) > 0:
        print(f"  Wrong:    {wrong_df['confidence'].mean():.3f}")

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
            print(f"  Confidence:   {row['confidence']:.3f} ({row['n_speech_frames']}/{row['n_total_frames']} frames)")
            print()

    # Save results
    results_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to: {args.output_csv}")
    print()


if __name__ == "__main__":
    main()
