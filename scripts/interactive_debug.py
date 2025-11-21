#!/usr/bin/env python3
"""
Interactive debugging script for speech classification.
Run LOCALLY (not on cluster) to listen to audio and see predictions.

Usage:
    python scripts/interactive_debug.py --n_samples 10 --snr 20 --duration 1000
"""

import argparse
import pandas as pd
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
import os
import sys

# Try to import audio playback library
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("WARNING: sounddevice not installed. Install with: pip install sounddevice")
    print("         Audio will be saved to files instead of playing directly.\n")


def play_audio(audio, sr):
    """Play audio or save to temp file."""
    if HAS_SOUNDDEVICE:
        print("  [Playing audio... press Ctrl+C to skip]")
        try:
            sd.play(audio, sr)
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("  [Stopped]")
    else:
        temp_path = "temp_debug_audio.wav"
        sf.write(temp_path, audio, sr)
        print(f"  [Audio saved to: {temp_path}]")
        print(f"  [Open it manually to listen]")


def main():
    parser = argparse.ArgumentParser(description="Interactive audio debugging")
    parser.add_argument("--test_csv", type=str, default="data/processed/experimental_variants/dev_metadata.csv")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to test")
    parser.add_argument("--snr", type=float, default=None, help="Filter by SNR (e.g., 20)")
    parser.add_argument("--duration", type=int, default=None, help="Filter by duration in ms (e.g., 1000)")
    parser.add_argument("--label", type=str, default=None, choices=["SPEECH", "NONSPEECH"], help="Filter by label")
    parser.add_argument("--no-model", action="store_true", help="Skip model loading (just play audio)")
    parser.add_argument("--prompt_file", type=str, default="prompts/prompt_base.txt", help="Prompt file to use")
    args = parser.parse_args()

    # Load and filter data
    print(f"Loading data from {args.test_csv}...")
    df = pd.read_csv(args.test_csv)
    label_col = 'ground_truth' if 'ground_truth' in df.columns else 'label'

    print(f"Total samples: {len(df)}")

    if args.snr is not None:
        df = df[df['snr_db'] == args.snr]
        print(f"Filtered by SNR={args.snr}dB: {len(df)} samples")

    if args.duration is not None:
        df = df[df['duration_ms'] == args.duration]
        print(f"Filtered by duration={args.duration}ms: {len(df)} samples")

    if args.label is not None:
        df = df[df[label_col] == args.label]
        print(f"Filtered by label={args.label}: {len(df)} samples")

    # Sample
    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=42)

    print(f"\nWill test {len(df)} samples\n")

    # Load prompt
    prompt_text = None
    if os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            prompt_text = f.read().strip()
        print(f"Prompt ({args.prompt_file}):")
        print("-" * 50)
        print(prompt_text)
        print("-" * 50)

    # Load model if needed
    model = None
    processor = None
    ids_A = None
    ids_B = None

    if not args.no_model:
        print("\nLoading model (this may take a minute)...")
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)

        # Get token IDs
        tokens_a = processor.tokenizer.encode('A', add_special_tokens=False)
        tokens_b = processor.tokenizer.encode('B', add_special_tokens=False)
        ids_A = [tokens_a[0]]
        ids_B = [tokens_b[0]]

        print(f"Model loaded! A token: {ids_A[0]}, B token: {ids_B[0]}\n")

    # Process each sample
    print("=" * 70)
    for i, (idx, row) in enumerate(df.iterrows()):
        audio_path = row['audio_path']
        if not audio_path.startswith('data/'):
            audio_path = 'data/' + audio_path

        ground_truth = row[label_col]
        clip_id = row.get('clip_id', f'sample_{i}')
        snr = row.get('snr_db', 'N/A')
        duration = row.get('duration_ms', 'N/A')

        print(f"\n[{i+1}/{len(df)}] {clip_id}")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Duration: {duration}ms | SNR: {snr}dB")
        print(f"  Path: {audio_path}")

        # Load and play audio
        if os.path.exists(audio_path):
            audio, sr = sf.read(audio_path)
            print(f"  Audio: {len(audio)} samples @ {sr}Hz ({len(audio)/sr:.2f}s)")

            # Play audio
            play_audio(audio, sr)

            # Get model prediction
            if model is not None:
                target_sr = processor.feature_extractor.sampling_rate
                if sr != target_sr:
                    import torchaudio.transforms as T
                    resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                    audio_resampled = resampler(torch.tensor(audio)).numpy()
                else:
                    audio_resampled = audio

                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": "placeholder"},
                            {"type": "text", "text": prompt_text or "Does this contain speech? A=yes, B=no"}
                        ]
                    }
                ]

                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                inputs = processor(
                    text=[text_prompt],
                    audio=[audio_resampled],
                    sampling_rate=target_sr,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits[0, -1, :]
                logit_A = logits[ids_A[0]].item()
                logit_B = logits[ids_B[0]].item()

                logit_diff = logit_A - logit_B
                prob_A = torch.sigmoid(torch.tensor(logit_diff)).item()

                prediction = 'SPEECH' if prob_A > 0.5 else 'NONSPEECH'
                correct = "✓" if prediction == ground_truth else "✗"

                print(f"\n  MODEL PREDICTION: {prediction} {correct}")
                print(f"  Logit A (SPEECH): {logit_A:.3f}")
                print(f"  Logit B (NONSPEECH): {logit_B:.3f}")
                print(f"  Logit diff (A-B): {logit_diff:.3f}")
                print(f"  Prob(SPEECH): {prob_A:.3f}")
        else:
            print(f"  ERROR: File not found!")

        print("-" * 70)

        # Wait for user input
        try:
            input("Press Enter for next sample (Ctrl+C to quit)...")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

    print("\nDone!")


if __name__ == "__main__":
    main()
