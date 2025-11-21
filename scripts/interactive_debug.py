#!/usr/bin/env python3
"""
Interactive debugging script for speech classification.
Shows Qwen's actual generated response and input spectrogram.
"""

import argparse
import pandas as pd
import soundfile as sf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def save_spectrogram(audio, sr, output_path="debug_spectrogram.png"):
    """Save spectrogram of the audio that Qwen receives."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Waveform
    time = np.arange(len(audio)) / sr
    axes[0].plot(time, audio, linewidth=0.5)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Waveform - {len(audio)/sr:.2f}s @ {sr}Hz")
    axes[0].set_xlim(0, len(audio)/sr)

    # Spectrogram
    axes[1].specgram(audio, Fs=sr, NFFT=512, noverlap=256, cmap='viridis')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Spectrogram (what Qwen sees)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Interactive audio debugging")
    parser.add_argument("--test_csv", type=str, default="data/processed/grouped_split_with_dev/dev_metadata.csv")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--snr", type=float, default=20)
    parser.add_argument("--duration", type=int, default=1000)
    parser.add_argument("--label", type=str, default=None, choices=["SPEECH", "NONSPEECH"])
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    # Default open-ended prompt
    prompt_text = "What do you hear in this audio? Describe it."

    # Load and filter data
    df = pd.read_csv(args.test_csv)
    label_col = 'ground_truth' if 'ground_truth' in df.columns else 'label'

    df = df[df['snr_db'] == args.snr]
    df = df[df['duration_ms'] == args.duration]

    if args.label:
        df = df[df[label_col] == args.label]

    print(f"Samples: {len(df)} (duration={args.duration}ms, SNR={args.snr}dB)", flush=True)

    if len(df) == 0:
        print("No samples match filters!")
        return

    if len(df) > args.n_samples:
        df = df.sample(n=args.n_samples, random_state=42)

    # Load model
    model = None
    processor = None

    if not args.no_model:
        print("\nLoading Qwen model...", flush=True)
        from src.qsm.models.qwen_audio import Qwen2AudioClassifier

        model = Qwen2AudioClassifier(load_in_4bit=True)
        processor = model.processor
        print("Model loaded!\n", flush=True)

    # Process samples
    i = 0
    df_list = list(df.iterrows())

    while i < len(df_list):
        idx, row = df_list[i]
        audio_path = row['audio_path']
        if not audio_path.startswith('data/'):
            audio_path = 'data/' + audio_path

        ground_truth = row[label_col]

        print("=" * 60, flush=True)
        print(f"SAMPLE {i+1}/{len(df_list)}: {os.path.basename(audio_path)}", flush=True)
        print(f"Ground Truth: {ground_truth}", flush=True)
        print("-" * 60, flush=True)
        print(f"PROMPT: {prompt_text}", flush=True)
        print("-" * 60, flush=True)

        if not os.path.exists(audio_path):
            print("ERROR: File not found!", flush=True)
            i += 1
            continue

        # Load and process audio exactly as Qwen receives it
        audio, sr = sf.read(audio_path)

        # Resample to Qwen's expected sample rate if needed
        if processor is not None:
            target_sr = processor.feature_extractor.sampling_rate
            if sr != target_sr:
                import torchaudio.transforms as T
                import torch
                resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                audio = resampler(torch.tensor(audio)).numpy()
                sr = target_sr

        print(f"Audio: {len(audio)/sr:.2f}s @ {sr}Hz", flush=True)

        # Save spectrogram of what Qwen receives
        spec_path = save_spectrogram(audio, sr)
        print(f"Spectrogram saved: {spec_path}", flush=True)

        if model is not None:
            # Set the prompt and get prediction
            model.user_prompt = prompt_text
            result = model.predict(audio_path)

            print("\n" + "=" * 60, flush=True)
            print("QWEN'S RESPONSE:", flush=True)
            print("=" * 60, flush=True)
            response = result.raw_output
            print(response, flush=True)
            print("=" * 60, flush=True)

        print("\nOptions:", flush=True)
        print("  [Enter] Next sample", flush=True)
        print("  [p]     Change prompt", flush=True)
        print("  [q]     Quit", flush=True)

        try:
            choice = input("\nYour choice: ").strip().lower()

            if choice == 'q':
                print("Exiting...")
                break
            elif choice == 'p':
                new_prompt = input("Enter new prompt: ").strip()
                if new_prompt:
                    prompt_text = new_prompt
                    print(f"Prompt changed to: {prompt_text}")
                # Don't increment i, repeat same sample with new prompt
            else:
                i += 1  # Next sample

        except KeyboardInterrupt:
            print("\n")
            break

    print("Done!")


if __name__ == "__main__":
    main()
