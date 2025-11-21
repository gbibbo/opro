#!/usr/bin/env python3
"""
Evaluate model using generation (not logits).
Uses multiple-choice prompt format that works better with Qwen2-Audio.

Usage:
    # Base model (no LoRA)
    python scripts/evaluate_with_generation.py --no-lora --test_csv data/processed/grouped_split_with_dev/test_metadata.csv

    # With LoRA checkpoint
    python scripts/evaluate_with_generation.py --checkpoint checkpoints/qwen_lora_seed42/final --test_csv data/processed/grouped_split_with_dev/test_metadata.csv

    # Filter by duration/SNR
    python scripts/evaluate_with_generation.py --no-lora --filter_duration 1000 --filter_snr 20
"""

import argparse
import pandas as pd
import os
import sys
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default prompt - multiple choice format
DEFAULT_PROMPT = """What is in this audio?
A) Human speech
B) Music
C) Noise/silence
D) Other sounds"""


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with generation")
    parser.add_argument("--test_csv", type=str, default="data/processed/grouped_split_with_dev/test_metadata.csv",
                        help="Path to test CSV")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save predictions CSV")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to LoRA checkpoint")
    parser.add_argument("--no-lora", action="store_true",
                        help="Use base model without LoRA")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="Path to prompt file")
    parser.add_argument("--filter_duration", type=int, default=None,
                        help="Filter by duration in ms")
    parser.add_argument("--filter_snr", type=float, default=None,
                        help="Filter by SNR in dB")
    args = parser.parse_args()

    # Validate args
    if not args.no_lora and args.checkpoint is None:
        parser.error("--checkpoint is required unless --no-lora is specified")

    # Load prompt
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r') as f:
            prompt = f.read().strip()
    else:
        prompt = DEFAULT_PROMPT

    print("=" * 60)
    print("EVALUATION WITH GENERATION")
    print("=" * 60)
    print(f"Test CSV: {args.test_csv}")
    print(f"LoRA: {'No' if args.no_lora else args.checkpoint}")
    print(f"Prompt:\n{prompt}")
    print("=" * 60)

    # Load test data
    df = pd.read_csv(args.test_csv)
    label_col = 'ground_truth' if 'ground_truth' in df.columns else 'label'

    print(f"Total samples: {len(df)}")

    # Apply filters
    if args.filter_duration is not None:
        df = df[df['duration_ms'] == args.filter_duration]
        print(f"Filtered by duration={args.filter_duration}ms: {len(df)} samples")

    if args.filter_snr is not None:
        df = df[df['snr_db'] == args.filter_snr]
        print(f"Filtered by SNR={args.filter_snr}dB: {len(df)} samples")

    if len(df) == 0:
        print("ERROR: No samples after filtering!")
        return

    # Load model
    print("\nLoading model...")
    from src.qsm.models.qwen_audio import Qwen2AudioClassifier

    if args.no_lora:
        model = Qwen2AudioClassifier(load_in_4bit=True)
    else:
        # Load base model then apply LoRA
        model = Qwen2AudioClassifier(load_in_4bit=True)
        # Apply LoRA weights
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, args.checkpoint)
        model.model.eval()
        print(f"LoRA checkpoint loaded: {args.checkpoint}")

    model.user_prompt = prompt
    print("Model loaded!\n")

    # Evaluate
    results = []
    correct = 0
    total = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        audio_path = row['audio_path']
        ground_truth = row[label_col]

        # Ensure path exists
        if not os.path.exists(audio_path):
            # Try adding data/ prefix
            if not audio_path.startswith('data/'):
                audio_path = 'data/' + audio_path

        if not os.path.exists(audio_path):
            print(f"WARNING: File not found: {audio_path}")
            continue

        try:
            result = model.predict(audio_path)
            prediction = result.label
            raw_output = result.raw_output
            confidence = result.confidence
            latency = result.latency_ms
        except Exception as e:
            print(f"ERROR processing {audio_path}: {e}")
            prediction = "UNKNOWN"
            raw_output = str(e)
            confidence = 0.0
            latency = 0.0

        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1
        total += 1

        results.append({
            'audio_path': row['audio_path'],
            'ground_truth': ground_truth,
            'prediction': prediction,
            'correct': is_correct,
            'confidence': confidence,
            'raw_output': raw_output,
            'latency_ms': latency,
            'duration_ms': row.get('duration_ms', None),
            'snr_db': row.get('snr_db', None),
        })

    # Calculate metrics
    accuracy = correct / total * 100 if total > 0 else 0

    # Per-class metrics
    results_df = pd.DataFrame(results)

    speech_samples = results_df[results_df['ground_truth'] == 'SPEECH']
    nonspeech_samples = results_df[results_df['ground_truth'] == 'NONSPEECH']

    speech_correct = speech_samples['correct'].sum() if len(speech_samples) > 0 else 0
    nonspeech_correct = nonspeech_samples['correct'].sum() if len(nonspeech_samples) > 0 else 0

    speech_acc = speech_correct / len(speech_samples) * 100 if len(speech_samples) > 0 else 0
    nonspeech_acc = nonspeech_correct / len(nonspeech_samples) * 100 if len(nonspeech_samples) > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"ACCURACY: {accuracy:.2f}%")
    print(f"")
    print(f"SPEECH accuracy: {speech_acc:.2f}% ({speech_correct}/{len(speech_samples)})")
    print(f"NONSPEECH accuracy: {nonspeech_acc:.2f}% ({nonspeech_correct}/{len(nonspeech_samples)})")
    print("=" * 60)

    # Save results
    if args.output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "baseline" if args.no_lora else "finetuned"
        args.output_csv = f"results/eval_{mode}_{timestamp}.csv"

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")

    # Per-condition breakdown if available
    if 'duration_ms' in results_df.columns and 'snr_db' in results_df.columns:
        print("\n" + "=" * 60)
        print("PER-CONDITION BREAKDOWN")
        print("=" * 60)

        for duration in sorted(results_df['duration_ms'].dropna().unique()):
            for snr in sorted(results_df['snr_db'].dropna().unique()):
                subset = results_df[(results_df['duration_ms'] == duration) & (results_df['snr_db'] == snr)]
                if len(subset) > 0:
                    acc = subset['correct'].mean() * 100
                    print(f"Duration={int(duration)}ms, SNR={int(snr)}dB: {acc:.1f}% ({len(subset)} samples)")


if __name__ == "__main__":
    main()
