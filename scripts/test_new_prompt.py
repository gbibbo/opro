#!/usr/bin/env python3
"""
Quick test of the new multiple-choice prompt.
Run locally with few samples.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# New prompt
PROMPT = """What is in this audio?
A) Human speech
B) Music
C) Noise/silence
D) Other sounds"""


def main():
    # Load test data - small subset
    csv_path = "data/processed/grouped_split_with_dev/dev_metadata.csv"
    df = pd.read_csv(csv_path)

    # Filter: 1000ms, SNR=20 (cleanest)
    df = df[(df['duration_ms'] == 1000) & (df['snr_db'] == 20)]

    # Take just 10 samples for quick test
    df = df.head(10)

    print(f"Testing {len(df)} samples")
    print(f"Prompt: {PROMPT[:50]}...")
    print("=" * 60)

    # Load model
    from src.qsm.models.qwen_audio import Qwen2AudioClassifier

    model = Qwen2AudioClassifier(load_in_4bit=True)
    model.user_prompt = PROMPT

    # Evaluate
    correct = 0
    total = 0
    results = []

    for idx, row in df.iterrows():
        audio_path = row['audio_path']
        ground_truth = row['ground_truth']

        result = model.predict(audio_path)
        prediction = result.label
        raw = result.raw_output

        is_correct = prediction == ground_truth
        if is_correct:
            correct += 1
        total += 1

        mark = "OK" if is_correct else "WRONG"
        print(f"[{mark}] GT={ground_truth}, Pred={prediction}, Raw='{raw}'")

        results.append({
            'audio_path': audio_path,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'raw_output': raw,
            'correct': is_correct
        })

    # Summary
    accuracy = correct / total * 100 if total > 0 else 0
    print("=" * 60)
    print(f"ACCURACY: {correct}/{total} = {accuracy:.1f}%")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("results_new_prompt_test.csv", index=False)
    print(f"Results saved to: results_new_prompt_test.csv")


if __name__ == "__main__":
    main()
