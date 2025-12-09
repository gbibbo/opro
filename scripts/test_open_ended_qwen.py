#!/usr/bin/env python3
"""
Test open-ended prompts with Qwen2-Audio on real audio samples.

This script:
1. Loads a small set of audio samples (SPEECH and NONSPEECH)
2. Tests both constrained and open-ended prompts
3. Records Qwen's raw outputs for analysis
4. Compares normalization results
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.qsm.models.qwen_audio import Qwen2AudioClassifier
from src.qsm.utils.normalize import normalize_to_binary


def test_prompts_on_samples(
    audio_paths: list[str],
    ground_truths: list[str],
    model: Qwen2AudioClassifier,
    output_file: str = "test_open_ended_results.json"
):
    """
    Test different prompt types on audio samples.

    Args:
        audio_paths: List of audio file paths
        ground_truths: List of ground truth labels (SPEECH/NONSPEECH)
        model: Qwen2AudioClassifier instance
        output_file: Where to save results
    """
    # Define test prompts
    prompts = {
        "constrained_ab": "Choose: A) SPEECH B) NON-SPEECH. Answer with A or B.",
        "constrained_direct": "Does this audio contain human speech? Answer: SPEECH or NON-SPEECH.",
        "open_what": "What do you hear in this audio?",
        "open_describe": "Describe the sound in this clip.",
        "open_yesno": "Is there human speech in this audio?",
        "open_type": "What type of sound is this?",
    }

    results = []

    for audio_path, ground_truth in zip(audio_paths, ground_truths):
        print(f"\n{'='*70}")
        print(f"Audio: {Path(audio_path).name}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*70}")

        sample_results = {
            "audio_path": str(audio_path),
            "ground_truth": ground_truth,
            "prompts": {}
        }

        for prompt_name, prompt_text in prompts.items():
            print(f"\n--- {prompt_name} ---")
            print(f"Prompt: {prompt_text}")

            # Set prompt
            model.user_prompt = prompt_text

            # Get prediction with raw output
            try:
                result = model.predict(audio_path, return_scores=True)
                raw_output = result.raw_output

                print(f"Raw output: '{raw_output}'")

                # Normalize
                if prompt_name.startswith("constrained_ab"):
                    # A/B mapping
                    mapping = {"A": "SPEECH", "B": "NONSPEECH"}
                    normalized, confidence = normalize_to_binary(
                        raw_output,
                        mode="ab",
                        mapping=mapping
                    )
                else:
                    # Auto mode
                    normalized, confidence = normalize_to_binary(
                        raw_output,
                        mode="auto"
                    )

                is_correct = (normalized == ground_truth) if normalized else False

                print(f"Normalized: {normalized} (confidence: {confidence:.2f})")
                print(f"Correct: {'✓' if is_correct else '✗'}")

                # Store results
                sample_results["prompts"][prompt_name] = {
                    "prompt_text": prompt_text,
                    "raw_output": raw_output,
                    "normalized_label": normalized,
                    "confidence": float(confidence),
                    "is_correct": is_correct
                }

            except Exception as e:
                print(f"ERROR: {e}")
                sample_results["prompts"][prompt_name] = {
                    "prompt_text": prompt_text,
                    "error": str(e)
                }

        results.append(sample_results)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")

    # Print summary
    print_summary(results)

    return results


def print_summary(results: list[dict]):
    """Print summary of results."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Collect accuracy by prompt type
    prompt_accuracies = {}

    for sample in results:
        for prompt_name, prompt_result in sample["prompts"].items():
            if "error" in prompt_result:
                continue

            if prompt_name not in prompt_accuracies:
                prompt_accuracies[prompt_name] = {"correct": 0, "total": 0}

            prompt_accuracies[prompt_name]["total"] += 1
            if prompt_result["is_correct"]:
                prompt_accuracies[prompt_name]["correct"] += 1

    # Print by prompt type
    for prompt_name, stats in sorted(prompt_accuracies.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"{prompt_name:25s}: {stats['correct']}/{stats['total']} = {acc*100:.1f}%")


def main():
    parser = argparse.ArgumentParser("Test open-ended prompts with Qwen2-Audio")

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest CSV with audio paths and labels"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to test per class (default: 5)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional LoRA checkpoint path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/test_open_ended_qwen.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )

    args = parser.parse_args()

    # Load manifest
    print("Loading manifest...")
    manifest_df = pd.read_csv(args.manifest)

    # Detect label column
    label_col = "label" if "label" in manifest_df.columns else "ground_truth"

    # Sample balanced dataset
    n_per_class = args.n_samples

    speech_samples = manifest_df[manifest_df[label_col] == "SPEECH"].sample(
        n=min(n_per_class, len(manifest_df[manifest_df[label_col] == "SPEECH"])),
        random_state=args.seed
    )
    nonspeech_samples = manifest_df[manifest_df[label_col] == "NONSPEECH"].sample(
        n=min(n_per_class, len(manifest_df[manifest_df[label_col] == "NONSPEECH"])),
        random_state=args.seed
    )

    test_df = pd.concat([speech_samples, nonspeech_samples], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    print(f"\nSelected {len(test_df)} samples:")
    print(f"  SPEECH: {len(speech_samples)}")
    print(f"  NONSPEECH: {len(nonspeech_samples)}")

    # Load model
    print("\nLoading Qwen2-Audio model...")
    model = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        load_in_4bit=True,
    )

    if args.checkpoint:
        print(f"Loading LoRA checkpoint: {args.checkpoint}")
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, args.checkpoint)
        model.model.eval()

    # Test prompts
    audio_paths = test_df["audio_path"].tolist()
    ground_truths = test_df[label_col].tolist()

    results = test_prompts_on_samples(
        audio_paths=audio_paths,
        ground_truths=ground_truths,
        model=model,
        output_file=args.output
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
