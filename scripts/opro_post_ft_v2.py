#!/usr/bin/env python3
"""
OPRO Post Fine-Tuning: Prompt Optimization on Frozen Fine-Tuned Model (v2)

This version uses Qwen2AudioClassifier (generation-based) instead of logit extraction.
Based on evaluate_with_generation.py which is known to work correctly.

Usage:
    python scripts/opro_post_ft_v2.py \
        --no_lora \
        --train_csv data/processed/experimental_variants/dev_metadata.csv \
        --output_dir results/opro_base \
        --num_iterations 15 \
        --samples_per_iter 20
"""

import argparse
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


def evaluate_sample_with_model(model, audio_path, ground_truth):
    """
    Evaluate a single sample using Qwen2AudioClassifier.

    Returns:
        correct: bool (whether prediction matches ground truth)
    """
    try:
        result = model.predict(audio_path)
        prediction = result.label

        # Map ground truth: SPEECH -> A, NONSPEECH -> B
        # Model returns SPEECH/NONSPEECH directly
        return prediction == ground_truth
    except Exception as e:
        print(f"  Error on {audio_path}: {e}")
        return False


def evaluate_prompt_on_samples(model, samples, prompt):
    """
    Evaluate a prompt on a set of samples.

    Args:
        model: Qwen2AudioClassifier instance
        samples: List of {audio_path, ground_truth} dicts
        prompt: User prompt string to test

    Returns:
        accuracy: float
    """
    # Update model's prompt
    model.user_prompt = prompt

    correct = 0
    total = len(samples)
    errors = []

    for sample in tqdm(samples, desc="  Evaluating", leave=False):
        is_correct = evaluate_sample_with_model(
            model,
            sample['audio_path'],
            sample['ground_truth']
        )

        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def generate_candidate_prompts(prompt_history, num_candidates=8):
    """
    Generate candidate prompts using templates.

    All templates use 4-option format: A=Speech, B=Music, C=Noise/silence, D=Other
    """
    templates = [
        # Baseline (from prompts/prompt_base.txt)
        "What is in this audio?\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds",

        # Explicit instruction
        "Listen carefully and select one option:\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds",

        # Question format
        "What type of audio is this?\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds",

        # Classify format
        "Classify this audio content:\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds",

        # Task framing
        "Identify the audio content. Choose one:\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds",

        # Detailed descriptions
        "What is in this audio?\nA) Human speech (talking, speaking, vocalizations)\nB) Music (instruments, singing)\nC) Noise or silence (background noise, quiet)\nD) Other sounds (animals, environment)",

        # Simple/direct
        "Select one:\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds",

        # Instruction style
        "Audio classification:\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds\n\nYour answer:",

        # Emphatic
        "Listen to this audio. What do you hear?\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds",

        # Analysis framing
        "Analyze this audio and categorize it:\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds",
    ]

    # If we have history, include best performing prompt
    if len(prompt_history) > 0:
        best_prompt, best_acc = max(prompt_history, key=lambda x: x[1])
        candidates = [best_prompt]

        # Add templates
        random.shuffle(templates)
        candidates.extend(templates[:num_candidates-1])
    else:
        # First iteration: use templates
        candidates = templates[:num_candidates]

    return candidates


def opro_optimize(model, train_df, num_iterations=15, samples_per_iter=20, num_candidates=8):
    """
    OPRO optimization loop using generation-based evaluation.

    Args:
        model: Qwen2AudioClassifier instance (frozen)
        train_df: DataFrame with training samples
        num_iterations: Number of optimization iterations
        samples_per_iter: Number of samples to evaluate per iteration
        num_candidates: Number of candidate prompts per iteration

    Returns:
        best_prompt, best_accuracy, history
    """
    # Prepare samples
    samples = []
    for _, row in train_df.iterrows():
        samples.append({
            'audio_path': row['audio_path'],
            'ground_truth': row['ground_truth']  # SPEECH or NONSPEECH
        })

    prompt_history = []  # List of (prompt, accuracy) tuples

    # Initialize with baseline prompt
    best_prompt = "What is in this audio?\nA) Human speech\nB) Music\nC) Noise/silence\nD) Other sounds"
    best_accuracy = 0.0

    print(f"\nStarting OPRO optimization:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Samples per iteration: {samples_per_iter}")
    print(f"  Candidates per iteration: {num_candidates}")
    print(f"  Total samples: {len(samples)}")
    print(f"  Baseline prompt: {best_prompt}")
    print()

    for iteration in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration+1}/{num_iterations}")
        print(f"{'='*80}")

        # Sample subset for this iteration
        iter_samples = random.sample(samples, min(samples_per_iter, len(samples)))

        # Generate candidate prompts
        candidates = generate_candidate_prompts(prompt_history, num_candidates)

        print(f"\nEvaluating {len(candidates)} candidate prompts...")

        # Evaluate each candidate
        candidate_results = []
        for i, prompt in enumerate(candidates):
            print(f"\n[{i+1}/{len(candidates)}] Testing prompt:")
            print(f"  {prompt[:80]}...")

            accuracy = evaluate_prompt_on_samples(model, iter_samples, prompt)

            candidate_results.append((prompt, accuracy))
            print(f"  Accuracy: {accuracy:.1%}")

            # Update history
            prompt_history.append((prompt, accuracy))

            # Update best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompt = prompt
                print(f"  ✓ New best! {best_accuracy:.1%}")

        # Summary
        best_this_iter = max(candidate_results, key=lambda x: x[1])
        print(f"\nIteration {iteration+1} summary:")
        print(f"  Best this iteration: {best_this_iter[1]:.1%}")
        print(f"  Best overall:        {best_accuracy:.1%}")

    return best_prompt, best_accuracy, prompt_history


def main():
    parser = argparse.ArgumentParser(description="OPRO Post Fine-Tuning (Generation-based)")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--no_lora', action='store_true',
                        help='Use base model without LoRA')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='CSV with training/dev samples for optimization')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--num_iterations', type=int, default=15,
                        help='Number of OPRO iterations')
    parser.add_argument('--samples_per_iter', type=int, default=20,
                        help='Number of samples to evaluate per iteration')
    parser.add_argument('--num_candidates', type=int, default=8,
                        help='Number of candidate prompts per iteration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Validate args
    if not args.no_lora and args.checkpoint is None:
        parser.error("--checkpoint is required unless --no_lora is specified")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPRO: Prompt Optimization (Generation-based)")
    print("=" * 80)
    print(f"\nModel: {'BASE (no LoRA)' if args.no_lora else args.checkpoint}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Output: {args.output_dir}")

    # Load model using Qwen2AudioClassifier
    print(f"\nLoading model...")
    from src.qsm.models.qwen_audio import Qwen2AudioClassifier

    if args.no_lora:
        model = Qwen2AudioClassifier(load_in_4bit=True)
    else:
        # Load base model then apply LoRA
        model = Qwen2AudioClassifier(load_in_4bit=True)
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, args.checkpoint)
        model.model.eval()
        print(f"LoRA checkpoint loaded: {args.checkpoint}")

    print(f"Model loaded!")

    # Load data
    print(f"\nLoading training data...")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loaded {len(train_df)} samples")
    print(f"  SPEECH:    {(train_df['ground_truth'] == 'SPEECH').sum()}")
    print(f"  NONSPEECH: {(train_df['ground_truth'] == 'NONSPEECH').sum()}")

    # Run OPRO
    best_prompt, best_accuracy, history = opro_optimize(
        model, train_df,
        num_iterations=args.num_iterations,
        samples_per_iter=args.samples_per_iter,
        num_candidates=args.num_candidates
    )

    # Save results
    print(f"\n{'='*80}")
    print("OPRO OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest prompt (accuracy: {best_accuracy:.1%}):")
    print(f"{best_prompt}")

    # Save best prompt
    best_prompt_file = output_dir / "best_prompt.txt"
    best_prompt_file.write_text(best_prompt)
    print(f"\nBest prompt saved to: {best_prompt_file}")

    # Save history
    history_file = output_dir / "optimization_history.json"
    with open(history_file, 'w') as f:
        json.dump({
            'best_accuracy': best_accuracy,
            'best_prompt': best_prompt,
            'history': [(p, float(a)) for p, a in history],
            'config': {
                'checkpoint': args.checkpoint,
                'num_iterations': args.num_iterations,
                'samples_per_iter': args.samples_per_iter,
                'num_candidates': args.num_candidates,
                'seed': args.seed,
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)
    print(f"History saved to: {history_file}")

    print(f"\n{'='*80}")
    print("Next steps:")
    print(f"  1. Evaluate best prompt on test set:")
    print(f"     sbatch eval_model.sh --no-lora")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
