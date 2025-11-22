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
    import os
    from pathlib import Path

    # Hardening: resolve paths robustly
    audio_path = Path(audio_path)
    if not audio_path.exists():
        # Try with data/ prefix
        audio_path = Path("data") / audio_path

    if not audio_path.exists():
        print(f"  ERROR: File not found: {audio_path}")
        return False

    try:
        result = model.predict(str(audio_path.resolve()))

        # Robust parsing: extract SPEECH/NONSPEECH from any format
        pred = str(result.label).strip().upper()

        # Normalize: handle A/B, JSON, or direct labels
        if "SPEECH" in pred and "NONSPEECH" not in pred:
            pred = "SPEECH"
        elif "NONSPEECH" in pred or "NON-SPEECH" in pred or "NO SPEECH" in pred:
            pred = "NONSPEECH"
        elif pred in ["A", "YES", "TRUE"]:
            pred = "SPEECH"
        elif pred in ["B", "NO", "FALSE"]:
            pred = "NONSPEECH"

        return pred == ground_truth
    except Exception as e:
        print(f"  Error processing {audio_path}: {e}")
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


def generate_candidate_prompts(prompt_history, num_candidates=12):
    """
    Generate candidate prompts using best-practice templates.

    All templates request SPEECH or NONSPEECH (no A/B, no JSON) for deterministic parsing.

    Based on prompt engineering research:
    - Direct label output (SPEECH/NONSPEECH)
    - Label descriptions (verbalizers)
    - Constrained output space
    - Few-shot examples
    - Calibration-friendly formats
    """
    templates = [
        # 1) Minimal direct
        "Does this audio contain human speech? Answer exactly one token: SPEECH or NONSPEECH.",

        # 2) Binary decision
        "Binary decision. Output exactly one token: SPEECH or NONSPEECH.",

        # 3) Label descriptions (verbalizers)
        "Decide the dominant content.\nDefinitions:\n- SPEECH = human voice, spoken words, syllables, conversational cues.\n- NONSPEECH = music, tones/beeps, environmental noise, silence.\nOutput exactly: SPEECH or NONSPEECH.",

        # 4) Contrastive/counter-examples
        "Detect human speech. Treat the following as NONSPEECH: pure tones/beeps, clicks, clock ticks, music, environmental noise, silence.\nAnswer: SPEECH or NONSPEECH.",

        # 5) 1-shot consistency
        "Example:\nAudio→ crowd noise, music → Output: NONSPEECH\nNow classify the new audio. Output exactly ONE token: SPEECH or NONSPEECH.",

        # 6) Forced decision
        "Make a definite decision for the clip.\nOutput exactly one token: SPEECH or NONSPEECH.",

        # 7) Conservative (reduce false positives)
        "Label SPEECH only if human voice is clearly present; otherwise label NONSPEECH.\nAnswer: SPEECH or NONSPEECH.",

        # 8) Liberal (reduce false negatives)
        "If there is any hint of human voice (even faint/short), label SPEECH; otherwise NONSPEECH.\nAnswer: SPEECH or NONSPEECH.",

        # 9) Acoustic focus (vocal tract features)
        "Focus on cues of human vocal tract (formants, syllabic rhythm, consonant onsets).\nAnswer exactly: SPEECH or NONSPEECH.",

        # 10) Task-oriented
        "TASK: Speech detection. Is human voice/speech present in this audio?\nAnswer: SPEECH or NONSPEECH.",

        # 11) Confidence calibration
        "Binary classification task.\nQ: Does this contain human speech?\nIf confident YES → SPEECH\nIf confident NO → NONSPEECH\nAnswer:",

        # 12) Delimiters
        "You will answer with one token only.\n<question>Does this audio contain human speech?</question>\n<answer>SPEECH or NONSPEECH only</answer>",

        # 13) Explicit instruction
        "Classify this audio. Output only: SPEECH or NONSPEECH.",

        # 14) Focus instruction
        "Listen for human voice. If present: SPEECH. Otherwise: NONSPEECH.\nAnswer:",

        # 15) Simplified baseline
        "Human speech present? Answer: SPEECH or NONSPEECH.",
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


def stratified_sample_df(df, n, seed=42):
    """
    Stratified sampling: 50/50 SPEECH/NONSPEECH.
    """
    n_half = n // 2
    speech_df = df[df['ground_truth'] == 'SPEECH']
    nonspeech_df = df[df['ground_truth'] == 'NONSPEECH']

    # Sample with replacement if needed
    replace_speech = len(speech_df) < n_half
    replace_nonspeech = len(nonspeech_df) < (n - n_half)

    a = speech_df.sample(n=n_half, replace=replace_speech, random_state=seed)
    b = nonspeech_df.sample(n=n - n_half, replace=replace_nonspeech, random_state=seed)

    # Shuffle combined sample
    return pd.concat([a, b]).sample(frac=1, random_state=seed)


def opro_optimize(model, train_df, num_iterations=15, samples_per_iter=20, num_candidates=8, seed=42):
    """
    OPRO optimization loop using generation-based evaluation.

    Args:
        model: Qwen2AudioClassifier instance (frozen)
        train_df: DataFrame with training samples
        num_iterations: Number of optimization iterations
        samples_per_iter: Number of samples to evaluate per iteration
        num_candidates: Number of candidate prompts per iteration
        seed: Random seed for reproducibility

    Returns:
        best_prompt, best_accuracy, history
    """
    # Configure model for deterministic generation
    import torch
    torch.manual_seed(seed)

    if hasattr(model, 'model') and hasattr(model.model, 'generation_config'):
        cfg = model.model.generation_config
        cfg.do_sample = False
        cfg.temperature = 0.0
        cfg.top_p = 1.0
        cfg.max_new_tokens = 3
        print("Model configured for deterministic generation (greedy, T=0)")

    prompt_history = []  # List of (prompt, accuracy) tuples

    # Initialize with baseline prompt
    best_prompt = "Does this audio contain human speech? Answer exactly one token: SPEECH or NONSPEECH."
    best_accuracy = 0.0

    print(f"\nStarting OPRO optimization:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Samples per iteration: {samples_per_iter} (stratified 50/50)")
    print(f"  Candidates per iteration: {num_candidates}")
    print(f"  Total samples available: {len(train_df)}")
    print(f"  Baseline prompt: {best_prompt[:60]}...")
    print()

    for iteration in range(num_iterations):
        print(f"\n{'='*80}")
        print(f"Iteration {iteration+1}/{num_iterations}")
        print(f"{'='*80}")

        # Stratified sampling: 50/50 SPEECH/NONSPEECH
        iter_samples_df = stratified_sample_df(train_df, samples_per_iter, seed=seed+iteration)
        iter_samples = iter_samples_df[['audio_path', 'ground_truth']].to_dict('records')

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
        num_candidates=args.num_candidates,
        seed=args.seed
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
