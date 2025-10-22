#!/usr/bin/env python3
"""
OPRO Post Fine-Tuning: Prompt Optimization on Frozen Fine-Tuned Model

Based on: Yang et al., "Large Language Models as Optimizers" (2023)
https://arxiv.org/abs/2309.03409

Concept:
- Fine-tuned model has learned good audio features
- OPRO optimizes the PROMPT to leverage those features better
- Model stays frozen (no gradient updates)
- Only the instruction text is optimized

Usage:
    python scripts/opro_post_ft.py \
        --checkpoint checkpoints/ablations/LORA_attn_mlp/seed_42/final \
        --train_csv data/processed/grouped_split/dev_metadata.csv \
        --output_dir results/opro_post_ft \
        --num_iterations 20 \
        --samples_per_iter 10
"""

import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor
)
from datetime import datetime

# Import evaluation function from evaluate_with_logits.py
import sys
sys.path.append(str(Path(__file__).parent))


def get_ab_token_ids(tokenizer):
    """
    Get all token IDs that represent 'A' or 'B'.
    Handles both variants (with and without leading space).
    """
    ids_A = []
    ids_B = []

    # Variant 1: No space
    ids_A.extend(tokenizer.encode('A', add_special_tokens=False))
    ids_B.extend(tokenizer.encode('B', add_special_tokens=False))

    # Variant 2: Leading space
    ids_A_space = tokenizer.encode(' A', add_special_tokens=False)
    ids_B_space = tokenizer.encode(' B', add_special_tokens=False)

    # Add space variants (avoiding duplicates)
    for id_val in ids_A_space:
        if id_val not in ids_A:
            ids_A.append(id_val)
    for id_val in ids_B_space:
        if id_val not in ids_B:
            ids_B.append(id_val)

    return ids_A, ids_B


def evaluate_sample_logits(model, processor, audio_path, ids_A, ids_B,
                           system_prompt, user_prompt, temperature=1.0):
    """
    Evaluate single sample using logit extraction.
    """
    import soundfile as sf

    # Load audio
    audio, sr = sf.read(audio_path)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Create conversation format
    conversation = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    # Process
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True)

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward pass (no generation)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits for last position
    logits = outputs.logits[0, -1, :]

    # Get logits for A and B tokens
    logits_A = logits[ids_A]
    logits_B = logits[ids_B]

    # Apply temperature
    logits_A = logits_A / temperature
    logits_B = logits_B / temperature

    # Aggregate using logsumexp
    logit_A = torch.logsumexp(logits_A, dim=0).item()
    logit_B = torch.logsumexp(logits_B, dim=0).item()

    # Compute probabilities
    logit_diff = logit_A - logit_B
    prob_A = torch.sigmoid(torch.tensor(logit_diff)).item()
    prob_B = 1.0 - prob_A

    # Prediction
    prediction = 'A' if prob_A > prob_B else 'B'

    return {
        'prediction': prediction,
        'confidence': max(prob_A, prob_B),
        'prob_A': prob_A,
        'prob_B': prob_B
    }


def evaluate_prompt_on_samples(model, processor, ids_A, ids_B, samples,
                                system_prompt, user_prompt, temperature=1.0):
    """
    Evaluate a prompt on a set of samples.
    Returns accuracy.
    """
    correct = 0
    total = len(samples)

    for sample in tqdm(samples, desc="Evaluating prompt", leave=False):
        try:
            result = evaluate_sample_logits(
                model, processor,
                sample['audio_path'],
                ids_A, ids_B,
                system_prompt, user_prompt,
                temperature
            )

            if result['prediction'] == sample['ground_truth_token']:
                correct += 1
        except Exception as e:
            print(f"Error evaluating {sample['audio_path']}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def generate_candidate_prompts(prompt_history, num_candidates=8, use_llm=False):
    """
    Generate candidate prompts.

    Args:
        prompt_history: List of (prompt, accuracy) tuples
        num_candidates: Number of new prompts to generate
        use_llm: If True, use LLM to generate (requires API). If False, use templates.

    Returns:
        List of candidate prompt strings
    """
    if use_llm:
        # TODO: Implement LLM-based generation using meta-prompt
        raise NotImplementedError("LLM-based generation requires API access")

    # Template-based generation (fallback)
    templates = [
        # Baseline
        "Choose one:\nA) SPEECH (human voice)\nB) NONSPEECH (music/noise/silence/animals)\n\nAnswer with A or B ONLY.",

        # Emphasis variations
        "Listen carefully. Choose:\nA) SPEECH (human voice speaking)\nB) NONSPEECH (music, noise, silence, animals)\n\nAnswer A or B.",

        "Classify this audio:\nA) SPEECH - human voice, talking, speaking\nB) NONSPEECH - music, noise, environmental sounds, animals\n\nYour answer (A or B):",

        # Task framing
        "Your task: Detect if this audio contains speech.\nA) SPEECH (any human voice)\nB) NONSPEECH (no human voice)\n\nAnswer:",

        # Decision-focused
        "Does this audio contain human speech?\nA) Yes (SPEECH)\nB) No (NONSPEECH)\n\nAnswer A or B only.",

        # Feature-focused
        "Analyze the audio features. Is there human vocal activity?\nA) SPEECH detected\nB) NONSPEECH (music/noise/silence/other)\n\nAnswer:",

        # Detailed
        "Audio classification task:\n- A: SPEECH (human voice, talking, vocalizations)\n- B: NONSPEECH (music, environmental sounds, noise, silence, non-human)\n\nYour classification:",

        # Simple
        "A) SPEECH\nB) NONSPEECH\n\nWhich one?",

        # Question format
        "Is this audio speech or non-speech?\nA) SPEECH\nB) NONSPEECH\n\nAnswer:",

        # Instruction format
        "Identify the audio type. Select one:\nA) SPEECH (human speaking)\nB) NONSPEECH (other sounds)\n\nSelection:",
    ]

    # If we have history, weight towards better performing variations
    if len(prompt_history) > 0:
        # Find best performing prompt
        best_prompt, best_acc = max(prompt_history, key=lambda x: x[1])

        # Include best prompt and variations
        candidates = [best_prompt]

        # Add templates
        random.shuffle(templates)
        candidates.extend(templates[:num_candidates-1])
    else:
        # First iteration: use all templates
        candidates = templates[:num_candidates]

    return candidates


def opro_optimize(model, processor, train_df, ids_A, ids_B,
                  num_iterations=20, samples_per_iter=10,
                  num_candidates=8, temperature=1.0):
    """
    OPRO optimization loop.

    Args:
        model: Fine-tuned frozen model
        processor: Qwen2Audio processor
        train_df: DataFrame with training samples
        ids_A, ids_B: Token IDs for A and B
        num_iterations: Number of optimization iterations
        samples_per_iter: Number of samples to evaluate per iteration
        num_candidates: Number of candidate prompts per iteration
        temperature: Temperature for logit scaling

    Returns:
        best_prompt, best_accuracy, history
    """
    system_prompt = "You classify audio content."

    # Prepare samples
    samples = []
    for _, row in train_df.iterrows():
        samples.append({
            'audio_path': row['audio_path'],
            'ground_truth_token': row['ground_truth_token']
        })

    prompt_history = []  # List of (prompt, accuracy) tuples
    best_prompt = None
    best_accuracy = 0.0

    print(f"\nStarting OPRO optimization:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Samples per iteration: {samples_per_iter}")
    print(f"  Candidates per iteration: {num_candidates}")
    print(f"  Total samples: {len(samples)}")
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

            accuracy = evaluate_prompt_on_samples(
                model, processor, ids_A, ids_B,
                iter_samples, system_prompt, prompt, temperature
            )

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
    parser = argparse.ArgumentParser(description="OPRO Post Fine-Tuning")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--train_csv', type=str, required=True,
                        help='CSV with training/dev samples for optimization')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--num_iterations', type=int, default=20,
                        help='Number of OPRO iterations')
    parser.add_argument('--samples_per_iter', type=int, default=10,
                        help='Number of samples to evaluate per iteration')
    parser.add_argument('--num_candidates', type=int, default=8,
                        help='Number of candidate prompts per iteration')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for logit scaling')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPRO POST FINE-TUNING: Prompt Optimization on Frozen Model")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Output: {args.output_dir}")

    # Load model
    print(f"\nLoading fine-tuned model...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()  # Inference mode

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    print(f"Model loaded on: {model.device}")
    print(f"Model frozen: All {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load processor
    print(f"\nLoading processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")

    # Get token IDs
    ids_A, ids_B = get_ab_token_ids(processor.tokenizer)
    print(f"Token IDs: A={ids_A}, B={ids_B}")

    # Load data
    print(f"\nLoading training data...")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loaded {len(train_df)} samples")
    print(f"  SPEECH:    {(train_df['ground_truth'] == 'SPEECH').sum()}")
    print(f"  NONSPEECH: {(train_df['ground_truth'] == 'NONSPEECH').sum()}")

    # Run OPRO
    best_prompt, best_accuracy, history = opro_optimize(
        model, processor, train_df, ids_A, ids_B,
        num_iterations=args.num_iterations,
        samples_per_iter=args.samples_per_iter,
        num_candidates=args.num_candidates,
        temperature=args.temperature
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
                'temperature': args.temperature,
                'seed': args.seed,
                'timestamp': datetime.now().isoformat()
            }
        }, f, indent=2)
    print(f"History saved to: {history_file}")

    print(f"\n{'='*80}")
    print("Next steps:")
    print(f"  1. Evaluate best prompt on test set:")
    print(f"     python scripts/evaluate_with_logits.py \\")
    print(f"       --checkpoint {args.checkpoint} \\")
    print(f"       --test_csv data/processed/grouped_split/test_metadata.csv \\")
    print(f"       --prompt \"{best_prompt}\" \\")
    print(f"       --output_csv results/comparisons/ft_opro.csv")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
