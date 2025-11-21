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


def get_abcd_token_ids(tokenizer):
    """
    Get all token IDs that represent 'A', 'B', 'C', or 'D'.
    Handles both variants (with and without leading space).
    """
    ids_A = []
    ids_B = []
    ids_C = []
    ids_D = []

    # Variant 1: No space
    ids_A.extend(tokenizer.encode('A', add_special_tokens=False))
    ids_B.extend(tokenizer.encode('B', add_special_tokens=False))
    ids_C.extend(tokenizer.encode('C', add_special_tokens=False))
    ids_D.extend(tokenizer.encode('D', add_special_tokens=False))

    # Variant 2: Leading space
    for ids_list, letter in [(ids_A, 'A'), (ids_B, 'B'), (ids_C, 'C'), (ids_D, 'D')]:
        space_ids = tokenizer.encode(f' {letter}', add_special_tokens=False)
        for id_val in space_ids:
            if id_val not in ids_list:
                ids_list.append(id_val)

    return ids_A, ids_B, ids_C, ids_D


def evaluate_sample_logits(model, processor, audio_path, ids_A, ids_B, ids_C, ids_D,
                           system_prompt, user_prompt, temperature=1.0):
    """
    Evaluate single sample using logit extraction.
    Uses 4-option format: A=SPEECH, B/C/D=NONSPEECH
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

    # Get logits for all 4 options
    logits_A = logits[ids_A]
    logits_B = logits[ids_B]
    logits_C = logits[ids_C]
    logits_D = logits[ids_D]

    # Apply temperature
    logits_A = logits_A / temperature
    logits_B = logits_B / temperature
    logits_C = logits_C / temperature
    logits_D = logits_D / temperature

    # Aggregate using logsumexp
    logit_A = torch.logsumexp(logits_A, dim=0).item()
    logit_B = torch.logsumexp(logits_B, dim=0).item()
    logit_C = torch.logsumexp(logits_C, dim=0).item()
    logit_D = torch.logsumexp(logits_D, dim=0).item()

    # Map to SPEECH (A) vs NONSPEECH (B/C/D)
    # Option A = SPEECH (Human speech)
    # Options B/C/D = NONSPEECH (Music/Noise/Other)
    logit_speech = logit_A
    logit_nonspeech = torch.logsumexp(torch.tensor([logit_B, logit_C, logit_D]), dim=0).item()

    # Compute probabilities
    logit_diff = logit_speech - logit_nonspeech
    prob_speech = torch.sigmoid(torch.tensor(logit_diff)).item()
    prob_nonspeech = 1.0 - prob_speech

    # Prediction: A if speech, B if nonspeech (for ground truth comparison)
    prediction = 'A' if prob_speech > prob_nonspeech else 'B'

    return {
        'prediction': prediction,
        'confidence': max(prob_speech, prob_nonspeech),
        'prob_A': prob_speech,
        'prob_B': prob_nonspeech
    }


def evaluate_prompt_on_samples(model, processor, ids_A, ids_B, ids_C, ids_D, samples,
                                system_prompt, user_prompt, temperature=1.0):
    """
    Evaluate a prompt on a set of samples.
    Returns accuracy.
    """
    correct = 0
    total = len(samples)
    errors = []

    for sample in tqdm(samples, desc="Evaluating prompt", leave=False):
        try:
            result = evaluate_sample_logits(
                model, processor,
                sample['audio_path'],
                ids_A, ids_B, ids_C, ids_D,
                system_prompt, user_prompt,
                temperature
            )

            if result['prediction'] == sample['ground_truth_token']:
                correct += 1
        except Exception as e:
            error_msg = f"{sample['audio_path']}: {str(e)[:50]}"
            errors.append(error_msg)
            continue

    # Print error summary if there were errors
    if errors:
        print(f"\n⚠️  {len(errors)}/{total} samples failed")
        for err in errors[:3]:  # Show first 3 errors
            print(f"  - {err}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more")

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
    # All templates use 4-option format: A=Speech, B=Music, C=Noise/silence, D=Other
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


def opro_optimize(model, processor, train_df, ids_A, ids_B, ids_C, ids_D,
                  num_iterations=20, samples_per_iter=10,
                  num_candidates=8, temperature=1.0):
    """
    OPRO optimization loop.

    Args:
        model: Fine-tuned frozen model
        processor: Qwen2Audio processor
        train_df: DataFrame with training samples
        ids_A, ids_B, ids_C, ids_D: Token IDs for A/B/C/D options
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
        # Map SPEECH/NONSPEECH to A/B tokens
        # A = SPEECH (Human speech), B = NONSPEECH (mapped from B/C/D options)
        ground_truth = row['ground_truth']
        ground_truth_token = 'A' if ground_truth == 'SPEECH' else 'B'

        samples.append({
            'audio_path': row['audio_path'],
            'ground_truth_token': ground_truth_token
        })

    prompt_history = []  # List of (prompt, accuracy) tuples

    # Initialize with baseline prompt (from prompts/prompt_base.txt)
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

            accuracy = evaluate_prompt_on_samples(
                model, processor, ids_A, ids_B, ids_C, ids_D,
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
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--no_lora', action='store_true',
                        help='Use base model without LoRA')
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

    # Validate args
    if not args.no_lora and args.checkpoint is None:
        parser.error("--checkpoint is required unless --no_lora is specified")

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPRO: Prompt Optimization on Frozen Model")
    print("=" * 80)
    print(f"\nModel: {'BASE (no LoRA)' if args.no_lora else args.checkpoint}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Output: {args.output_dir}")

    # Load model
    if args.no_lora:
        print(f"\nLoading BASE model...")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )

        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        print(f"\nLoading fine-tuned model from {args.checkpoint}...")
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

    # Get token IDs for A/B/C/D options
    ids_A, ids_B, ids_C, ids_D = get_abcd_token_ids(processor.tokenizer)
    print(f"Token IDs: A={ids_A}, B={ids_B}, C={ids_C}, D={ids_D}")

    # Load data
    print(f"\nLoading training data...")
    train_df = pd.read_csv(args.train_csv)
    print(f"Loaded {len(train_df)} samples")
    print(f"  SPEECH:    {(train_df['ground_truth'] == 'SPEECH').sum()}")
    print(f"  NONSPEECH: {(train_df['ground_truth'] == 'NONSPEECH').sum()}")

    # Run OPRO
    best_prompt, best_accuracy, history = opro_optimize(
        model, processor, train_df, ids_A, ids_B, ids_C, ids_D,
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
