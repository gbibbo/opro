#!/usr/bin/env python3
"""
Sprint 9: OPRO Local optimizado para 8GB VRAM - VERSIÓN MEJORADA

Mejoras implementadas:
1. Sanitización de candidatos (bloquea tokens <|audio_*|>)
2. Validación estricta de prompts
3. Mejor gestión de memoria
4. Circuit breaker por candidato
5. Meta-prompt mejorado sin mostrar tokens especiales
"""

import argparse
import gc
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from qsm.models import Qwen2AudioClassifier
from opro_optimizer_local import LocalLLMGenerator
from evaluate_prompt import evaluate_prompt


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(0.5)  # Give GPU time to release


def sanitize_prompt(prompt: str) -> Tuple[str, bool]:
    """
    Sanitize and validate prompt candidate.

    Returns:
        (cleaned_prompt, is_valid)
    """
    # Remove any audio special tokens
    forbidden_tokens = [
        '<|audio_bos|>', '<|AUDIO|>', '<|audio_eos|>',
        '<|im_start|>', '<|im_end|>',
        '<audio>', '</audio>',
    ]

    cleaned = prompt.strip()

    for token in forbidden_tokens:
        if token in cleaned:
            print(f"      ⚠️  Rejected: Contains forbidden token '{token}'")
            return cleaned, False

    # Check length
    if len(cleaned) < 10:
        print(f"      ⚠️  Rejected: Too short ({len(cleaned)} chars)")
        return cleaned, False

    if len(cleaned) > 300:
        print(f"      ⚠️  Rejected: Too long ({len(cleaned)} chars)")
        return cleaned, False

    # Must contain SPEECH and NON-SPEECH keywords
    upper = cleaned.upper()
    if 'SPEECH' not in upper or 'NON-SPEECH' not in upper:
        if 'SPEECH' not in upper and 'NONSPEECH' not in upper:
            print(f"      ⚠️  Rejected: Missing SPEECH/NON-SPEECH keywords")
            return cleaned, False

    # Remove multiple spaces and newlines
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()

    return cleaned, True


def parse_and_sanitize_candidates(llm_output: str, n_expected: int) -> List[str]:
    """
    Parse candidates from LLM output and sanitize them.

    Returns:
        List of valid prompts
    """
    candidates_raw = []
    lines = llm_output.strip().split("\n")

    # Try structured parsing first
    for line in lines:
        if "PROMPT_" in line and ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                prompt = parts[1].strip().strip('"').strip("'")
                if prompt:
                    candidates_raw.append(prompt)

    # Fallback: split by double newline
    if len(candidates_raw) == 0:
        chunks = llm_output.split("\n\n")
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk and len(chunk) > 10 and "PROMPT" not in chunk:
                candidates_raw.append(chunk)

    # Sanitize all candidates
    candidates_clean = []
    print(f"\n  Parsing {len(candidates_raw)} raw candidates...")

    for i, prompt_raw in enumerate(candidates_raw, 1):
        print(f"    Candidate {i}: {prompt_raw[:60]}{'...' if len(prompt_raw) > 60 else ''}")
        cleaned, is_valid = sanitize_prompt(prompt_raw)

        if is_valid:
            candidates_clean.append(cleaned)
            print(f"      ✓ Valid")

        if len(candidates_clean) >= n_expected:
            break

    if len(candidates_clean) == 0:
        print(f"\n  ⚠️  WARNING: No valid candidates generated!")
        print(f"  Falling back to baseline variations...")
        # Fallback: generate simple variations
        candidates_clean = [
            "Does this audio contain human speech? Answer: SPEECH or NON-SPEECH.",
            "Is there speech in this audio? Reply: SPEECH or NON-SPEECH.",
            "Audio classification: SPEECH or NON-SPEECH?",
        ][:n_expected]

    return candidates_clean[:n_expected]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sprint 9: OPRO Local (8GB, FIXED)")

    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Modelo pequeño para generar prompts",
    )
    parser.add_argument(
        "--candidates_per_iter",
        type=int,
        default=3,
        help="Candidatos por iteración",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=5,
        help="Iteraciones máximas",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=3,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/conditions_final/conditions_manifest_split.parquet"),
        help="Manifest path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/sprint9_opro_laptop_fixed"),
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print("OPRO LOCAL - 8GB VRAM (VERSIÓN MEJORADA)")
    print(f"{'='*60}")
    print(f"GPU: RTX 4070 Laptop (8GB)")
    print(f"Optimizer LLM: {args.optimizer_llm}")
    print(f"Iteraciones: {args.n_iterations}")
    print(f"Candidatos/iter: {args.candidates_per_iter}")
    print(f"\nMEJORAS:")
    print(f"  ✓ Sanitización de tokens especiales")
    print(f"  ✓ Validación estricta de prompts")
    print(f"  ✓ Circuit breaker por candidato")
    print(f"  ✓ Mejor gestión de memoria")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    import pandas as pd
    import json
    from dataclasses import asdict
    from opro_optimizer_local import PromptCandidate

    manifest_df = pd.read_parquet(args.manifest)
    split_df = manifest_df[manifest_df["split"] == args.split].copy()
    print(f"\nDev set: {len(split_df)} variants")

    # Initialize tracking
    memory = []
    history = []
    top_k = 10

    reward_weights = {
        "ba_clip": 1.0,
        "ba_cond": 0.25,
        "length_penalty": 0.05,
    }

    def compute_reward(ba_clip: float, ba_cond: float, prompt_length: int) -> float:
        return (
            reward_weights["ba_clip"] * ba_clip
            + reward_weights["ba_cond"] * ba_cond
            - reward_weights["length_penalty"] * (prompt_length / 100.0)
        )

    # BASELINE: Sin tokens especiales, solo el texto del usuario
    baseline_user_prompt = "Does this audio contain human speech?\nReply with ONLY one word: SPEECH or NON-SPEECH."

    # ============================================
    # ITERATION 0: Evaluate baseline
    # ============================================
    print(f"\n{'='*60}")
    print("ITERATION 0: BASELINE")
    print(f"{'='*60}")

    print("\nLoading Qwen2-Audio (evaluator)...")
    evaluator = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda",
        load_in_4bit=True,
    )

    print("\nEvaluating baseline prompt...")
    print(f"User prompt: {baseline_user_prompt}")

    ba_clip, ba_cond, metrics = evaluate_prompt(
        prompt=baseline_user_prompt,
        manifest_path=args.manifest,
        model=evaluator,
        split=args.split,
        seed=args.seed,
        save_predictions=False,
    )

    baseline_reward = compute_reward(ba_clip, ba_cond, len(baseline_user_prompt))

    baseline_candidate = PromptCandidate(
        prompt=baseline_user_prompt,
        reward=baseline_reward,
        ba_clip=ba_clip,
        ba_conditions=ba_cond,
        prompt_length=len(baseline_user_prompt),
        iteration=0,
        timestamp=time.time(),
        metrics=metrics,
    )

    memory.append(baseline_candidate)
    history.append(baseline_candidate)

    print(f"\nBaseline results:")
    print(f"  BA_clip: {ba_clip:.3f}")
    print(f"  BA_cond: {ba_cond:.3f}")
    print(f"  Reward: {baseline_reward:.4f}")

    print("\nUnloading evaluator...")
    del evaluator
    clear_gpu_memory()

    # ============================================
    # OPTIMIZATION LOOP
    # ============================================
    best_reward = baseline_reward
    no_improvement_count = 0

    for iteration in range(1, args.n_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")

        # ----------------------------------------
        # PHASE 1: Generate candidates
        # ----------------------------------------
        print(f"\nPHASE 1: Generating {args.candidates_per_iter} candidates...")

        print(f"Loading optimizer LLM: {args.optimizer_llm}...")
        optimizer_llm = LocalLLMGenerator(
            model_name=args.optimizer_llm,
            device="cuda",
            load_in_4bit=True,
            temperature=0.7,
        )

        # Build meta-prompt (sin mostrar tokens especiales en ejemplos)
        sorted_memory = sorted(memory, key=lambda x: x.reward, reverse=True)
        history_str = ""
        for i, candidate in enumerate(sorted_memory[:min(5, top_k)], 1):
            # Mostrar solo el texto limpio del prompt
            clean_prompt = candidate.prompt.replace('<|audio_bos|><|AUDIO|><|audio_eos|>', '').strip()
            history_str += f"\n{i}. Reward={candidate.reward:.4f} | BA_clip={candidate.ba_clip:.3f}\n"
            history_str += f"   \"{clean_prompt}\"\n"

        meta_prompt = f"""TASK: Optimize text prompts for Qwen2-Audio speech classification.

OBJECTIVE: Maximize balanced accuracy on psychoacoustically degraded audio:
- Short durations (20-200ms)
- Low SNR (-10 to 0 dB, masked by noise)
- Band-pass filters (telephony, low/high-pass)
- Reverberation (T60: 0-1.5s)

REWARD FUNCTION:
R = BA_clip + 0.25×BA_conditions - 0.05×len/100

CONSTRAINTS:
- Prompts must be plain text (NO special tokens, NO markup)
- Must instruct model to respond with "SPEECH" or "NON-SPEECH"
- Keep prompts concise (<150 characters)
- Focus on short/noisy clip detection

BASELINE:
Prompt: "{baseline_user_prompt}"
Reward: {baseline_reward:.4f}

TOP PROMPTS (iteration {iteration}):{history_str}

INSTRUCTIONS:
Generate {args.candidates_per_iter} NEW text prompts that improve on the baseline.
Each prompt should be a single clear question or instruction.

OUTPUT FORMAT (plain text prompts, one per line):
PROMPT_1: <your prompt here>
PROMPT_2: <your prompt here>
PROMPT_3: <your prompt here>

Generate now:"""

        print("Generating...")
        llm_output = optimizer_llm.generate(meta_prompt)

        # Parse and sanitize candidates
        candidates = parse_and_sanitize_candidates(llm_output, args.candidates_per_iter)

        print(f"\n  ✓ {len(candidates)} valid candidates after sanitization")
        for i, p in enumerate(candidates, 1):
            print(f"    {i}. {p[:70]}{'...' if len(p) > 70 else ''}")

        print("\nUnloading optimizer LLM...")
        del optimizer_llm
        clear_gpu_memory()

        # ----------------------------------------
        # PHASE 2: Evaluate candidates
        # ----------------------------------------
        print(f"\nPHASE 2: Evaluating {len(candidates)} candidates...")

        print("Loading Qwen2-Audio (evaluator)...")
        evaluator = Qwen2AudioClassifier(
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
            device="cuda",
            load_in_4bit=True,
        )

        # Evaluate each candidate with circuit breaker
        for i, prompt in enumerate(candidates, 1):
            print(f"\nEvaluating candidate {i}/{len(candidates)}...")
            print(f"  Prompt: \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")

            try:
                ba_clip, ba_cond, metrics = evaluate_prompt(
                    prompt=prompt,
                    manifest_path=args.manifest,
                    model=evaluator,
                    split=args.split,
                    seed=args.seed,
                    save_predictions=False,
                )

                reward = compute_reward(ba_clip, ba_cond, len(prompt))

                candidate = PromptCandidate(
                    prompt=prompt,
                    reward=reward,
                    ba_clip=ba_clip,
                    ba_conditions=ba_cond,
                    prompt_length=len(prompt),
                    iteration=iteration,
                    timestamp=time.time(),
                    metrics=metrics,
                )

                memory.append(candidate)
                history.append(candidate)
                memory = sorted(memory, key=lambda x: x.reward, reverse=True)[:top_k]

                print(f"  ✓ Results: BA_clip={ba_clip:.3f}, BA_cond={ba_cond:.3f}, Reward={reward:.4f}")

            except Exception as e:
                print(f"  ✗ ERROR evaluating candidate: {e}")
                print(f"  Skipping this candidate...")
                continue

        print("\nUnloading evaluator...")
        del evaluator
        clear_gpu_memory()

        # ----------------------------------------
        # Check improvement
        # ----------------------------------------
        current_best_reward = memory[0].reward
        if current_best_reward > best_reward:
            improvement = current_best_reward - best_reward
            print(f"\n🎉 NEW BEST REWARD: {current_best_reward:.4f} (+{improvement:.4f})")
            best_reward = current_best_reward
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"\nNo improvement (patience: {no_improvement_count}/{args.early_stopping})")

        # Save state
        history_path = args.output_dir / "opro_prompts.jsonl"
        with open(history_path, "w") as f:
            for c in history:
                f.write(json.dumps(asdict(c)) + "\n")

        best_path = args.output_dir / "best_prompt.txt"
        with open(best_path, "w") as f:
            f.write(memory[0].prompt)

        metrics_path = args.output_dir / "best_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(asdict(memory[0]), f, indent=2)

        # Early stopping
        if no_improvement_count >= args.early_stopping:
            print(f"\n⏹️  Early stopping: No improvement for {args.early_stopping} iterations")
            break

    # ============================================
    # FINAL SUMMARY
    # ============================================
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal iterations: {iteration}")
    print(f"\nBest prompt found:")
    print(f'  "{memory[0].prompt}"')
    print(f"\nMetrics:")
    print(f"  Reward: {memory[0].reward:.4f}")
    print(f"  BA_clip: {memory[0].ba_clip:.3f}")
    print(f"  BA_cond: {memory[0].ba_conditions:.3f}")
    print(f"  Length: {memory[0].prompt_length} chars")
    print(f"\nBaseline comparison:")
    print(f"  Baseline BA_clip: {baseline_candidate.ba_clip:.3f}")
    print(f"  OPRO BA_clip: {memory[0].ba_clip:.3f}")
    print(f"  Improvement: {memory[0].ba_clip - baseline_candidate.ba_clip:+.3f}")
    print(f"\nResults saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
