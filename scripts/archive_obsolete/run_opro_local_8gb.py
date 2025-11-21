#!/usr/bin/env python3
"""
Sprint 9: OPRO Local optimizado para GPUs con 8GB VRAM.

Estrategia:
1. Carga el optimizador LLM (Llama 3.2-3B)
2. Genera los N candidatos
3. DESCARGA el optimizador
4. Carga el evaluador (Qwen2-Audio)
5. Evalúa los N candidatos
6. DESCARGA el evaluador
7. Repite

Es más lento pero funciona con 8GB VRAM.
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Tuple

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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sprint 9: OPRO Local (8GB VRAM optimized)")

    # Optimizer LLM settings
    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Modelo pequeño para generar prompts (3B recomendado para 8GB)",
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

    # Data settings
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

    # Output
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/sprint9_opro_local_8gb"),
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
    print("OPRO LOCAL - OPTIMIZADO PARA 8GB VRAM")
    print(f"{'='*60}")
    print(f"GPU: RTX 4070 Laptop (8GB)")
    print(f"Estrategia: Carga/descarga alternada de modelos")
    print(f"Optimizer LLM: {args.optimizer_llm}")
    print(f"Iteraciones: {args.n_iterations}")
    print(f"Candidatos/iter: {args.candidates_per_iter}")
    print(f"\nNOTA: Más lento pero funciona con 8GB VRAM")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    import pandas as pd
    manifest_df = pd.read_parquet(args.manifest)
    split_df = manifest_df[manifest_df["split"] == args.split].copy()

    print(f"\nDev set: {len(split_df)} variants")

    # Initialize tracking
    from dataclasses import asdict
    from opro_optimizer_local import PromptCandidate
    import time
    import json

    memory = []  # Top-k prompts
    history = []  # All evaluated prompts
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

    baseline_prompt = (
        "<|audio_bos|><|AUDIO|><|audio_eos|>Does this audio contain human speech?\n"
        "Reply with ONLY one word: SPEECH or NON-SPEECH."
    )

    # ============================================
    # ITERATION 0: Evaluate baseline
    # ============================================
    print(f"\n{'='*60}")
    print("ITERATION 0: BASELINE")
    print(f"{'='*60}")

    # Load evaluator
    print("\nLoading Qwen2-Audio (evaluator)...")
    evaluator = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda",
        load_in_4bit=True,
    )

    # Evaluate baseline
    print("\nEvaluating baseline prompt...")
    ba_clip, ba_cond, metrics = evaluate_prompt(
        prompt=baseline_prompt,
        manifest_path=args.manifest,
        model=evaluator,
        split=args.split,
        seed=args.seed,
        save_predictions=False,
    )

    baseline_reward = compute_reward(ba_clip, ba_cond, len(baseline_prompt))

    baseline_candidate = PromptCandidate(
        prompt=baseline_prompt,
        reward=baseline_reward,
        ba_clip=ba_clip,
        ba_conditions=ba_cond,
        prompt_length=len(baseline_prompt),
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

    # Unload evaluator
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

        # Load optimizer LLM
        print(f"Loading optimizer LLM: {args.optimizer_llm}...")
        optimizer_llm = LocalLLMGenerator(
            model_name=args.optimizer_llm,
            device="cuda",
            load_in_4bit=True,
            temperature=0.7,
        )

        # Build meta-prompt
        sorted_memory = sorted(memory, key=lambda x: x.reward, reverse=True)
        history_str = ""
        for i, candidate in enumerate(sorted_memory[:top_k], 1):
            history_str += f"\n{i}. Reward={candidate.reward:.4f} | BA_clip={candidate.ba_clip:.3f}\n"
            history_str += f"   Prompt: {candidate.prompt}\n"

        meta_prompt = f"""TASK: Optimize prompts for audio classification (Qwen2-Audio).
The model receives audio and must classify it as SPEECH or NON-SPEECH.

OBJECTIVE: Maximize performance on:
- Short durations (20-200ms)
- Low SNR (-10 to 0 dB)
- Band-pass filtered audio
- Reverberant audio

REWARD = BA_clip + 0.25×BA_cond - 0.05×len/100

BASELINE:
Prompt: {baseline_prompt}
Reward: {baseline_reward:.4f}

ITERATION: {iteration}

TOP PROMPTS:{history_str}

INSTRUCTIONS:
Generate {args.candidates_per_iter} NEW prompts that:
1. Are concise (<150 chars)
2. Focus on SHORT and NOISY clips
3. Use simple language
4. Include "SPEECH or NON-SPEECH" response format

FORMAT:
PROMPT_1: <your prompt>
PROMPT_2: <your prompt>
PROMPT_3: <your prompt>

Generate now:"""

        # Generate candidates
        print("Generating...")
        llm_output = optimizer_llm.generate(meta_prompt)

        # Parse candidates
        candidates = []
        for line in llm_output.strip().split("\n"):
            if "PROMPT_" in line and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    prompt = parts[1].strip().strip('"').strip("'")
                    if prompt and len(prompt) > 10:
                        candidates.append(prompt)

        candidates = candidates[:args.candidates_per_iter]

        print(f"\nGenerated {len(candidates)} candidates:")
        for i, p in enumerate(candidates, 1):
            print(f"  {i}. {p[:80]}{'...' if len(p) > 80 else ''}")

        # Unload optimizer
        print("\nUnloading optimizer LLM...")
        del optimizer_llm
        clear_gpu_memory()

        # ----------------------------------------
        # PHASE 2: Evaluate candidates
        # ----------------------------------------
        print(f"\nPHASE 2: Evaluating {len(candidates)} candidates...")

        # Load evaluator
        print("Loading Qwen2-Audio (evaluator)...")
        evaluator = Qwen2AudioClassifier(
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
            device="cuda",
            load_in_4bit=True,
        )

        # Evaluate each candidate
        for i, prompt in enumerate(candidates, 1):
            print(f"\nEvaluating candidate {i}/{len(candidates)}...")
            print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

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

            print(f"Results: BA_clip={ba_clip:.3f}, BA_cond={ba_cond:.3f}, Reward={reward:.4f}")

        # Unload evaluator
        print("\nUnloading evaluator...")
        del evaluator
        clear_gpu_memory()

        # ----------------------------------------
        # Check improvement
        # ----------------------------------------
        current_best_reward = memory[0].reward
        if current_best_reward > best_reward:
            improvement = current_best_reward - best_reward
            print(f"\nNEW BEST REWARD: {current_best_reward:.4f} (+{improvement:.4f})")
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

        # Early stopping
        if no_improvement_count >= args.early_stopping:
            print(f"\nEarly stopping: No improvement for {args.early_stopping} iterations")
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
    print(f"\nBaseline comparison:")
    print(f"  Baseline BA_clip: {baseline_candidate.ba_clip:.3f}")
    print(f"  OPRO BA_clip: {memory[0].ba_clip:.3f}")
    print(f"  Improvement: {memory[0].ba_clip - baseline_candidate.ba_clip:+.3f}")
    print(f"\nResults saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
