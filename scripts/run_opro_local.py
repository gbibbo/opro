#!/usr/bin/env python3
"""
Sprint 9: Run OPRO with LOCAL LLM (no API keys needed).

Uses local transformer models for prompt generation.
Everything runs on your GPU - completely free!
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# Add src and scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from qsm.models import Qwen2AudioClassifier
from opro_optimizer_local import OPROOptimizerLocal
from evaluate_prompt import evaluate_prompt


def create_evaluator_fn(
    manifest_path: Path,
    model: Qwen2AudioClassifier,
    split: str = "dev",
    seed: int = 42,
):
    """Create evaluator function for OPRO optimizer."""

    def evaluator(prompt: str) -> Tuple[float, float, dict]:
        """Evaluate prompt on dev set."""
        ba_clip, ba_cond, metrics = evaluate_prompt(
            prompt=prompt,
            manifest_path=manifest_path,
            model=model,  # Reuse loaded model
            split=split,
            seed=seed,
            save_predictions=False,
        )
        return ba_clip, ba_cond, metrics

    return evaluator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sprint 9: OPRO with LOCAL LLM")

    # Optimizer LLM settings
    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model for prompt generation (e.g., Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--optimizer_device",
        type=str,
        default="cuda",
        help="Device for optimizer LLM (cuda or cpu)",
    )
    parser.add_argument(
        "--optimizer_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for optimizer LLM",
    )

    # OPRO settings
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k memory size",
    )
    parser.add_argument(
        "--candidates_per_iter",
        type=int,
        default=3,
        help="Candidates per iteration",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=30,
        help="Maximum iterations",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=5,
        help="Early stopping patience",
    )

    # Evaluator settings
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
        choices=["dev", "test"],
        help="Split to evaluate on",
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Qwen2-Audio model for evaluation",
    )
    parser.add_argument(
        "--evaluator_device",
        type=str,
        default="cuda",
        help="Device for evaluator model",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/sprint9_opro_local"),
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
    print("OPRO OPTIMIZATION WITH LOCAL LLM")
    print(f"{'='*60}")
    print(f"Optimizer LLM: {args.optimizer_llm}")
    print(f"Optimizer device: {args.optimizer_device}")
    print(f"Evaluator model: {args.evaluator_model}")
    print(f"Evaluator device: {args.evaluator_device}")
    print(f"Split: {args.split}")
    print(f"Iterations: {args.n_iterations}")
    print(f"Candidates/iter: {args.candidates_per_iter}")
    print(f"Output: {args.output_dir}")
    print(f"\nNOTE: Everything runs locally - no API keys needed!")

    # Load evaluator model ONCE
    print(f"\n{'='*60}")
    print("LOADING EVALUATOR MODEL (Qwen2-Audio)")
    print(f"{'='*60}")
    evaluator_model = Qwen2AudioClassifier(
        model_name=args.evaluator_model,
        device=args.evaluator_device,
        load_in_4bit=True,
    )

    # Create evaluator function
    evaluator_fn = create_evaluator_fn(
        manifest_path=args.manifest,
        model=evaluator_model,
        split=args.split,
        seed=args.seed,
    )

    # Initialize OPRO optimizer with local LLM
    print(f"\n{'='*60}")
    print("LOADING OPTIMIZER LLM")
    print(f"{'='*60}")
    optimizer = OPROOptimizerLocal(
        optimizer_llm=args.optimizer_llm,
        device=args.optimizer_device,
        load_in_4bit=args.optimizer_4bit,
        top_k=args.top_k,
        candidates_per_iter=args.candidates_per_iter,
        seed=args.seed,
    )

    # Run optimization
    best_prompt = optimizer.run_optimization(
        evaluator_fn=evaluator_fn,
        n_iterations=args.n_iterations,
        early_stopping_patience=args.early_stopping,
        output_dir=args.output_dir,
    )

    # Final summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest prompt found:")
    print(f'  "{best_prompt.prompt}"')
    print(f"\nMetrics:")
    print(f"  Reward: {best_prompt.reward:.4f}")
    print(f"  BA_clip: {best_prompt.ba_clip:.3f}")
    print(f"  BA_conditions: {best_prompt.ba_conditions:.3f}")
    print(f"  Length: {best_prompt.prompt_length} chars")
    print(f"\nBaseline comparison:")
    baseline = [c for c in optimizer.memory if c.iteration == 0][0]
    print(f"  Baseline BA_clip: {baseline.ba_clip:.3f}")
    print(f"  Improvement: {best_prompt.ba_clip - baseline.ba_clip:+.3f}")
    print(f"\nResults saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
