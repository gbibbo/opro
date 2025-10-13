#!/usr/bin/env python3
"""
Sprint 9: Main runner for OPRO optimization.

Integrates OPRO optimizer with prompt evaluator.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.models import Qwen2AudioClassifier
from opro_optimizer import OPROOptimizer
from evaluate_prompt import evaluate_prompt


def create_evaluator_fn(
    manifest_path: Path,
    model: Qwen2AudioClassifier,
    split: str = "dev",
    seed: int = 42,
):
    """
    Create evaluator function for OPRO optimizer.

    Args:
        manifest_path: Path to manifest
        model: Pre-loaded Qwen2AudioClassifier instance
        split: Split to evaluate on
        seed: Random seed

    Returns:
        Evaluator function that takes prompt and returns (ba_clip, ba_cond, metrics)
    """

    def evaluator(prompt: str) -> Tuple[float, float, dict]:
        """Evaluate prompt on dev set."""
        ba_clip, ba_cond, metrics = evaluate_prompt(
            prompt=prompt,
            manifest_path=manifest_path,
            model=model,  # Reuse loaded model
            split=split,
            seed=seed,
            save_predictions=False,  # Don't save intermediate results
        )
        return ba_clip, ba_cond, metrics

    return evaluator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sprint 9: Run OPRO optimization")

    # OPRO settings
    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM for generating candidates",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for LLM",
    )
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
        "--model_name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/sprint9_opro"),
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

    print(f"\n{'='*60}")
    print("OPRO OPTIMIZATION SETUP")
    print(f"{'='*60}")
    print(f"Optimizer LLM: {args.optimizer_llm}")
    print(f"Evaluator model: {args.model_name}")
    print(f"Split: {args.split}")
    print(f"Iterations: {args.n_iterations}")
    print(f"Candidates/iter: {args.candidates_per_iter}")
    print(f"Top-k: {args.top_k}")
    print(f"Output: {args.output_dir}")

    # Load model ONCE (reuse for all evaluations)
    print(f"\nLoading Qwen2-Audio model...")
    model = Qwen2AudioClassifier(
        model_name=args.model_name,
        device=args.device,
        load_in_4bit=True,
    )

    # Create evaluator function
    evaluator_fn = create_evaluator_fn(
        manifest_path=args.manifest,
        model=model,
        split=args.split,
        seed=args.seed,
    )

    # Initialize OPRO optimizer
    optimizer = OPROOptimizer(
        optimizer_llm=args.optimizer_llm,
        api_key=args.api_key,
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
    print(f"  Baseline BA_clip: {optimizer.memory[0].ba_clip if optimizer.baseline_reward else 'N/A':.3f}")
    print(f"  Improvement: +{best_prompt.ba_clip - optimizer.memory[0].ba_clip:.3f}")
    print(f"\nResults saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
