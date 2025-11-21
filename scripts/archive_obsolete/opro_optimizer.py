#!/usr/bin/env python3
"""
Sprint 9: OPRO (Optimization by PROmpting) for prompt optimization.

Reference: Yang et al. (2023). "Large Language Models as Optimizers."
arXiv:2309.03409

Algorithm:
1. Maintain top-k memory of (prompt, reward) pairs
2. Each iteration: LLM generates N new candidates from meta-prompt
3. Evaluate each candidate on dev set
4. Update top-k memory
5. Repeat until convergence or max iterations
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from anthropic import Anthropic
from openai import OpenAI

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class PromptCandidate:
    """A candidate prompt with its evaluation results."""

    prompt: str
    reward: float
    ba_clip: float
    ba_conditions: float
    prompt_length: int
    iteration: int
    timestamp: float
    metrics: dict


class OPROOptimizer:
    """
    OPRO optimizer for prompt optimization.

    Uses an LLM (Claude or GPT-4) to generate prompt candidates,
    evaluates them on dev set, and maintains top-k memory.
    """

    def __init__(
        self,
        optimizer_llm: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        top_k: int = 10,
        candidates_per_iter: int = 3,
        reward_weights: dict = None,
        seed: int = 42,
    ):
        """
        Initialize OPRO optimizer.

        Args:
            optimizer_llm: LLM for generating candidates ("claude-*" or "gpt-4*")
            api_key: API key (if None, reads from env)
            top_k: Number of best prompts to keep in memory
            candidates_per_iter: Number of candidates to generate per iteration
            reward_weights: Reward function weights {"ba_clip": 1.0, "ba_cond": 0.25, "length": 0.05}
            seed: Random seed
        """
        self.optimizer_llm = optimizer_llm
        self.top_k = top_k
        self.candidates_per_iter = candidates_per_iter
        self.seed = seed

        # Default reward weights
        if reward_weights is None:
            reward_weights = {
                "ba_clip": 1.0,  # Primary: clip-level balanced accuracy
                "ba_cond": 0.25,  # Secondary: condition-averaged BA
                "length_penalty": 0.05,  # Tertiary: penalize long prompts
            }
        self.reward_weights = reward_weights

        # Initialize LLM client
        if "claude" in optimizer_llm.lower():
            self.llm_client = Anthropic(api_key=api_key)
            self.llm_provider = "anthropic"
        elif "gpt" in optimizer_llm.lower():
            self.llm_client = OpenAI(api_key=api_key)
            self.llm_provider = "openai"
        else:
            raise ValueError(f"Unsupported LLM: {optimizer_llm}")

        # Top-k memory (list of PromptCandidate objects)
        self.memory: List[PromptCandidate] = []

        # Full history (all evaluated prompts)
        self.history: List[PromptCandidate] = []

        # Baseline prompt and reward
        self.baseline_prompt = (
            "<|audio_bos|><|AUDIO|><|audio_eos|>Does this audio contain human speech?\n"
            "Reply with ONLY one word: SPEECH or NON-SPEECH."
        )
        self.baseline_reward = None  # Set after first evaluation

        print(f"OPRO Optimizer initialized:")
        print(f"  LLM: {optimizer_llm}")
        print(f"  Top-k: {top_k}")
        print(f"  Candidates/iter: {candidates_per_iter}")
        print(f"  Reward weights: {reward_weights}")
        print(f"  Seed: {seed}")

    def compute_reward(self, ba_clip: float, ba_conditions: float, prompt_length: int) -> float:
        """
        Compute reward for a prompt based on performance metrics.

        R = BA_clip + α × BA_conditions - β × len(prompt)/100

        Args:
            ba_clip: Clip-level balanced accuracy
            ba_conditions: Macro-average balanced accuracy across conditions
            prompt_length: Character count of prompt

        Returns:
            Reward value (higher is better)
        """
        reward = (
            self.reward_weights["ba_clip"] * ba_clip
            + self.reward_weights["ba_cond"] * ba_conditions
            - self.reward_weights["length_penalty"] * (prompt_length / 100.0)
        )
        return reward

    def build_meta_prompt(self, iteration: int) -> str:
        """
        Build meta-prompt for LLM to generate new candidates.

        Includes:
        - Task description
        - Objective and constraints
        - Top-k previous prompts with rewards
        - Instructions for generating new candidates

        Args:
            iteration: Current iteration number

        Returns:
            Meta-prompt string
        """
        # Sort memory by reward (descending)
        sorted_memory = sorted(self.memory, key=lambda x: x.reward, reverse=True)

        # Build top-k history string
        history_str = ""
        for i, candidate in enumerate(sorted_memory[: self.top_k], 1):
            history_str += f"\n{i}. Reward={candidate.reward:.4f} | BA_clip={candidate.ba_clip:.3f} | BA_cond={candidate.ba_conditions:.3f}\n"
            history_str += f"   Prompt: {candidate.prompt}\n"

        meta_prompt = f"""TASK: Optimize prompts for audio classification (Qwen2-Audio-7B-Instruct).
The model receives audio and must classify it as SPEECH or NON-SPEECH.

OBJECTIVE: Maximize performance on psychoacoustic degradations:
- Short durations (20-200ms clips)
- Low SNR (-10 to 0 dB, noise masked speech)
- Band-pass filtered audio (telephony, low-pass, high-pass)
- Reverberant audio (T60: 0-1.5s)

REWARD FUNCTION:
R = BA_clip + 0.25 × BA_conditions - 0.05 × len(prompt)/100
where:
- BA_clip: Balanced accuracy at clip level (primary metric, range 0-1)
- BA_conditions: Macro-average BA across all psychoacoustic conditions
- len(prompt): Character count (penalty for verbosity)

BASELINE PROMPT:
Prompt: {self.baseline_prompt}
Reward: {self.baseline_reward:.4f if self.baseline_reward else 'evaluating...'}

CURRENT ITERATION: {iteration}

TOP-{self.top_k} PROMPTS:{history_str}

INSTRUCTIONS:
Generate {self.candidates_per_iter} NEW prompt candidates that:
1. Are clear and concise (target <150 chars, absolute max 300 chars)
2. Encourage robust detection on SHORT and NOISY clips
3. Use simple, direct language (model is instruction-tuned)
4. Build on insights from top prompts above
5. Explore semantic variations: question style, command style, description style
6. Consider emphasizing: brevity detection, noise robustness, voice/speech keywords

CONSTRAINTS:
- Each prompt must be COMPLETE and STANDALONE (no placeholders)
- Must include instruction to respond with ONLY "SPEECH" or "NON-SPEECH" (or similar binary format)
- Avoid overly complex or multi-step instructions
- NO audio template markers needed (system handles that)

OUTPUT FORMAT (exactly {self.candidates_per_iter} prompts, one per line):
PROMPT_1: <your complete prompt here>
PROMPT_2: <your complete prompt here>
PROMPT_3: <your complete prompt here>
{f"PROMPT_{self.candidates_per_iter}: <your complete prompt here>" if self.candidates_per_iter > 3 else ""}

Generate the prompts now:"""

        return meta_prompt

    def generate_candidates(self, iteration: int) -> List[str]:
        """
        Use LLM to generate new prompt candidates.

        Args:
            iteration: Current iteration number

        Returns:
            List of candidate prompts
        """
        meta_prompt = self.build_meta_prompt(iteration)

        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Generating {self.candidates_per_iter} candidates...")
        print(f"{'='*60}")

        # Call LLM
        if self.llm_provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.optimizer_llm,
                max_tokens=2000,
                temperature=0.7,  # Some creativity for exploration
                messages=[{"role": "user", "content": meta_prompt}],
            )
            llm_output = response.content[0].text

        elif self.llm_provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.optimizer_llm,
                messages=[{"role": "user", "content": meta_prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            llm_output = response.choices[0].message.content

        # Parse candidates from LLM output
        candidates = self._parse_candidates(llm_output)

        print(f"Generated {len(candidates)} candidates:")
        for i, prompt in enumerate(candidates, 1):
            print(f"  {i}. {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        return candidates

    def _parse_candidates(self, llm_output: str) -> List[str]:
        """
        Parse prompt candidates from LLM output.

        Expected format:
        PROMPT_1: <text>
        PROMPT_2: <text>
        ...

        Args:
            llm_output: Raw LLM response

        Returns:
            List of parsed prompts
        """
        candidates = []
        lines = llm_output.strip().split("\n")

        for line in lines:
            # Look for "PROMPT_N:" prefix
            if "PROMPT_" in line and ":" in line:
                # Extract everything after the colon
                parts = line.split(":", 1)
                if len(parts) == 2:
                    prompt = parts[1].strip()
                    # Remove quotes if present
                    prompt = prompt.strip('"').strip("'")
                    if prompt and len(prompt) > 10:  # Sanity check
                        candidates.append(prompt)

        # Fallback: if parsing failed, try to extract any meaningful text
        if len(candidates) == 0:
            print("Warning: Failed to parse structured output, attempting fallback...")
            # Split by common delimiters and take first N non-empty chunks
            chunks = llm_output.split("\n\n")
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk and len(chunk) > 10 and "PROMPT" not in chunk:
                    candidates.append(chunk)
                if len(candidates) >= self.candidates_per_iter:
                    break

        return candidates[: self.candidates_per_iter]

    def update_memory(self, candidate: PromptCandidate):
        """
        Add candidate to memory and history, maintain top-k.

        Args:
            candidate: PromptCandidate to add
        """
        # Add to full history
        self.history.append(candidate)

        # Add to memory
        self.memory.append(candidate)

        # Sort by reward and keep top-k
        self.memory = sorted(self.memory, key=lambda x: x.reward, reverse=True)[: self.top_k]

        print(f"Memory updated: {len(self.memory)} prompts, best reward={self.memory[0].reward:.4f}")

    def save_state(self, output_dir: Path):
        """
        Save optimizer state to disk.

        Args:
            output_dir: Directory to save state
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full history as JSONL
        history_path = output_dir / "opro_prompts.jsonl"
        with open(history_path, "w") as f:
            for candidate in self.history:
                f.write(json.dumps(asdict(candidate)) + "\n")

        # Save top-k memory
        memory_path = output_dir / "opro_memory.json"
        with open(memory_path, "w") as f:
            json.dump([asdict(c) for c in self.memory], f, indent=2)

        # Save best prompt
        if len(self.memory) > 0:
            best_prompt_path = output_dir / "best_prompt.txt"
            with open(best_prompt_path, "w") as f:
                f.write(self.memory[0].prompt)

            best_metrics_path = output_dir / "best_metrics.json"
            with open(best_metrics_path, "w") as f:
                json.dump(asdict(self.memory[0]), f, indent=2)

        # Save reward history (for plotting)
        rewards = [c.reward for c in self.history]
        iterations = [c.iteration for c in self.history]
        history_summary = {
            "iterations": iterations,
            "rewards": rewards,
            "best_reward_per_iteration": [],
        }

        # Compute best reward up to each iteration
        best_so_far = float("-inf")
        for it in sorted(set(iterations)):
            iter_candidates = [c for c in self.history if c.iteration == it]
            max_reward = max([c.reward for c in iter_candidates])
            best_so_far = max(best_so_far, max_reward)
            history_summary["best_reward_per_iteration"].append(best_so_far)

        history_summary_path = output_dir / "opro_history.json"
        with open(history_summary_path, "w") as f:
            json.dump(history_summary, f, indent=2)

        print(f"\nSaved state to: {output_dir}")
        print(f"  - Full history: {history_path}")
        print(f"  - Top-k memory: {memory_path}")
        print(f"  - Best prompt: {best_prompt_path}")

    def run_optimization(
        self,
        evaluator_fn,
        n_iterations: int = 30,
        early_stopping_patience: int = 5,
        output_dir: Path = None,
    ) -> PromptCandidate:
        """
        Run OPRO optimization loop.

        Args:
            evaluator_fn: Function that takes prompt and returns (ba_clip, ba_conditions, metrics)
            n_iterations: Maximum number of iterations
            early_stopping_patience: Stop if no improvement for N iterations
            output_dir: Directory to save results

        Returns:
            Best PromptCandidate found
        """
        print(f"\n{'='*60}")
        print("STARTING OPRO OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Iterations: {n_iterations}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Output dir: {output_dir}")

        # Evaluate baseline if not done yet
        if self.baseline_reward is None:
            print("\nEvaluating baseline prompt...")
            ba_clip, ba_cond, metrics = evaluator_fn(self.baseline_prompt)
            self.baseline_reward = self.compute_reward(ba_clip, ba_cond, len(self.baseline_prompt))

            baseline_candidate = PromptCandidate(
                prompt=self.baseline_prompt,
                reward=self.baseline_reward,
                ba_clip=ba_clip,
                ba_conditions=ba_cond,
                prompt_length=len(self.baseline_prompt),
                iteration=0,
                timestamp=time.time(),
                metrics=metrics,
            )
            self.update_memory(baseline_candidate)
            print(f"Baseline reward: {self.baseline_reward:.4f}")

        best_reward = self.memory[0].reward
        no_improvement_count = 0

        for iteration in range(1, n_iterations + 1):
            # Generate candidates
            candidates = self.generate_candidates(iteration)

            # Evaluate each candidate
            for i, prompt in enumerate(candidates, 1):
                print(f"\nEvaluating candidate {i}/{len(candidates)}...")
                print(f"Prompt: {prompt[:150]}{'...' if len(prompt) > 150 else ''}")

                ba_clip, ba_cond, metrics = evaluator_fn(prompt)
                reward = self.compute_reward(ba_clip, ba_cond, len(prompt))

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

                self.update_memory(candidate)

                print(f"Results: BA_clip={ba_clip:.3f}, BA_cond={ba_cond:.3f}, Reward={reward:.4f}")

            # Check for improvement
            current_best_reward = self.memory[0].reward
            if current_best_reward > best_reward:
                improvement = current_best_reward - best_reward
                print(f"\n🎉 NEW BEST REWARD: {current_best_reward:.4f} (+{improvement:.4f})")
                best_reward = current_best_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(f"\nNo improvement (patience: {no_improvement_count}/{early_stopping_patience})")

            # Save state after each iteration
            if output_dir:
                self.save_state(output_dir)

            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                print(f"\n⏹️  Early stopping: No improvement for {early_stopping_patience} iterations")
                break

        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total iterations: {iteration}")
        print(f"Best reward: {self.memory[0].reward:.4f}")
        print(f"Best BA_clip: {self.memory[0].ba_clip:.3f}")
        print(f"Best BA_cond: {self.memory[0].ba_conditions:.3f}")
        print(f"Best prompt: {self.memory[0].prompt}")

        return self.memory[0]


def main():
    """Main entry point for running OPRO optimization."""
    parser = argparse.ArgumentParser(description="Sprint 9: OPRO prompt optimization")
    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="LLM for generating candidates (claude-* or gpt-4*)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for LLM (or set ANTHROPIC_API_KEY / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of best prompts to keep in memory",
    )
    parser.add_argument(
        "--candidates_per_iter",
        type=int,
        default=3,
        help="Number of candidates to generate per iteration",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=30,
        help="Maximum number of optimization iterations",
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=5,
        help="Stop if no improvement for N iterations",
    )
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

    # Initialize optimizer
    optimizer = OPROOptimizer(
        optimizer_llm=args.optimizer_llm,
        api_key=args.api_key,
        top_k=args.top_k,
        candidates_per_iter=args.candidates_per_iter,
        seed=args.seed,
    )

    # Define evaluator function (placeholder - will be implemented in evaluate_prompt.py)
    def evaluator_fn(prompt: str) -> Tuple[float, float, dict]:
        """
        Evaluate prompt on dev set.

        Returns:
            (ba_clip, ba_conditions, metrics)
        """
        # TODO: Import and call evaluate_prompt.py
        # For now, return dummy values
        print(f"  [Evaluator placeholder] Would evaluate: {prompt[:50]}...")
        import random

        ba_clip = random.uniform(0.5, 0.8)
        ba_cond = random.uniform(0.4, 0.7)
        metrics = {"dummy": True}
        return ba_clip, ba_cond, metrics

    # Run optimization
    best_prompt = optimizer.run_optimization(
        evaluator_fn=evaluator_fn,
        n_iterations=args.n_iterations,
        early_stopping_patience=args.early_stopping,
        output_dir=args.output_dir,
    )

    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    print(f"Best prompt: {best_prompt.prompt}")
    print(f"Reward: {best_prompt.reward:.4f}")
    print(f"BA_clip: {best_prompt.ba_clip:.3f}")
    print(f"BA_cond: {best_prompt.ba_conditions:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
