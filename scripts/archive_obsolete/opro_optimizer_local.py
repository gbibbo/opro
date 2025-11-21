#!/usr/bin/env python3
"""
Sprint 9: OPRO optimizer with LOCAL LLM support.

Uses transformers library to run LLMs locally (e.g., Qwen2.5, Llama 3, etc.)
No API keys required - runs entirely on your GPU.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


class LocalLLMGenerator:
    """Local LLM for generating prompt candidates."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
    ):
        """
        Initialize local LLM.

        Args:
            model_name: HuggingFace model name
            device: Device to use
            load_in_4bit: Use 4-bit quantization
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading local LLM: {model_name}...")
        print(f"  Device: {device}")
        print(f"  4-bit quantization: {load_in_4bit}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        model_kwargs = {"torch_dtype": torch.float16}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = device if device == "cuda" else None

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if device == "cpu" and not load_in_4bit:
            self.model = self.model.to(device)

        self.model.eval()

        print("Local LLM loaded successfully!")

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Format as chat if model supports it
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        generated_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return generated_text


class OPROOptimizerLocal:
    """
    OPRO optimizer using LOCAL LLM.

    Uses transformers to run LLM locally on GPU.
    No API keys required.
    """

    def __init__(
        self,
        optimizer_llm: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        top_k: int = 10,
        candidates_per_iter: int = 3,
        reward_weights: dict = None,
        seed: int = 42,
    ):
        """
        Initialize OPRO optimizer with local LLM.

        Args:
            optimizer_llm: HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct")
            device: Device to use
            load_in_4bit: Use 4-bit quantization
            top_k: Number of best prompts to keep in memory
            candidates_per_iter: Number of candidates to generate per iteration
            reward_weights: Reward function weights
            seed: Random seed
        """
        self.optimizer_llm = optimizer_llm
        self.top_k = top_k
        self.candidates_per_iter = candidates_per_iter
        self.seed = seed

        # Default reward weights
        if reward_weights is None:
            reward_weights = {
                "ba_clip": 1.0,
                "ba_cond": 0.25,
                "length_penalty": 0.05,
            }
        self.reward_weights = reward_weights

        # Initialize local LLM
        self.llm = LocalLLMGenerator(
            model_name=optimizer_llm,
            device=device,
            load_in_4bit=load_in_4bit,
        )

        # Top-k memory
        self.memory: List[PromptCandidate] = []
        self.history: List[PromptCandidate] = []

        # Baseline prompt
        self.baseline_prompt = (
            "<|audio_bos|><|AUDIO|><|audio_eos|>Does this audio contain human speech?\n"
            "Reply with ONLY one word: SPEECH or NON-SPEECH."
        )
        self.baseline_reward = None

        print(f"\nOPRO Optimizer (LOCAL) initialized:")
        print(f"  LLM: {optimizer_llm}")
        print(f"  Device: {device}")
        print(f"  Top-k: {top_k}")
        print(f"  Candidates/iter: {candidates_per_iter}")
        print(f"  Reward weights: {reward_weights}")
        print(f"  Seed: {seed}")

    def compute_reward(self, ba_clip: float, ba_conditions: float, prompt_length: int) -> float:
        """Compute reward for a prompt."""
        reward = (
            self.reward_weights["ba_clip"] * ba_clip
            + self.reward_weights["ba_cond"] * ba_conditions
            - self.reward_weights["length_penalty"] * (prompt_length / 100.0)
        )
        return reward

    def build_meta_prompt(self, iteration: int) -> str:
        """Build meta-prompt for LLM."""
        sorted_memory = sorted(self.memory, key=lambda x: x.reward, reverse=True)

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
        """Use local LLM to generate new prompt candidates."""
        meta_prompt = self.build_meta_prompt(iteration)

        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Generating {self.candidates_per_iter} candidates...")
        print(f"{'='*60}")

        # Generate from local LLM
        llm_output = self.llm.generate(meta_prompt)

        # Parse candidates
        candidates = self._parse_candidates(llm_output)

        print(f"Generated {len(candidates)} candidates:")
        for i, prompt in enumerate(candidates, 1):
            print(f"  {i}. {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        return candidates

    def _parse_candidates(self, llm_output: str) -> List[str]:
        """Parse prompt candidates from LLM output."""
        candidates = []
        lines = llm_output.strip().split("\n")

        for line in lines:
            if "PROMPT_" in line and ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    prompt = parts[1].strip()
                    prompt = prompt.strip('"').strip("'")
                    if prompt and len(prompt) > 10:
                        candidates.append(prompt)

        # Fallback parsing
        if len(candidates) == 0:
            print("Warning: Failed to parse structured output, attempting fallback...")
            chunks = llm_output.split("\n\n")
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk and len(chunk) > 10 and "PROMPT" not in chunk:
                    candidates.append(chunk)
                if len(candidates) >= self.candidates_per_iter:
                    break

        return candidates[: self.candidates_per_iter]

    def update_memory(self, candidate: PromptCandidate):
        """Add candidate to memory and history."""
        self.history.append(candidate)
        self.memory.append(candidate)
        self.memory = sorted(self.memory, key=lambda x: x.reward, reverse=True)[: self.top_k]
        print(f"Memory updated: {len(self.memory)} prompts, best reward={self.memory[0].reward:.4f}")

    def save_state(self, output_dir: Path):
        """Save optimizer state."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save history
        history_path = output_dir / "opro_prompts.jsonl"
        with open(history_path, "w") as f:
            for candidate in self.history:
                f.write(json.dumps(asdict(candidate)) + "\n")

        # Save memory
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

        # Save reward history
        rewards = [c.reward for c in self.history]
        iterations = [c.iteration for c in self.history]
        history_summary = {
            "iterations": iterations,
            "rewards": rewards,
            "best_reward_per_iteration": [],
        }

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

    def run_optimization(
        self,
        evaluator_fn,
        n_iterations: int = 30,
        early_stopping_patience: int = 5,
        output_dir: Path = None,
    ) -> PromptCandidate:
        """Run OPRO optimization loop."""
        print(f"\n{'='*60}")
        print("STARTING OPRO OPTIMIZATION (LOCAL LLM)")
        print(f"{'='*60}")
        print(f"Iterations: {n_iterations}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Output dir: {output_dir}")

        # Evaluate baseline
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
                print(f"\nNEW BEST REWARD: {current_best_reward:.4f} (+{improvement:.4f})")
                best_reward = current_best_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(f"\nNo improvement (patience: {no_improvement_count}/{early_stopping_patience})")

            # Save state
            if output_dir:
                self.save_state(output_dir)

            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                print(f"\nEarly stopping: No improvement for {early_stopping_patience} iterations")
                break

        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total iterations: {iteration}")
        print(f"Best reward: {self.memory[0].reward:.4f}")
        print(f"Best BA_clip: {self.memory[0].ba_clip:.3f}")
        print(f"Best prompt: {self.memory[0].prompt}")

        return self.memory[0]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sprint 9: OPRO with LOCAL LLM")
    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model for prompt generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for optimizer LLM",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization for optimizer LLM",
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

    # Initialize optimizer with local LLM
    optimizer = OPROOptimizerLocal(
        optimizer_llm=args.optimizer_llm,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        top_k=args.top_k,
        candidates_per_iter=args.candidates_per_iter,
        seed=args.seed,
    )

    # Placeholder evaluator
    def evaluator_fn(prompt: str) -> Tuple[float, float, dict]:
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

    print(f"\nBest prompt: {best_prompt.prompt}")
    print(f"Reward: {best_prompt.reward:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
