#!/usr/bin/env python3
"""
OPRO Classic: Prompt Optimization with Local LLM

This script restores the "classic" OPRO that was successful (>90% on dev set).

Key features:
- Uses LOCAL LLM (Qwen2.5 / Qwen2-Audio / Llama) to generate prompts
- Optimizes a composite reward: R = w_clip·BA_clip + w_cond·BA_cond - w_len·(len/100)
- Evaluates prompts with Qwen2-Audio-7B-Instruct (base or with LoRA)
- Reuses working code from:
  * opro_optimizer_local.py (core optimizer logic)
  * run_opro_local_8gb_fixed.py (sanitization, circuit breaker)
  * Qwen2AudioClassifier for evaluation

Usage:
    # Base model (no fine-tuning)
    python scripts/opro_classic_optimize.py \
        --manifest data/processed/conditions_final/conditions_manifest_split.parquet \
        --split dev \
        --output_dir results/opro_classic \
        --no_lora

    # With LoRA checkpoint
    python scripts/opro_classic_optimize.py \
        --manifest data/processed/conditions_final/conditions_manifest_split.parquet \
        --split dev \
        --output_dir results/opro_classic_lora \
        --checkpoint checkpoints/qwen_lora_seed42/final
"""

import argparse
import gc
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.qsm.models.qwen_audio import Qwen2AudioClassifier
from src.qsm.utils.normalize import normalize_to_binary

# ============================================================================
# Data Classes
# ============================================================================


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


# ============================================================================
# Local LLM for Prompt Generation
# ============================================================================


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


# ============================================================================
# Prompt Sanitization and Validation
# ============================================================================


def sanitize_prompt(prompt: str) -> tuple[str, bool]:
    """
    Sanitize and validate prompt candidate.

    Returns:
        (cleaned_prompt, is_valid)
    """
    # Remove any audio special tokens
    forbidden_tokens = [
        "<|audio_bos|>",
        "<|AUDIO|>",
        "<|audio_eos|>",
        "<|im_start|>",
        "<|im_end|>",
        "<audio>",
        "</audio>",
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
    if "SPEECH" not in upper or "NON-SPEECH" not in upper:
        if "SPEECH" not in upper and "NONSPEECH" not in upper:
            print("      ⚠️  Rejected: Missing SPEECH/NON-SPEECH keywords")
            return cleaned, False

    # Remove multiple spaces and newlines
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    return cleaned, True


# ============================================================================
# OPRO Classic Optimizer
# ============================================================================


class OPROClassicOptimizer:
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
        baseline_prompt: str = None,
        initial_prompts: list[str] = None,
        max_new_tokens: int = 2000,
        temperature: float = 0.7,
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
            baseline_prompt: Initial baseline prompt
            initial_prompts: Optional list of initial prompts to seed memory
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Top-k memory
        self.memory: list[PromptCandidate] = []
        self.history: list[PromptCandidate] = []

        # Baseline prompt
        if baseline_prompt is None:
            baseline_prompt = (
                "Does this audio contain human speech?\n"
                "Reply with ONLY one word: SPEECH or NON-SPEECH."
            )
        self.baseline_prompt = baseline_prompt
        self.baseline_reward = None

        # Initial prompts (if provided)
        self.initial_prompts = initial_prompts or []

        print("\nOPRO Classic Optimizer initialized:")
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
            # Show only clean text (no special tokens)
            clean_prompt = candidate.prompt.replace(
                "<|audio_bos|><|AUDIO|><|audio_eos|>", ""
            ).strip()
            history_str += f"\n{i}. Reward={candidate.reward:.4f} | BA_clip={candidate.ba_clip:.3f} | BA_cond={candidate.ba_conditions:.3f}\n"
            history_str += f'   "{clean_prompt}"\n'

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
Prompt: "{self.baseline_prompt}"
Reward: {f"{self.baseline_reward:.4f}" if self.baseline_reward is not None else "evaluating..."}

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
- Prompts must be plain text (NO special tokens, NO markup)

OUTPUT FORMAT (exactly {self.candidates_per_iter} prompts, one per line):
PROMPT_1: <your complete prompt here>
PROMPT_2: <your complete prompt here>
PROMPT_3: <your complete prompt here>
{f"PROMPT_{self.candidates_per_iter}: <your complete prompt here>" if self.candidates_per_iter > 3 else ""}

Generate the prompts now:"""

        return meta_prompt

    def generate_candidates(self, iteration: int) -> list[str]:
        """Use local LLM to generate new prompt candidates."""
        meta_prompt = self.build_meta_prompt(iteration)

        print(f"\n{'='*60}")
        print(f"Iteration {iteration}: Generating {self.candidates_per_iter} candidates...")
        print(f"{'='*60}")

        # Generate from local LLM
        llm_output = self.llm.generate(meta_prompt)

        # Parse and sanitize candidates
        candidates = self._parse_and_sanitize_candidates(llm_output)

        print(f"Generated {len(candidates)} valid candidates:")
        for i, prompt in enumerate(candidates, 1):
            print(f"  {i}. {prompt[:100]}{'...' if len(prompt) > 100 else ''}")

        return candidates

    def _parse_and_sanitize_candidates(self, llm_output: str) -> list[str]:
        """Parse candidates from LLM output and sanitize them."""
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
                print("      ✓ Valid")

            if len(candidates_clean) >= self.candidates_per_iter:
                break

        # Fallback if no valid candidates generated
        if len(candidates_clean) == 0:
            print("\n  ⚠️  WARNING: No valid candidates generated!")
            print("  Falling back to baseline variations...")
            candidates_clean = [
                "Does this audio contain human speech? Answer: SPEECH or NON-SPEECH.",
                "Is there speech in this audio? Reply: SPEECH or NON-SPEECH.",
                "Audio classification: SPEECH or NON-SPEECH?",
            ][: self.candidates_per_iter]

        return candidates_clean[: self.candidates_per_iter]

    def update_memory(self, candidate: PromptCandidate):
        """Add candidate to memory and history."""
        self.history.append(candidate)
        self.memory.append(candidate)
        self.memory = sorted(self.memory, key=lambda x: x.reward, reverse=True)[: self.top_k]
        print(
            f"Memory updated: {len(self.memory)} prompts, best reward={self.memory[0].reward:.4f}"
        )

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
        print("STARTING OPRO CLASSIC OPTIMIZATION")
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

        # Evaluate initial prompts if provided
        for i, prompt in enumerate(self.initial_prompts):
            print(f"\nEvaluating initial prompt {i+1}/{len(self.initial_prompts)}...")
            ba_clip, ba_cond, metrics = evaluator_fn(prompt)
            reward = self.compute_reward(ba_clip, ba_cond, len(prompt))

            candidate = PromptCandidate(
                prompt=prompt,
                reward=reward,
                ba_clip=ba_clip,
                ba_conditions=ba_cond,
                prompt_length=len(prompt),
                iteration=0,
                timestamp=time.time(),
                metrics=metrics,
            )
            self.update_memory(candidate)

        best_reward = self.memory[0].reward
        no_improvement_count = 0

        for iteration in range(1, n_iterations + 1):
            # Generate candidates
            candidates = self.generate_candidates(iteration)

            # Evaluate each candidate with circuit breaker
            for i, prompt in enumerate(candidates, 1):
                print(f"\nEvaluating candidate {i}/{len(candidates)}...")
                print(f"Prompt: {prompt[:150]}{'...' if len(prompt) > 150 else ''}")

                try:
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
                    print(
                        f"Results: BA_clip={ba_clip:.3f}, BA_cond={ba_cond:.3f}, Reward={reward:.4f}"
                    )

                except Exception as e:
                    print(f"  ✗ ERROR evaluating candidate: {e}")
                    print("  Skipping this candidate...")
                    continue

            # Check for improvement
            current_best_reward = self.memory[0].reward
            if current_best_reward > best_reward:
                improvement = current_best_reward - best_reward
                print(f"\nNEW BEST REWARD: {current_best_reward:.4f} (+{improvement:.4f})")
                best_reward = current_best_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                print(
                    f"\nNo improvement (patience: {no_improvement_count}/{early_stopping_patience})"
                )

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


# ============================================================================
# Evaluation Functions
# ============================================================================


def build_evaluator_from_args(args) -> Qwen2AudioClassifier:
    """Build evaluator model from command-line arguments."""
    print(f"\nLoading evaluator model: {args.evaluator_model_name}...")
    print(f"  Device: {args.evaluator_device}")
    print("  4-bit quantization: True")

    model = Qwen2AudioClassifier(
        model_name=args.evaluator_model_name,
        device=args.evaluator_device,
        load_in_4bit=True,
    )

    if not args.no_lora and args.checkpoint is not None:
        print(f"  Loading LoRA checkpoint: {args.checkpoint}")
        from peft import PeftModel

        model.model = PeftModel.from_pretrained(model.model, args.checkpoint)
        model.model.eval()
        print("  LoRA checkpoint loaded!")

    return model


def make_evaluator_fn(args, evaluator_model):
    """
    Create evaluator function that evaluates a prompt on the dev/test set.

    Args:
        args: Command-line arguments
        evaluator_model: Qwen2AudioClassifier instance

    Returns:
        Function that takes a prompt and returns (ba_clip, ba_cond, metrics)
    """
    # Load manifest
    manifest_df = pd.read_parquet(args.manifest)
    split_df = manifest_df[manifest_df["split"] == args.split].copy()

    print("\nDataset loaded:")
    print(f"  Manifest: {args.manifest}")
    print(f"  Split: {args.split}")
    print(f"  Samples: {len(split_df)}")

    def evaluator_fn(prompt: str) -> tuple[float, float, dict]:
        """
        Evaluate prompt on dataset.

        Returns:
            (ba_clip, ba_cond, metrics)
        """
        # Update model prompt
        evaluator_model.user_prompt = prompt

        # Evaluate on all samples
        results = []
        for _, row in tqdm(
            split_df.iterrows(), total=len(split_df), desc="  Evaluating", leave=False
        ):
            audio_path = row["audio_path"]
            ground_truth = row["label"]

            # Handle path resolution - convert Windows backslashes to forward slashes
            audio_path_str = str(audio_path).replace("\\", "/")
            audio_path = Path(audio_path_str)

            # If path is relative and doesn't exist, it's already correct
            # (manifest paths are relative to repo root)
            if not audio_path.exists():
                print(f"  WARNING: File not found: {audio_path}")
                continue

            try:
                result = evaluator_model.predict(str(audio_path.resolve()), return_scores=True)

                # Normalize prediction
                normalized_label, confidence = normalize_to_binary(
                    result.raw_output,
                    probs=result.probs,
                    mode="auto",
                    verbalizers=["SPEECH", "NONSPEECH"],
                )

                if normalized_label is None:
                    normalized_label = result.label

                is_correct = (normalized_label == ground_truth) if normalized_label else False

                # Create condition string from variant_type and corresponding value
                variant_type = row.get("variant_type", "unknown")
                if variant_type == "duration":
                    condition = f"dur_{row.get('duration_ms', 'unk')}ms"
                elif variant_type == "snr":
                    condition = f"snr_{row.get('snr_db', 'unk')}dB"
                elif variant_type == "band":
                    condition = f"filter_{row.get('band_filter', 'unk')}"
                elif variant_type == "rir":
                    condition = f"reverb_T60_{row.get('T60', 'unk')}s"
                else:
                    condition = "unknown"

                results.append(
                    {
                        "audio_path": str(audio_path),
                        "ground_truth": ground_truth,
                        "prediction": normalized_label,
                        "correct": is_correct,
                        "condition": condition,
                        "variant_type": variant_type,
                        # NUEVO: dimensiones psicoacústicas explícitas
                        "duration_ms": row.get("duration_ms", None),
                        "snr_db": row.get("snr_db", None),
                        "band_filter": row.get("band_filter", None),
                        "T60": row.get("T60", None),
                    }
                )

            except Exception as e:
                print(f"  Error processing {audio_path}: {e}")
                continue

        # Calculate metrics
        results_df = pd.DataFrame(results)

        # Empty safeguard
        if len(results_df) == 0:
            return 0.0, 0.0, {}

        # ------------------------------------------------------------------
        # 1) Balanced accuracy at clip level
        # ------------------------------------------------------------------
        speech_samples = results_df[results_df["ground_truth"] == "SPEECH"]
        nonspeech_samples = results_df[results_df["ground_truth"] == "NONSPEECH"]

        speech_acc = speech_samples["correct"].mean() if len(speech_samples) > 0 else 0.0
        nonspeech_acc = nonspeech_samples["correct"].mean() if len(nonspeech_samples) > 0 else 0.0
        ba_clip = (speech_acc + nonspeech_acc) / 2.0

        # ------------------------------------------------------------------
        # 2) Balanced accuracy por condición compuesta (string 'condition')
        # ------------------------------------------------------------------
        condition_bas = []
        condition_metrics = {}

        for condition in results_df["condition"].unique():
            cond_df = results_df[results_df["condition"] == condition]
            cond_speech = cond_df[cond_df["ground_truth"] == "SPEECH"]
            cond_nonspeech = cond_df[cond_df["ground_truth"] == "NONSPEECH"]

            cond_speech_acc = cond_speech["correct"].mean() if len(cond_speech) > 0 else 0.0
            cond_nonspeech_acc = (
                cond_nonspeech["correct"].mean() if len(cond_nonspeech) > 0 else 0.0
            )
            cond_ba = (cond_speech_acc + cond_nonspeech_acc) / 2.0

            condition_bas.append(cond_ba)
            condition_metrics[str(condition)] = {
                "ba": float(cond_ba),
                "speech_acc": float(cond_speech_acc),
                "nonspeech_acc": float(cond_nonspeech_acc),
                "n_samples": int(len(cond_df)),
            }

        ba_conditions = float(np.mean(condition_bas)) if len(condition_bas) > 0 else 0.0

        # ------------------------------------------------------------------
        # 3) Helper para BA por columna (duración, SNR, filtro, reverb)
        # ------------------------------------------------------------------
        def compute_ba_by_column(df: pd.DataFrame, col: str):
            if col not in df.columns:
                return 0.0, {}

            values = [v for v in df[col].unique() if pd.notna(v)]
            if not values:
                return 0.0, {}

            bas = []
            metrics_by_value = {}

            for v in values:
                sub = df[df[col] == v]
                speech = sub[sub["ground_truth"] == "SPEECH"]
                nonspeech = sub[sub["ground_truth"] == "NONSPEECH"]

                speech_acc_v = speech["correct"].mean() if len(speech) > 0 else 0.0
                nonspeech_acc_v = nonspeech["correct"].mean() if len(nonspeech) > 0 else 0.0
                ba_v = (speech_acc_v + nonspeech_acc_v) / 2.0

                bas.append(ba_v)
                metrics_by_value[str(v)] = {
                    "ba": float(ba_v),
                    "speech_acc": float(speech_acc_v),
                    "nonspeech_acc": float(nonspeech_acc_v),
                    "n_samples": int(len(sub)),
                }

            return float(np.mean(bas)), metrics_by_value

        # ------------------------------------------------------------------
        # 4) BA por dimensión psicoacústica
        # ------------------------------------------------------------------
        ba_duration, duration_metrics = compute_ba_by_column(results_df, "duration_ms")
        ba_snr, snr_metrics = compute_ba_by_column(results_df, "snr_db")
        ba_filter, filter_metrics = compute_ba_by_column(results_df, "band_filter")
        ba_reverb, reverb_metrics = compute_ba_by_column(results_df, "T60")

        # ------------------------------------------------------------------
        # 5) Empaquetar todas las métricas en un dict
        # ------------------------------------------------------------------
        metrics = {
            "ba_clip": float(ba_clip),
            "ba_conditions": float(ba_conditions),
            "ba_duration": float(ba_duration),
            "ba_snr": float(ba_snr),
            "ba_filter": float(ba_filter),
            "ba_reverb": float(ba_reverb),
            "speech_acc": float(speech_acc),
            "nonspeech_acc": float(nonspeech_acc),
            "n_samples": int(len(results_df)),
            "condition_metrics": condition_metrics,
            "duration_metrics": duration_metrics,
            "snr_metrics": snr_metrics,
            "filter_metrics": filter_metrics,
            "reverb_metrics": reverb_metrics,
        }

        return ba_clip, ba_conditions, metrics

    return evaluator_fn


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(0.5)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser("OPRO Classic Prompt Optimization")

    # Data
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest Parquet with psychoacoustic conditions.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Split to use in manifest (e.g., dev, test).",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for best prompts, history, and logs.",
    )

    # Evaluator (Qwen2-Audio)
    parser.add_argument(
        "--evaluator_model_name",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
    )
    parser.add_argument(
        "--evaluator_device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="If passed, do not load LoRA (use base model).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional LoRA checkpoint (same as in evaluate_with_generation.py).",
    )

    # OPRO / LLM optimizer
    parser.add_argument(
        "--optimizer_llm",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model for generating prompts.",
    )
    parser.add_argument(
        "--optimizer_device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--optimizer_load_in_4bit",
        action="store_true",
        default=True,
        help="Load optimizer LLM in 4-bit.",
    )
    parser.add_argument(
        "--optimizer_max_new_tokens",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--optimizer_temperature",
        type=float,
        default=0.7,
    )

    # OPRO configuration
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--candidates_per_iter",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--early_stopping",
        type=int,
        default=5,
    )

    # Reward weights
    parser.add_argument(
        "--reward_w_ba_clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--reward_w_ba_cond",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--reward_w_length_penalty",
        type=float,
        default=0.05,
    )

    # Baseline / initial prompts
    parser.add_argument(
        "--baseline_prompt",
        type=str,
        default="Does this audio contain human speech?\nReply with ONLY one word: SPEECH or NON-SPEECH.",
    )
    parser.add_argument(
        "--initial_prompts_json",
        type=str,
        default=None,
        help="Optional JSON file with initial prompts (list of strings).",
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print("OPRO CLASSIC OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Manifest: {args.manifest}")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")
    print(f"Evaluator: {args.evaluator_model_name}")
    print(f"Optimizer LLM: {args.optimizer_llm}")
    print(f"Seed: {args.seed}")

    # Load evaluator model
    evaluator_model = build_evaluator_from_args(args)

    # Prepare evaluator function
    evaluator_fn = make_evaluator_fn(args, evaluator_model)

    # Load initial prompts if provided
    initial_prompts = []
    if args.initial_prompts_json:
        with open(args.initial_prompts_json, encoding="utf-8") as f:
            initial_prompts = json.load(f)
        print(f"\nLoaded {len(initial_prompts)} initial prompts from {args.initial_prompts_json}")

    # Configure reward weights
    reward_weights = {
        "ba_clip": args.reward_w_ba_clip,
        "ba_cond": args.reward_w_ba_cond,
        "length_penalty": args.reward_w_length_penalty,
    }

    # Instantiate OPRO optimizer
    optimizer = OPROClassicOptimizer(
        optimizer_llm=args.optimizer_llm,
        device=args.optimizer_device,
        load_in_4bit=args.optimizer_load_in_4bit,
        top_k=args.top_k,
        candidates_per_iter=args.candidates_per_iter,
        reward_weights=reward_weights,
        seed=args.seed,
        baseline_prompt=args.baseline_prompt,
        initial_prompts=initial_prompts,
        max_new_tokens=args.optimizer_max_new_tokens,
        temperature=args.optimizer_temperature,
    )

    # Run optimization
    best_candidate = optimizer.run_optimization(
        evaluator_fn=evaluator_fn,
        n_iterations=args.num_iterations,
        early_stopping_patience=args.early_stopping,
        output_dir=output_dir,
    )

    # Save final summary
    summary = {
        "best_prompt": best_candidate.prompt,
        "reward": best_candidate.reward,
        "ba_clip": best_candidate.ba_clip,
        "ba_conditions": best_candidate.ba_conditions,
        "prompt_length": best_candidate.prompt_length,
        "iteration": best_candidate.iteration,
    }
    with open(output_dir / "best_prompt_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== BEST PROMPT ===")
    print(best_candidate.prompt)
    print(f"\nBA_clip: {best_candidate.ba_clip:.3f}")
    print(f"BA_conditions: {best_candidate.ba_conditions:.3f}")
    print(f"Reward: {best_candidate.reward:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
