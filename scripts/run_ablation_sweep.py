#!/usr/bin/env python3
"""
Systematic ablation sweep for small-data optimization.

Runs all experiments from EXPERIMENT_MATRIX_SMALL_DATA.md systematically.

Usage:
    python scripts/run_ablation_sweep.py --phase 1
    python scripts/run_ablation_sweep.py --phase 2
    python scripts/run_ablation_sweep.py --phase all
"""

import argparse
import subprocess
import json
import pandas as pd
from pathlib import Path
from datetime import datetime


class ExperimentRunner:
    def __init__(self, base_dir="checkpoints/ablations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = Path("results/ablation_results.jsonl")
        self.results_file.parent.mkdir(parents=True, exist_ok=True)

    def run_training(self, exp_id, config, seed=42):
        """Run a single training experiment."""
        print(f"\n{'='*80}")
        print(f"Running experiment: {exp_id} (seed={seed})")
        print(f"Configuration: {config}")
        print(f"{'='*80}\n")

        output_dir = self.base_dir / exp_id / f"seed_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build training command
        cmd = [
            "python", "scripts/finetune_qwen_audio.py",
            "--seed", str(seed),
            "--output_dir", str(output_dir),
            "--lora_r", str(config.get("lora_r", 16)),
            "--lora_alpha", str(config.get("lora_alpha", 32)),
        ]

        if config.get("add_mlp_targets", False):
            cmd.append("--add_mlp_targets")

        # Run training
        print(f"Command: {' '.join(cmd)}")
        start_time = datetime.now()

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            success = True
            error_msg = None
        except subprocess.CalledProcessError as e:
            success = False
            error_msg = e.stderr
            print(f"ERROR: Training failed for {exp_id}")
            print(e.stderr)

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        return {
            "exp_id": exp_id,
            "config": config,
            "seed": seed,
            "output_dir": str(output_dir),
            "training_time": training_time,
            "success": success,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        }

    def run_evaluation(self, exp_id, checkpoint_dir, test_csv, temperature=1.0):
        """Run evaluation for a trained model."""
        print(f"\nEvaluating {exp_id} with T={temperature}")

        output_csv = Path("results/ablations") / f"{exp_id}_predictions.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python", "scripts/evaluate_with_logits.py",
            "--checkpoint", str(checkpoint_dir / "final"),
            "--test_csv", str(test_csv),
            "--temperature", str(temperature),
            "--output_csv", str(output_csv)
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            success = True

            # Parse results from output
            output = result.stdout
            overall_acc = self._parse_accuracy(output, "Overall Accuracy:")
            speech_acc = self._parse_accuracy(output, "SPEECH (A):")
            nonspeech_acc = self._parse_accuracy(output, "NONSPEECH (B):")

        except subprocess.CalledProcessError as e:
            success = False
            overall_acc = speech_acc = nonspeech_acc = None
            print(f"ERROR: Evaluation failed for {exp_id}")
            print(e.stderr)

        return {
            "exp_id": exp_id,
            "predictions_csv": str(output_csv),
            "overall_acc": overall_acc,
            "speech_acc": speech_acc,
            "nonspeech_acc": nonspeech_acc,
            "success": success
        }

    def _parse_accuracy(self, output, prefix):
        """Parse accuracy from evaluation output."""
        for line in output.split('\n'):
            if prefix in line:
                # Extract percentage
                parts = line.split('=')
                if len(parts) >= 2:
                    # Format: "X/Y = Z%"
                    pct = parts[-1].strip().replace('%', '')
                    try:
                        return float(pct)
                    except ValueError:
                        return None
        return None

    def save_result(self, result):
        """Save result to JSONL file."""
        with open(self.results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

    def load_results(self):
        """Load all results from JSONL file."""
        if not self.results_file.exists():
            return pd.DataFrame()

        results = []
        with open(self.results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))

        return pd.DataFrame(results)


def phase1_lora_targets(runner, seeds=[42, 123, 456]):
    """
    Phase 1: LoRA Targets Ablation
    Compare attention-only vs attention+MLP
    """
    print("\n" + "="*80)
    print("PHASE 1: LoRA Targets Ablation")
    print("="*80)

    experiments = [
        {
            "exp_id": "LORA-2a_attn_only",
            "config": {
                "lora_r": 16,
                "lora_alpha": 32,
                "add_mlp_targets": False
            }
        },
        {
            "exp_id": "LORA-2b_attn_mlp",
            "config": {
                "lora_r": 16,
                "lora_alpha": 32,
                "add_mlp_targets": True
            }
        }
    ]

    test_csv = "data/processed/grouped_split/test_metadata.csv"

    for exp in experiments:
        for seed in seeds:
            # Train
            train_result = runner.run_training(
                exp["exp_id"], exp["config"], seed=seed
            )
            runner.save_result(train_result)

            if train_result["success"]:
                # Evaluate
                checkpoint_dir = Path(train_result["output_dir"])
                eval_result = runner.run_evaluation(
                    exp["exp_id"], checkpoint_dir, test_csv, temperature=1.0
                )
                runner.save_result(eval_result)


def phase2_hyperparameter_grid(runner, seeds=[42, 123]):
    """
    Phase 2: Hyperparameter Grid Search
    """
    print("\n" + "="*80)
    print("PHASE 2: Hyperparameter Grid Search")
    print("="*80)

    # Grid: r × alpha × lr (keep dropout=0.05 fixed for now)
    grid = [
        {"lora_r": 8, "lora_alpha": 16},
        {"lora_r": 8, "lora_alpha": 32},
        {"lora_r": 16, "lora_alpha": 32},  # baseline
        {"lora_r": 16, "lora_alpha": 64},
        {"lora_r": 32, "lora_alpha": 32},
        {"lora_r": 32, "lora_alpha": 64},
    ]

    test_csv = "data/processed/grouped_split/test_metadata.csv"

    for i, params in enumerate(grid):
        exp_id = f"HP-{i+1}_r{params['lora_r']}_a{params['lora_alpha']}"
        config = {**params, "add_mlp_targets": False}  # attention-only

        for seed in seeds:
            train_result = runner.run_training(exp_id, config, seed=seed)
            runner.save_result(train_result)

            if train_result["success"]:
                checkpoint_dir = Path(train_result["output_dir"])
                eval_result = runner.run_evaluation(
                    exp_id, checkpoint_dir, test_csv, temperature=1.0
                )
                runner.save_result(eval_result)


def generate_results_summary(runner):
    """Generate summary table from all results."""
    df = runner.load_results()

    if df.empty:
        print("No results found.")
        return

    # Filter to evaluation results
    eval_df = df[df['exp_id'].notna() & df['overall_acc'].notna()].copy()

    # Group by exp_id to compute mean ± std
    summary = eval_df.groupby('exp_id').agg({
        'overall_acc': ['mean', 'std', 'count'],
        'speech_acc': ['mean', 'std'],
        'nonspeech_acc': ['mean', 'std'],
    }).round(2)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(summary.to_string())

    # Save to CSV
    summary_path = Path("results/ablation_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "all", "summary"],
        required=True,
        help="Which phase to run"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds to use"
    )

    args = parser.parse_args()

    runner = ExperimentRunner()

    if args.phase in ["1", "all"]:
        phase1_lora_targets(runner, seeds=args.seeds)

    if args.phase in ["2", "all"]:
        phase2_hyperparameter_grid(runner, seeds=args.seeds[:2])  # Only 2 seeds for HP grid

    if args.phase == "summary":
        generate_results_summary(runner)

    # Always generate summary at the end
    if args.phase != "summary":
        generate_results_summary(runner)


if __name__ == "__main__":
    main()
