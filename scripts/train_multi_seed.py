"""Multi-seed training for robust evaluation.

Trains the model with multiple random seeds and reports:
- Mean ± SD accuracy
- Bootstrap confidence intervals
- Individual run results for McNemar test
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from typing import List, Dict
import subprocess

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


@dataclass
class RunResult:
    """Results from a single training run."""

    seed: int
    accuracy: float
    speech_acc: float
    nonspeech_acc: float
    train_loss: float
    confidence_correct: float
    confidence_wrong: float
    predictions: List[int]  # 0/1 for each test sample


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95):
    """Compute bootstrap confidence interval."""
    bootstrapped = np.random.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = bootstrapped.mean(axis=1)
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    return lower, upper


def train_single_seed(seed: int, output_dir: Path) -> RunResult:
    """Train model with a specific seed."""
    print(f"\n{'=' * 80}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'=' * 80}\n")

    # Clean up CUDA cache before training (prevents OOM between runs)
    import torch
    import gc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Set seed in environment
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)

    # Import after setting seed
    import random

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Run training
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "finetune_qwen_audio.py"),
            "--seed",
            str(seed),
            "--output_dir",
            str(output_dir),
        ],
        check=True,
    )

    # Run evaluation and capture results
    eval_script = script_dir / "test_normalized_model.py"
    result = subprocess.run(
        [sys.executable, str(eval_script), "--checkpoint", str(output_dir / "final")],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse results from test output
    import re

    lines = result.stdout.split("\n")
    accuracy = 0.0
    speech_acc = 0.0
    nonspeech_acc = 0.0
    train_loss = 0.0
    conf_correct = 0.0
    conf_wrong = 0.0
    predictions = []

    for line in lines:
        # Match: "RESULT: 29/32 = 90.6%"
        if "RESULT:" in line and "=" in line:
            match = re.search(r'(\d+)/(\d+)\s*=\s*([\d.]+)%', line)
            if match:
                accuracy = float(match.group(3))

        # Match class-specific lines with regex to distinguish SPEECH vs NONSPEECH
        # IMPORTANT: Check NONSPEECH first since "NONSPEECH" contains "SPEECH"!
        elif re.match(r'^\s*NONSPEECH:', line):
            match = re.search(r'(\d+)/(\d+)\s*=\s*([\d.]+)%', line)
            if match:
                nonspeech_acc = float(match.group(3))

        elif re.match(r'^\s*SPEECH:', line):
            match = re.search(r'(\d+)/(\d+)\s*=\s*([\d.]+)%', line)
            if match:
                speech_acc = float(match.group(3))

        # Match: "  Correct avg:  0.731"
        elif "Correct avg:" in line:
            match = re.search(r'Correct avg:\s*([\d.]+)', line)
            if match:
                conf_correct = float(match.group(1))

        # Match: "  Wrong avg:    0.574"
        elif "Wrong avg:" in line:
            match = re.search(r'Wrong avg:\s*([\d.]+)', line)
            if match:
                conf_wrong = float(match.group(1))

    return RunResult(
        seed=seed,
        accuracy=accuracy,
        speech_acc=speech_acc,
        nonspeech_acc=nonspeech_acc,
        train_loss=train_loss,
        confidence_correct=conf_correct,
        confidence_wrong=conf_wrong,
        predictions=predictions,
    )


def main():
    """Run multi-seed training."""
    # Configuration
    seeds = [42, 123, 456, 789, 2024]  # 5 seeds
    base_output_dir = (
        project_root / "checkpoints" / "qwen2_audio_speech_detection_multiseed"
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("MULTI-SEED TRAINING FOR ROBUST EVALUATION")
    print(f"{'=' * 80}")
    print(f"Seeds: {seeds}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'=' * 80}\n")

    # Run training for each seed
    results = []
    for seed in seeds:
        seed_dir = base_output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        try:
            result = train_single_seed(seed, seed_dir)
            results.append(result)

            print(f"\nSeed {seed} results:")
            print(f"  Accuracy: {result.accuracy:.1f}%")
            print(f"  SPEECH: {result.speech_acc:.1f}%")
            print(f"  NONSPEECH: {result.nonspeech_acc:.1f}%")

        except Exception as e:
            print(f"ERROR with seed {seed}: {e}")
            continue

    # Compute statistics
    if len(results) < 2:
        print("\nERROR: Need at least 2 successful runs for statistics")
        return

    accuracies = np.array([r.accuracy for r in results])
    speech_accs = np.array([r.speech_acc for r in results])
    nonspeech_accs = np.array([r.nonspeech_acc for r in results])

    # Bootstrap CIs
    acc_ci = bootstrap_ci(accuracies)
    speech_ci = bootstrap_ci(speech_accs)
    nonspeech_ci = bootstrap_ci(nonspeech_accs)

    # Print summary
    print(f"\n{'=' * 80}")
    print("MULTI-SEED RESULTS SUMMARY")
    print(f"{'=' * 80}\n")

    print(f"Number of runs: {len(results)}")
    print(f"\nOverall Accuracy:")
    print(f"  Mean: {accuracies.mean():.1f}%")
    print(f"  SD: {accuracies.std():.1f}%")
    print(f"  95% CI: [{acc_ci[0]:.1f}%, {acc_ci[1]:.1f}%]")
    print(f"  Range: [{accuracies.min():.1f}%, {accuracies.max():.1f}%]")

    print(f"\nSPEECH Accuracy:")
    print(f"  Mean: {speech_accs.mean():.1f}%")
    print(f"  SD: {speech_accs.std():.1f}%")
    print(f"  95% CI: [{speech_ci[0]:.1f}%, {speech_ci[1]:.1f}%]")

    print(f"\nNONSPEECH Accuracy:")
    print(f"  Mean: {nonspeech_accs.mean():.1f}%")
    print(f"  SD: {nonspeech_accs.std():.1f}%")
    print(f"  95% CI: [{nonspeech_ci[0]:.1f}%, {nonspeech_ci[1]:.1f}%]")

    print(f"\nIndividual runs:")
    for r in results:
        print(f"  Seed {r.seed}: {r.accuracy:.1f}% (SPEECH: {r.speech_acc:.1f}%, "
              f"NONSPEECH: {r.nonspeech_acc:.1f}%)")

    # Save results
    results_file = base_output_dir / "multiseed_results.json"
    results_dict = {
        "seeds": seeds,
        "runs": [
            {
                "seed": r.seed,
                "accuracy": r.accuracy,
                "speech_acc": r.speech_acc,
                "nonspeech_acc": r.nonspeech_acc,
                "train_loss": r.train_loss,
            }
            for r in results
        ],
        "summary": {
            "n_runs": len(results),
            "accuracy_mean": float(accuracies.mean()),
            "accuracy_std": float(accuracies.std()),
            "accuracy_ci_95": [float(acc_ci[0]), float(acc_ci[1])],
            "speech_mean": float(speech_accs.mean()),
            "speech_std": float(speech_accs.std()),
            "nonspeech_mean": float(nonspeech_accs.mean()),
            "nonspeech_std": float(nonspeech_accs.std()),
        },
    }

    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
