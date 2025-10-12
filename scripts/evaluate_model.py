#!/usr/bin/env python3
"""
Comprehensive model evaluation script for Qwen2-Audio.

Evaluates performance across:
- Balanced 50/50 speech/no-speech samples
- All durations (20, 50, 100, 200, 500, 1000 ms)
- All SNR levels (-10, -5, 0, 5, 10, 20 dB)
- All band filters (none, telephony, lp3400, hp300)
- All reverberation levels (T60 bins)

Usage:
    # Basic evaluation (default prompts)
    python scripts/evaluate_model.py --n_samples 100

    # Custom user prompt only
    python scripts/evaluate_model.py \\
        --n_samples 100 \\
        --user_prompt "Does this audio contain human speech?" \\
        --use_prompt

    # Both system and user prompts
    python scripts/evaluate_model.py \\
        --n_samples 100 \\
        --system_prompt "You are an audio classifier." \\
        --user_prompt "Classify this audio segment." \\
        --use_prompt \\
        --output_dir results/evaluation_$(date +%Y%m%d_%H%M%S)

Output:
    - Accuracy by duration
    - Accuracy by SNR level
    - Accuracy by band filter
    - Accuracy by reverberation (T60)
    - Overall average accuracy across all conditions
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, List
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.models import Qwen2AudioClassifier
from qsm.data.loaders import load_ava_speech, FrameTable


def validate_sample_availability(
    speech_samples: int,
    nonspeech_samples: int,
    n_samples_requested: int
) -> None:
    """
    Validate that enough samples are available.

    Args:
        speech_samples: Number of available speech samples
        nonspeech_samples: Number of available non-speech samples
        n_samples_requested: Total number of samples requested

    Raises:
        ValueError: If not enough samples are available
    """
    n_per_class = n_samples_requested // 2

    if speech_samples < n_per_class:
        raise ValueError(
            f"Not enough speech samples available. "
            f"Requested: {n_per_class}, Available: {speech_samples}"
        )

    if nonspeech_samples < n_per_class:
        raise ValueError(
            f"Not enough non-speech samples available. "
            f"Requested: {n_per_class}, Available: {nonspeech_samples}"
        )


def load_complete_clips_dataset(
    conditions_manifest_path: Path,
    n_clips: int,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load dataset by selecting N complete clips with ALL their variants.

    Each clip has multiple variants:
    - Multiple durations (20, 40, 60, 80, 100, 200, 500, 1000 ms)
    - Multiple SNR levels (-10, -5, 0, 5, 10, 20 dB)
    - Multiple band filters (none, telephony, lp3400, hp300)

    Args:
        conditions_manifest_path: Path to conditions manifest parquet
        n_clips: Number of base clips to evaluate (will evaluate ALL variants of each)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all variants of selected clips
    """
    # Load full manifest
    df = pd.read_parquet(conditions_manifest_path)

    # Normalize labels
    df["label_normalized"] = df["label"].str.replace("-", "").str.replace("_", "")

    # Get unique clips per label (clip_id already represents the base clip)
    speech_clips = df[df["label_normalized"] == "SPEECH"]["clip_id"].unique()
    nonspeech_clips = df[df["label_normalized"] == "NONSPEECH"]["clip_id"].unique()

    # Sample n_clips/2 from each class
    n_per_class = n_clips // 2

    if len(speech_clips) < n_per_class:
        raise ValueError(
            f"Not enough speech clips available. "
            f"Requested: {n_per_class}, Available: {len(speech_clips)}"
        )

    if len(nonspeech_clips) < n_per_class:
        raise ValueError(
            f"Not enough non-speech clips available. "
            f"Requested: {n_per_class}, Available: {len(nonspeech_clips)}"
        )

    np.random.seed(seed)
    selected_speech_clips = np.random.choice(speech_clips, size=n_per_class, replace=False)
    selected_nonspeech_clips = np.random.choice(nonspeech_clips, size=n_per_class, replace=False)

    # Get ALL variants of selected clips
    selected_clips = list(selected_speech_clips) + list(selected_nonspeech_clips)
    selected_df = df[df["clip_id"].isin(selected_clips)]

    # Shuffle
    selected_df = selected_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return selected_df


def load_balanced_dataset(
    conditions_manifest_path: Path,
    n_samples: int,
    seed: int = 42,
    balance_by_variant: bool = True
) -> pd.DataFrame:
    """
    Load balanced dataset with 50% speech, 50% non-speech.

    Optionally balances by variant type (SNR, band, RIR) to ensure equal
    representation of each condition type within each label class.

    Args:
        conditions_manifest_path: Path to conditions manifest parquet
        n_samples: Total number of samples to load
        seed: Random seed for reproducibility
        balance_by_variant: If True, balance samples across variant types

    Returns:
        Balanced DataFrame with n_samples rows
    """
    # Load full manifest
    df = pd.read_parquet(conditions_manifest_path)

    # Normalize labels to handle variations (NONSPEECH vs NON-SPEECH)
    df["label_normalized"] = df["label"].str.replace("-", "").str.replace("_", "")

    # Separate by label
    speech_df = df[df["label_normalized"] == "SPEECH"]
    nonspeech_df = df[df["label_normalized"] == "NONSPEECH"]

    # Validate availability
    validate_sample_availability(
        len(speech_df),
        len(nonspeech_df),
        n_samples
    )

    # Sample 50/50
    n_per_class = n_samples // 2

    np.random.seed(seed)

    if balance_by_variant:
        # Balance by variant type within each class
        def sample_balanced_by_variant(class_df, n_samples_needed, random_state):
            """Sample ensuring proportional representation of variant types."""
            variant_types = class_df["variant_type"].unique()
            n_variants = len(variant_types)

            if n_variants == 0:
                return pd.DataFrame()

            samples_per_variant = n_samples_needed // n_variants
            remainder = n_samples_needed % n_variants

            sampled_dfs = []
            for i, variant in enumerate(variant_types):
                variant_df = class_df[class_df["variant_type"] == variant]
                n_to_sample = samples_per_variant + (1 if i < remainder else 0)

                if len(variant_df) >= n_to_sample:
                    sampled = variant_df.sample(n=n_to_sample, random_state=random_state + i)
                else:
                    # If not enough samples, take all available
                    sampled = variant_df

                sampled_dfs.append(sampled)

            return pd.concat(sampled_dfs, ignore_index=True)

        speech_sample = sample_balanced_by_variant(speech_df, n_per_class, seed)
        nonspeech_sample = sample_balanced_by_variant(nonspeech_df, n_per_class, seed + 100)
    else:
        # Simple random sampling
        speech_sample = speech_df.sample(n=n_per_class, random_state=seed)
        nonspeech_sample = nonspeech_df.sample(n=n_per_class, random_state=seed + 1)

    # Combine and shuffle
    balanced_df = pd.concat([speech_sample, nonspeech_sample], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)

    return balanced_df


def evaluate_samples(
    model: Qwen2AudioClassifier,
    samples_df: pd.DataFrame
) -> List[Dict]:
    """
    Evaluate all samples with the model.

    Args:
        model: Qwen2AudioClassifier instance
        samples_df: DataFrame with audio samples

    Returns:
        List of result dictionaries
    """
    results = []

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Evaluating"):
        # Convert path to use forward slashes (works on both Windows and Linux/WSL)
        audio_path_str = str(row["audio_path"]).replace("\\", "/")
        audio_path = Path(audio_path_str)

        if not audio_path.exists():
            print(f"\nWARNING: Audio file not found: {audio_path}")
            continue

        try:
            # Run prediction
            pred = model.predict(audio_path)

            # Check correctness
            ground_truth = row["label"]
            pred_normalized = pred.label.replace("-", "").replace("_", "")
            gt_normalized = ground_truth.replace("-", "").replace("_", "")
            is_correct = (pred_normalized == gt_normalized)

            # Store result
            result = {
                # Identification
                "clip_id": row["clip_id"],
                "audio_path": str(audio_path),

                # Ground truth
                "ground_truth": ground_truth,
                "duration_ms": row["duration_ms"],

                # Manipulation parameters
                "variant_type": row["variant_type"],
                "snr_db": row.get("snr_db"),
                "band_filter": row.get("band_filter"),
                "rir_id": row.get("rir_id"),
                "T60": row.get("T60"),
                "T60_bin": row.get("T60_bin"),

                # Model output
                "predicted": pred.label,
                "confidence": pred.confidence,
                "correct": is_correct,
                "raw_output": pred.raw_output,
                "latency_ms": pred.latency_ms,
            }

            results.append(result)

        except Exception as e:
            print(f"\nERROR processing {audio_path}: {e}")
            # Store error result
            result = {
                "clip_id": row["clip_id"],
                "audio_path": str(audio_path),
                "ground_truth": row["label"],
                "duration_ms": row["duration_ms"],
                "variant_type": row["variant_type"],
                "snr_db": row.get("snr_db"),
                "band_filter": row.get("band_filter"),
                "rir_id": row.get("rir_id"),
                "T60": row.get("T60"),
                "T60_bin": row.get("T60_bin"),
                "predicted": "ERROR",
                "confidence": 0.0,
                "correct": False,
                "raw_output": str(e),
                "latency_ms": None,
            }
            results.append(result)

    return results


def compute_metrics(results_df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive metrics from results.

    Args:
        results_df: DataFrame with evaluation results

    Returns:
        Dictionary with all metrics
    """
    metrics = {}

    # Overall accuracy
    metrics["overall_accuracy"] = results_df["correct"].mean() * 100
    metrics["total_samples"] = len(results_df)
    metrics["correct_samples"] = int(results_df["correct"].sum())

    # Accuracy by duration
    metrics["by_duration"] = {}
    duration_groups = results_df.groupby("duration_ms")["correct"]
    for duration_ms, group in duration_groups:
        metrics["by_duration"][int(duration_ms)] = {
            "accuracy": group.mean() * 100,
            "n_samples": len(group),
            "n_correct": int(group.sum())
        }

    # Accuracy by SNR level
    metrics["by_snr"] = {}
    if "snr_db" in results_df.columns and results_df["snr_db"].notna().any():
        snr_groups = results_df[results_df["snr_db"].notna()].groupby("snr_db")["correct"]
        for snr_db, group in snr_groups:
            metrics["by_snr"][float(snr_db)] = {
                "accuracy": group.mean() * 100,
                "n_samples": len(group),
                "n_correct": int(group.sum())
            }

    # Accuracy by band filter
    metrics["by_band_filter"] = {}
    if "band_filter" in results_df.columns and results_df["band_filter"].notna().any():
        band_groups = results_df[results_df["band_filter"].notna()].groupby("band_filter")["correct"]
        for band_filter, group in band_groups:
            metrics["by_band_filter"][str(band_filter)] = {
                "accuracy": group.mean() * 100,
                "n_samples": len(group),
                "n_correct": int(group.sum())
            }

    # Accuracy by reverberation (T60 bin)
    metrics["by_t60"] = {}
    if "T60_bin" in results_df.columns and results_df["T60_bin"].notna().any():
        t60_groups = results_df[results_df["T60_bin"].notna()].groupby("T60_bin")["correct"]
        for t60_bin, group in t60_groups:
            metrics["by_t60"][str(t60_bin)] = {
                "accuracy": group.mean() * 100,
                "n_samples": len(group),
                "n_correct": int(group.sum())
            }

    # Compute global average across all condition types
    # (average of: durations, SNRs, band filters, T60s)
    condition_averages = []

    if metrics["by_duration"]:
        condition_averages.append(
            np.mean([v["accuracy"] for v in metrics["by_duration"].values()])
        )

    if metrics["by_snr"]:
        condition_averages.append(
            np.mean([v["accuracy"] for v in metrics["by_snr"].values()])
        )

    if metrics["by_band_filter"]:
        condition_averages.append(
            np.mean([v["accuracy"] for v in metrics["by_band_filter"].values()])
        )

    if metrics["by_t60"]:
        condition_averages.append(
            np.mean([v["accuracy"] for v in metrics["by_t60"].values()])
        )

    if condition_averages:
        metrics["global_average"] = np.mean(condition_averages)
    else:
        metrics["global_average"] = metrics["overall_accuracy"]

    # Latency statistics
    valid_latencies = results_df[results_df["latency_ms"].notna()]["latency_ms"]
    if len(valid_latencies) > 0:
        metrics["latency"] = {
            "mean_ms": float(valid_latencies.mean()),
            "median_ms": float(valid_latencies.median()),
            "std_ms": float(valid_latencies.std()),
            "min_ms": float(valid_latencies.min()),
            "max_ms": float(valid_latencies.max()),
        }

    return metrics


def print_metrics(metrics: Dict) -> None:
    """Print formatted metrics report."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall
    print(f"\nOverall Performance")
    print("-" * 40)
    print(f"  Total samples: {metrics['total_samples']:,}")
    print(f"  Correct: {metrics['correct_samples']:,}")
    print(f"  Accuracy: {metrics['overall_accuracy']:.2f}%")

    # Global average
    print(f"\nGlobal Average (across all conditions)")
    print("-" * 40)
    print(f"  Average accuracy: {metrics['global_average']:.2f}%")

    # By duration
    if metrics["by_duration"]:
        print(f"\nAccuracy by Duration")
        print("-" * 40)
        for duration_ms in sorted(metrics["by_duration"].keys()):
            m = metrics["by_duration"][duration_ms]
            print(f"  {duration_ms:4d} ms: {m['accuracy']:6.2f}% (n={m['n_samples']:,})")

    # By SNR
    if metrics["by_snr"]:
        print(f"\nAccuracy by SNR Level")
        print("-" * 40)
        for snr_db in sorted(metrics["by_snr"].keys()):
            m = metrics["by_snr"][snr_db]
            print(f"  {snr_db:+5.0f} dB: {m['accuracy']:6.2f}% (n={m['n_samples']:,})")

    # By band filter
    if metrics["by_band_filter"]:
        print(f"\nAccuracy by Band Filter")
        print("-" * 40)
        for band_filter in sorted(metrics["by_band_filter"].keys()):
            m = metrics["by_band_filter"][band_filter]
            print(f"  {band_filter:12s}: {m['accuracy']:6.2f}% (n={m['n_samples']:,})")

    # By T60
    if metrics["by_t60"]:
        print(f"\nAccuracy by Reverberation (T60)")
        print("-" * 40)
        for t60_bin in sorted(metrics["by_t60"].keys()):
            m = metrics["by_t60"][t60_bin]
            print(f"  {t60_bin:15s}: {m['accuracy']:6.2f}% (n={m['n_samples']:,})")

    # Latency
    if "latency" in metrics:
        print(f"\nLatency Statistics")
        print("-" * 40)
        lat = metrics["latency"]
        print(f"  Mean: {lat['mean_ms']:.1f} ms")
        print(f"  Median: {lat['median_ms']:.1f} ms")
        print(f"  Std: {lat['std_ms']:.1f} ms")
        print(f"  Range: [{lat['min_ms']:.1f}, {lat['max_ms']:.1f}] ms")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments (mutually exclusive)
    eval_mode = parser.add_mutually_exclusive_group(required=True)
    eval_mode.add_argument(
        "--n_samples",
        type=int,
        help="Total number of VARIANTS to evaluate (random sampling, must be even for 50/50 balance)"
    )
    eval_mode.add_argument(
        "--n_clips",
        type=int,
        help="Number of base CLIPS to evaluate (will evaluate ALL variants of each clip: ~72 variants per clip)"
    )

    # Optional arguments
    parser.add_argument(
        "--conditions_manifest",
        type=Path,
        default=Path("data/processed/conditions/conditions_manifest.parquet"),
        help="Path to conditions manifest (default: data/processed/conditions/conditions_manifest.parquet)"
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Custom system prompt for the model (optional)"
    )

    parser.add_argument(
        "--user_prompt",
        type=str,
        default=None,
        help="Custom user prompt for the model (optional)"
    )

    parser.add_argument(
        "--use_prompt",
        action="store_true",
        help="Enable custom prompts (if not set, uses default model prompts)"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results/)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run model on (cuda/cpu)"
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization (default: True)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--balance_by_variant",
        action="store_true",
        default=True,
        help="Balance samples across variant types (SNR, band, RIR) within each class (default: True)"
    )

    args = parser.parse_args()

    # Determine evaluation mode
    eval_mode = "clips" if args.n_clips else "samples"

    # Validate inputs
    if eval_mode == "samples" and args.n_samples % 2 != 0:
        print(f"ERROR: n_samples must be even for 50/50 balance. Got: {args.n_samples}")
        sys.exit(1)

    if eval_mode == "clips" and args.n_clips % 2 != 0:
        print(f"ERROR: n_clips must be even for 50/50 balance. Got: {args.n_clips}")
        sys.exit(1)

    # Print configuration
    print("=" * 80)
    print("MODEL EVALUATION CONFIGURATION")
    print("=" * 80)
    if eval_mode == "clips":
        print(f"Evaluation mode: COMPLETE CLIPS")
        print(f"Number of clips: {args.n_clips} (50% speech, 50% non-speech)")
        print(f"Expected variants: ~{args.n_clips * 20} total (20 variants per clip)")
        print(f"  - 8 durations: 20, 40, 60, 80, 100, 200, 500, 1000 ms")
        print(f"  - 6 SNR levels: -10, -5, 0, +5, +10, +20 dB")
        print(f"  - 3 band filters: telephony, hp300, lp3400")
        print(f"  - 3 reverb bins: T60 0.0-0.4, 0.4-0.8, 0.8-1.5 s")
    else:
        print(f"Evaluation mode: RANDOM VARIANTS")
        print(f"Number of variants: {args.n_samples} (50% speech, 50% non-speech)")
    print(f"Conditions manifest: {args.conditions_manifest}")
    print(f"Custom prompt enabled: {args.use_prompt}")
    if args.use_prompt:
        if args.system_prompt:
            print(f"Custom system prompt: {args.system_prompt}")
        if args.user_prompt:
            print(f"Custom user prompt: {args.user_prompt}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Quantization: {'4-bit' if args.load_in_4bit else 'None'}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Check if manifest exists
    if not args.conditions_manifest.exists():
        print(f"\nERROR: Conditions manifest not found at {args.conditions_manifest}")
        print("\nPlease run: python scripts/build_conditions.py")
        sys.exit(1)

    # Load dataset based on mode
    print(f"\nLoading dataset...")
    start_time = time.time()

    try:
        if eval_mode == "clips":
            samples_df = load_complete_clips_dataset(
                args.conditions_manifest,
                args.n_clips,
                seed=args.seed
            )
        else:
            samples_df = load_balanced_dataset(
                args.conditions_manifest,
                args.n_samples,
                seed=args.seed,
                balance_by_variant=args.balance_by_variant
            )
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    load_time = time.time() - start_time
    print(f"Loaded {len(samples_df)} samples in {load_time:.2f}s")

    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Speech samples: {(samples_df['label_normalized'] == 'SPEECH').sum()}")
    print(f"  Non-speech samples: {(samples_df['label_normalized'] == 'NONSPEECH').sum()}")

    # Breakdown by variant type and label
    print(f"\n  Breakdown by variant type:")
    for variant_type in sorted(samples_df["variant_type"].unique()):
        variant_subset = samples_df[samples_df["variant_type"] == variant_type]
        n_speech = (variant_subset["label_normalized"] == "SPEECH").sum()
        n_nonspeech = (variant_subset["label_normalized"] == "NONSPEECH").sum()
        print(f"    {variant_type:8s}: {len(variant_subset):3d} total ({n_speech:2d} speech, {n_nonspeech:2d} non-speech)")

    print(f"\n  Unique durations: {samples_df['duration_ms'].nunique()}")
    if "snr_db" in samples_df.columns and samples_df["snr_db"].notna().any():
        print(f"  Unique SNR levels: {samples_df['snr_db'].nunique()}")
    if "band_filter" in samples_df.columns and samples_df["band_filter"].notna().any():
        print(f"  Unique band filters: {samples_df['band_filter'].nunique()}")
    if "T60_bin" in samples_df.columns and samples_df["T60_bin"].notna().any():
        print(f"  Unique T60 bins: {samples_df['T60_bin'].nunique()}")

    # Load model
    print(f"\nLoading Qwen2-Audio model...")
    model = Qwen2AudioClassifier(
        device=args.device,
        torch_dtype="float16",
        load_in_4bit=args.load_in_4bit,
        auto_pad=False,  # Audio is already padded
    )

    # Set custom prompts if requested
    if args.use_prompt:
        print(f"\nSetting custom prompts...")
        model.set_prompt(
            system_prompt=args.system_prompt,
            user_prompt=args.user_prompt
        )

    print(f"\nActive Prompt Configuration:")
    print(f"  System: {model.system_prompt}")
    print(f"  User: {model.user_prompt}")

    # Evaluate
    print(f"\nStarting evaluation...")
    eval_start = time.time()

    results = evaluate_samples(model, samples_df)

    eval_time = time.time() - eval_start
    print(f"\nEvaluation completed in {eval_time:.1f}s ({eval_time/60:.1f} minutes)")

    # Check if we have any results
    if len(results) == 0:
        print("\nERROR: No samples were successfully evaluated!")
        print("All audio files were not found. Please check:")
        print(f"  1. Manifest path: {args.conditions_manifest}")
        print(f"  2. Audio file paths in the manifest")
        print(f"  3. Run 'python scripts/build_conditions.py' to generate audio files")
        sys.exit(1)

    print(f"Average time per sample: {eval_time/len(results):.2f}s")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_metrics(results_df)

    # Print metrics
    print_metrics(metrics)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_path = args.output_dir / "evaluation_results.parquet"
    results_df.to_parquet(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")

    # Save metrics JSON
    metrics_path = args.output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    # Save configuration
    config_path = args.output_dir / "evaluation_config.json"
    config = {
        "n_samples": args.n_samples,
        "conditions_manifest": str(args.conditions_manifest),
        "system_prompt": args.system_prompt if args.use_prompt else None,
        "user_prompt": args.user_prompt if args.use_prompt else None,
        "use_prompt": args.use_prompt,
        "device": args.device,
        "load_in_4bit": args.load_in_4bit,
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation_time_seconds": eval_time,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
