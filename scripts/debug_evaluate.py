#!/usr/bin/env python3
"""
Debug evaluation script with detailed logging.

Evaluates a small number of samples with extensive logging for debugging.
Saves:
- Detailed prediction logs for each sample
- Audio samples for manual inspection
- Comparison between prompt strategies
"""

import sys
import argparse
import json
import time
import shutil
from pathlib import Path
from dataclasses import asdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.models import Qwen2AudioClassifier


def setup_logging(output_dir: Path):
    """Create output directory and log file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "debug_log.txt"

    def log(message: str):
        """Log to both console and file."""
        print(message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    return log


def load_sample_dataset(
    manifest_path: Path,
    n_clips: int,
    seed: int = 42
) -> pd.DataFrame:
    """
    Load a small balanced dataset for debugging.

    Args:
        manifest_path: Path to conditions manifest
        n_clips: Number of clips per class (total will be n_clips * 2 * 20 variants)
        seed: Random seed

    Returns:
        DataFrame with selected samples
    """
    df = pd.read_parquet(manifest_path)

    # Normalize labels and add ground_truth column
    df["ground_truth"] = df["label"].str.replace("-", "").str.replace("_", "").str.upper()

    # Get unique clips per label
    speech_clips = df[df["ground_truth"] == "SPEECH"]["clip_id"].unique()
    nonspeech_clips = df[df["ground_truth"] == "NONSPEECH"]["clip_id"].unique()

    # Validate we have enough clips
    max_clips_speech = len(speech_clips)
    max_clips_nonspeech = len(nonspeech_clips)
    max_clips = min(max_clips_speech, max_clips_nonspeech)

    if n_clips > max_clips:
        print(f"Warning: Requested {n_clips} clips per class, but only {max_clips_speech} SPEECH and {max_clips_nonspeech} NONSPEECH available.")
        print(f"Using {max_clips} clips per class instead.")
        n_clips = max_clips

    # Sample clips
    np.random.seed(seed)
    selected_speech = np.random.choice(speech_clips, size=n_clips, replace=False)
    selected_nonspeech = np.random.choice(nonspeech_clips, size=n_clips, replace=False)

    # Get all variants
    selected_clips = list(selected_speech) + list(selected_nonspeech)
    selected_df = df[df["clip_id"].isin(selected_clips)]

    return selected_df


def evaluate_with_logging(
    model: Qwen2AudioClassifier,
    samples_df: pd.DataFrame,
    output_dir: Path,
    log_func,
    copy_audio: bool = True
):
    """
    Evaluate samples with detailed logging.

    Args:
        model: Loaded Qwen2AudioClassifier
        samples_df: DataFrame with samples to evaluate
        output_dir: Output directory for logs and audio copies
        log_func: Logging function
        copy_audio: Whether to copy audio files for manual inspection
    """
    results = []
    audio_dir = output_dir / "audio_samples"
    if copy_audio:
        audio_dir.mkdir(exist_ok=True)

    log_func(f"\n{'=' * 80}")
    log_func("STARTING DETAILED EVALUATION")
    log_func(f"{'=' * 80}\n")

    for idx, row in tqdm(samples_df.iterrows(), total=len(samples_df), desc="Evaluating"):
        # Convert Windows paths to cross-platform format
        audio_path_str = str(row["audio_path"]).replace(chr(92), '/')
        audio_path = Path(audio_path_str)
        ground_truth = row["ground_truth"]

        if not audio_path.exists():
            log_func(f"\n[ERROR] Audio not found: {audio_path}")
            continue

        try:
            # Predict
            pred = model.predict(audio_path)

            is_correct = pred.label == ground_truth

            # Log detailed info
            log_func(f"\n{'-' * 80}")
            log_func(f"Sample {idx + 1}/{len(samples_df)}")
            log_func(f"{'-' * 80}")
            log_func(f"Audio: {audio_path.name}")
            log_func(f"Clip ID: {row['clip_id']}")
            log_func(f"Variant type: {row['variant_type']}")

            # Variant-specific info
            if row['variant_type'] == 'duration':
                log_func(f"  Duration: {row['duration_ms']:.0f} ms")
            elif row['variant_type'] == 'snr':
                log_func(f"  SNR: {row['snr_db']:+.0f} dB")
            elif row['variant_type'] == 'band':
                log_func(f"  Band filter: {row['band_filter']}")
            elif row['variant_type'] == 'rir':
                log_func(f"  T60: {row['T60']:.3f}s ({row['T60_bin']})")

            log_func(f"\nGround Truth: {ground_truth}")
            log_func(f"Predicted: {pred.label}")
            log_func(f"Raw Output: '{pred.raw_output}'")
            log_func(f"Confidence: {pred.confidence:.2f}")
            log_func(f"Latency: {pred.latency_ms:.1f} ms")
            log_func(f"Result: {'[OK] CORRECT' if is_correct else '[FAIL] INCORRECT'}")

            # Copy audio for manual inspection if incorrect or low confidence
            if copy_audio and (not is_correct or pred.confidence < 0.9):
                status = "correct" if is_correct else "incorrect"
                conf_str = f"{int(pred.confidence * 100):02d}"
                new_name = f"{status}_{ground_truth}_{pred.label}_conf{conf_str}_{audio_path.name}"
                dest_path = audio_dir / new_name
                shutil.copy2(audio_path, dest_path)
                log_func(f"[COPY] Copied to: {dest_path.name}")

            # Store result
            results.append({
                "clip_id": row["clip_id"],
                "audio_path": str(audio_path),
                "audio_filename": audio_path.name,
                "ground_truth": ground_truth,
                "predicted": pred.label,
                "raw_output": pred.raw_output,
                "confidence": pred.confidence,
                "correct": is_correct,
                "latency_ms": pred.latency_ms,
                "variant_type": row["variant_type"],
                "duration_ms": row.get("duration_ms"),
                "snr_db": row.get("snr_db"),
                "band_filter": row.get("band_filter"),
                "T60": row.get("T60"),
                "T60_bin": row.get("T60_bin"),
            })

        except Exception as e:
            log_func(f"\n[ERROR] ERROR processing {audio_path}: {e}")
            import traceback
            log_func(traceback.format_exc())

    return results


def print_summary(results: list, log_func):
    """Print evaluation summary."""
    df = pd.DataFrame(results)

    log_func(f"\n{'=' * 80}")
    log_func("EVALUATION SUMMARY")
    log_func(f"{'=' * 80}\n")

    # Check if we have any results
    if len(df) == 0 or "correct" not in df.columns:
        log_func("ERROR: No results to summarize!")
        log_func("")
        log_func("Possible causes:")
        log_func("  - All audio files were not found")
        log_func("  - Dataset manifest points to non-existent files")
        log_func("  - Path format issues (Windows vs Linux)")
        log_func("")
        log_func("Please regenerate the dataset or check file paths.")
        return

    # Overall accuracy
    total = len(df)
    correct = df["correct"].sum()
    accuracy = correct / total if total > 0 else 0.0

    log_func(f"Overall Performance:")
    log_func(f"  Total samples: {total}")
    log_func(f"  Correct: {correct}")
    log_func(f"  Accuracy: {accuracy * 100:.2f}%\n")

    # By variant type
    log_func("Accuracy by Variant Type:")
    for vtype in df["variant_type"].unique():
        subset = df[df["variant_type"] == vtype]
        vtype_acc = subset["correct"].mean()
        log_func(f"  {vtype:12s}: {vtype_acc * 100:5.1f}% (n={len(subset)})")

    # By label
    log_func("\nAccuracy by Ground Truth Label:")
    for label in ["SPEECH", "NONSPEECH"]:
        subset = df[df["ground_truth"] == label]
        if len(subset) > 0:
            label_acc = subset["correct"].mean()
            log_func(f"  {label:12s}: {label_acc * 100:5.1f}% (n={len(subset)})")

    # Confusion matrix
    log_func("\nConfusion Matrix:")
    confusion = pd.crosstab(
        df["ground_truth"],
        df["predicted"],
        rownames=["True"],
        colnames=["Predicted"]
    )
    log_func(str(confusion))

    # Duration breakdown (if available)
    if "duration_ms" in df.columns and df["duration_ms"].notna().any():
        log_func("\nAccuracy by Duration:")
        duration_df = df[df["variant_type"] == "duration"].copy()
        if len(duration_df) > 0:
            for dur in sorted(duration_df["duration_ms"].unique()):
                subset = duration_df[duration_df["duration_ms"] == dur]
                dur_acc = subset["correct"].mean()
                log_func(f"  {dur:4.0f} ms: {dur_acc * 100:5.1f}% (n={len(subset)})")

    # SNR breakdown (if available)
    if "snr_db" in df.columns and df["snr_db"].notna().any():
        log_func("\nAccuracy by SNR:")
        snr_df = df[df["variant_type"] == "snr"].copy()
        if len(snr_df) > 0:
            for snr in sorted(snr_df["snr_db"].unique()):
                subset = snr_df[snr_df["snr_db"] == snr]
                snr_acc = subset["correct"].mean()
                log_func(f"  {snr:+5.0f} dB: {snr_acc * 100:5.1f}% (n={len(subset)})")

    # Latency stats
    log_func("\nLatency Statistics:")
    log_func(f"  Mean: {df['latency_ms'].mean():.1f} ms")
    log_func(f"  Median: {df['latency_ms'].median():.1f} ms")
    log_func(f"  Std: {df['latency_ms'].std():.1f} ms")
    log_func(f"  Min: {df['latency_ms'].min():.1f} ms")
    log_func(f"  Max: {df['latency_ms'].max():.1f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions_manifest",
        type=Path,
        default=Path("data/processed/conditions_final/conditions_manifest.parquet"),
        help="Path to conditions manifest"
    )
    parser.add_argument(
        "--n_clips",
        type=int,
        default=2,
        help="Number of clips per class to evaluate (default: 2, total 40 variants)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/debug"),
        help="Output directory for logs and audio samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--copy_audio",
        action="store_true",
        default=True,
        help="Copy problematic audio samples for manual inspection"
    )

    args = parser.parse_args()

    # Setup logging
    log = setup_logging(args.output_dir)

    log(f"{'=' * 80}")
    log("DEBUG EVALUATION SCRIPT")
    log(f"{'=' * 80}\n")
    log(f"Configuration:")
    log(f"  Manifest: {args.conditions_manifest}")
    log(f"  Clips per class: {args.n_clips}")
    log(f"  Expected variants: ~{args.n_clips * 2 * 20} (2 classes × {args.n_clips} clips × 20 variants)")
    log(f"  Output: {args.output_dir}")
    log(f"  Device: {args.device}")
    log(f"  4-bit quantization: {args.load_in_4bit}")
    log(f"  Seed: {args.seed}")
    log(f"  Copy audio: {args.copy_audio}")

    # Load dataset
    log(f"\nLoading dataset...")
    samples_df = load_sample_dataset(
        args.conditions_manifest,
        args.n_clips,
        args.seed
    )
    log(f"Loaded {len(samples_df)} samples")
    log(f"  SPEECH: {len(samples_df[samples_df['ground_truth'] == 'SPEECH'])}")
    log(f"  NONSPEECH: {len(samples_df[samples_df['ground_truth'] == 'NONSPEECH'])}")

    # Load model
    log(f"\nLoading Qwen2-Audio model...")
    model = Qwen2AudioClassifier(
        device=args.device,
        load_in_4bit=args.load_in_4bit
    )

    # Show active prompt
    log(f"\nActive Prompt Configuration:")
    log(f"  System: {model.system_prompt}")
    log(f"  User: {model.user_prompt}")

    # Evaluate
    results = evaluate_with_logging(
        model,
        samples_df,
        args.output_dir,
        log,
        copy_audio=args.copy_audio
    )

    # Save results
    results_df = pd.DataFrame(results)
    results_path = args.output_dir / "debug_results.parquet"
    results_df.to_parquet(results_path, index=False)
    log(f"\n[OK] Results saved to {results_path}")

    results_json = args.output_dir / "debug_results.json"
    results_df.to_json(results_json, orient="records", indent=2)
    log(f"[OK] Results saved to {results_json}")

    # Print summary
    print_summary(results, log)

    log(f"\n{'=' * 80}")
    log("DEBUG EVALUATION COMPLETE")
    log(f"{'=' * 80}")
    log(f"\nCheck the following files:")
    log(f"  - Detailed log: {args.output_dir / 'debug_log.txt'}")
    log(f"  - Results: {results_path}")
    log(f"  - Results (JSON): {results_json}")
    if args.copy_audio:
        log(f"  - Audio samples: {args.output_dir / 'audio_samples/'}")


if __name__ == "__main__":
    main()
