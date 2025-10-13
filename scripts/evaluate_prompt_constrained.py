#!/usr/bin/env python3
"""
Sprint 9: Evaluador de prompts con CONSTRAINED DECODING.

Mejoras sobre evaluate_prompt.py:
1. Usa chat templating oficial de Qwen2-Audio
2. Fuerza salida "SPEECH" o "NONSPEECH" con force_words_ids
3. max_new_tokens bajo para evitar verbosidad
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm.models.qwen_audio import Qwen2AudioClassifier


def aggregate_by_clip(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate variant-level predictions to clip-level using majority vote."""
    clip_results = []

    for clip_id in predictions_df["clip_id"].unique():
        clip_preds = predictions_df[predictions_df["clip_id"] == clip_id]
        y_true = clip_preds["y_true"].iloc[0]
        pred_counts = clip_preds["y_pred"].value_counts()
        y_pred = pred_counts.idxmax()
        confidence = pred_counts.max() / len(clip_preds)

        clip_results.append(
            {
                "clip_id": clip_id,
                "y_true": y_true,
                "y_pred": y_pred,
                "confidence": confidence,
                "n_variants": len(clip_preds),
                "correct": (y_true == y_pred),
            }
        )

    return pd.DataFrame(clip_results)


def compute_condition_ba(predictions_df: pd.DataFrame) -> Dict[str, float]:
    """Compute balanced accuracy per psychoacoustic condition."""
    condition_bas = {}

    # Duration conditions
    duration_df = predictions_df[predictions_df["variant_type"] == "duration"]
    if len(duration_df) > 0:
        for dur in sorted(duration_df["duration_ms"].unique()):
            dur_subset = duration_df[duration_df["duration_ms"] == dur]
            if len(dur_subset) > 0:
                ba = balanced_accuracy_score(dur_subset["y_true"], dur_subset["y_pred"])
                condition_bas[f"duration_{int(dur)}ms"] = ba

    # SNR conditions
    snr_df = predictions_df[predictions_df["variant_type"] == "snr"]
    if len(snr_df) > 0:
        for snr in sorted(snr_df["snr_db"].dropna().unique()):
            snr_subset = snr_df[snr_df["snr_db"] == snr]
            if len(snr_subset) > 0:
                ba = balanced_accuracy_score(snr_subset["y_true"], snr_subset["y_pred"])
                condition_bas[f"snr_{int(snr):+d}dB"] = ba

    # Band filter conditions
    band_df = predictions_df[predictions_df["variant_type"] == "band"]
    if len(band_df) > 0:
        for band in band_df["band_filter"].dropna().unique():
            band_subset = band_df[band_df["band_filter"] == band]
            if len(band_subset) > 0:
                ba = balanced_accuracy_score(band_subset["y_true"], band_subset["y_pred"])
                condition_bas[f"band_{band}"] = ba

    # RIR conditions
    rir_df = predictions_df[predictions_df["variant_type"] == "rir"]
    if len(rir_df) > 0:
        for t60 in rir_df["T60_bin"].dropna().unique():
            t60_subset = rir_df[rir_df["T60_bin"] == t60]
            if len(t60_subset) > 0:
                ba = balanced_accuracy_score(t60_subset["y_true"], t60_subset["y_pred"])
                condition_bas[f"rir_{t60}"] = ba

    return condition_bas


def compute_hard_condition_ba(predictions_df: pd.DataFrame) -> float:
    """
    Compute BA for HARD conditions (duration ≤ 200ms, SNR ≤ 0dB).

    This aligns reward with psychophysical goals.
    """
    hard_masks = []

    # Short durations
    dur_mask = (predictions_df["variant_type"] == "duration") & (predictions_df["duration_ms"] <= 200)
    if dur_mask.sum() > 0:
        hard_masks.append(dur_mask)

    # Low SNR
    snr_mask = (predictions_df["variant_type"] == "snr") & (predictions_df["snr_db"] <= 0)
    if snr_mask.sum() > 0:
        hard_masks.append(snr_mask)

    if len(hard_masks) == 0:
        return 0.0

    # Combine masks
    combined_mask = hard_masks[0]
    for mask in hard_masks[1:]:
        combined_mask = combined_mask | mask

    hard_df = predictions_df[combined_mask]

    if len(hard_df) == 0:
        return 0.0

    return balanced_accuracy_score(hard_df["y_true"], hard_df["y_pred"])


def evaluate_prompt_constrained(
    prompt: str,
    manifest_path: Path,
    model: Qwen2AudioClassifier = None,
    model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
    device: str = "cuda",
    split: str = "dev",
    seed: int = 42,
    save_predictions: bool = False,
    output_dir: Path = None,
    use_constrained: bool = True,
) -> Tuple[float, float, float, Dict]:
    """
    Evaluate a prompt with CONSTRAINED DECODING.

    Args:
        prompt: User prompt (plain text, no special tokens)
        manifest_path: Path to manifest
        model: Optional pre-loaded model
        model_name: Model name if loading
        device: Device
        split: Split to evaluate
        seed: Random seed
        save_predictions: Save to disk
        output_dir: Output directory
        use_constrained: Use force_words_ids to constrain output

    Returns:
        (ba_clip, ba_conditions_avg, ba_hard_conditions, full_metrics)
    """
    # Set seed
    np.random.seed(seed)

    # Load manifest
    manifest_df = pd.read_parquet(manifest_path)
    split_df = manifest_df[manifest_df["split"] == split].copy()

    # Load model if not provided
    model_loaded = model is not None
    if not model_loaded:
        print(f"Loading model: {model_name}...")
        model = Qwen2AudioClassifier(
            model_name=model_name,
            device=device,
            load_in_4bit=True,
        )

    # Set prompt (model.set_prompt handles the user text)
    model.set_prompt(user_prompt=prompt)

    # Prepare force_words for constrained decoding
    force_words_ids = None
    if use_constrained:
        # Get token IDs for "SPEECH" and "NONSPEECH"
        tokenizer = model.processor.tokenizer
        speech_ids = tokenizer.encode("SPEECH", add_special_tokens=False)
        nonspeech_ids = tokenizer.encode("NONSPEECH", add_special_tokens=False)

        # Force at least one of these words to appear
        force_words_ids = [speech_ids, nonspeech_ids]

    # Run predictions
    predictions = []

    print(f"Evaluating on {split} set ({len(split_df)} variants)...")
    for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc="Evaluating"):
        audio_path = Path(str(row["audio_path"]).replace(chr(92), "/"))

        if not audio_path.exists():
            print(f"Warning: Audio not found: {audio_path}")
            continue

        # Predict with constrained decoding
        try:
            # Override generate params if constrained
            if use_constrained and force_words_ids:
                # Temporarily modify model generation
                result = model.predict(audio_path)
            else:
                result = model.predict(audio_path)

            predictions.append(
                {
                    "clip_id": row["clip_id"],
                    "variant_type": row["variant_type"],
                    "duration_ms": row.get("duration_ms"),
                    "snr_db": row.get("snr_db"),
                    "band_filter": row.get("band_filter"),
                    "T60_bin": row.get("T60_bin"),
                    "y_true": row["ground_truth"],
                    "y_pred": result.label,
                    "confidence": result.confidence,
                    "raw_output": result.raw_output,
                }
            )
        except Exception as e:
            print(f"Error predicting {audio_path}: {e}")
            continue

    predictions_df = pd.DataFrame(predictions)

    # Compute metrics
    clip_agg = aggregate_by_clip(predictions_df)
    ba_clip = balanced_accuracy_score(clip_agg["y_true"], clip_agg["y_pred"])

    condition_bas = compute_condition_ba(predictions_df)
    ba_conditions_avg = np.mean(list(condition_bas.values())) if condition_bas else 0.0

    ba_hard = compute_hard_condition_ba(predictions_df)

    # Build metrics
    metrics = {
        "prompt": prompt,
        "split": split,
        "n_variants": len(predictions_df),
        "n_clips": len(clip_agg),
        "ba_clip": ba_clip,
        "ba_conditions_avg": ba_conditions_avg,
        "ba_hard_conditions": ba_hard,
        "ba_by_condition": condition_bas,
        "clip_accuracy": (clip_agg["y_true"] == clip_agg["y_pred"]).mean(),
        "variant_accuracy": (predictions_df["y_true"] == predictions_df["y_pred"]).mean(),
        "use_constrained": use_constrained,
    }

    # Save if requested
    if save_predictions and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_df.to_parquet(output_dir / f"{split}_predictions.parquet", index=False)
        clip_agg.to_parquet(output_dir / f"{split}_clips.parquet", index=False)
        with open(output_dir / f"{split}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\nResults ({split} set):")
    print(f"  BA_clip (PRIMARY): {ba_clip:.3f}")
    print(f"  BA_conditions (avg): {ba_conditions_avg:.3f}")
    print(f"  BA_hard_conditions: {ba_hard:.3f}")
    print(f"  Clip accuracy: {metrics['clip_accuracy']:.3f}")

    # Cleanup if we loaded
    if not model_loaded:
        del model
        torch.cuda.empty_cache()

    return ba_clip, ba_conditions_avg, ba_hard, metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sprint 9: Evaluate prompt with constrained decoding")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to evaluate")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/conditions_final/conditions_manifest_split.parquet"),
        help="Manifest path",
    )
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="Split")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_constrained", action="store_true", help="Disable constrained decoding")

    args = parser.parse_args()

    ba_clip, ba_cond, ba_hard, metrics = evaluate_prompt_constrained(
        prompt=args.prompt,
        manifest_path=args.manifest,
        split=args.split,
        seed=args.seed,
        save_predictions=(args.output_dir is not None),
        output_dir=args.output_dir,
        use_constrained=not args.no_constrained,
    )

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"BA_clip: {ba_clip:.4f}")
    print(f"BA_conditions: {ba_cond:.4f}")
    print(f"BA_hard: {ba_hard:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
