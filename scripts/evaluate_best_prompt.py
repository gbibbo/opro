"""
Evaluate the best prompt from OPRO optimization on full dev set.

Reproduces Sprint 6/7 evaluation methodology:
- Clip-level BA
- Per-condition BA (duration, SNR, band, RIR)
- Bootstrap confidence intervals
- Saves predictions and metrics
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from qsm.models.qwen_audio import Qwen2AudioClassifier


def resolve_audio_path(raw_path: str, script_dir: Path) -> str:
    """Convert to absolute path and handle WSL/Windows conversion."""
    p = Path(str(raw_path))

    # Make absolute if relative
    if not p.is_absolute():
        p = (script_dir / p).resolve()

    # Convert Windows path to WSL if needed (C:\ -> /mnt/c/)
    path_str = str(p)
    if sys.platform.startswith('linux') and ':' in path_str and '\\' in path_str:
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        path_str = f"/mnt/{drive}{rest}"
    elif sys.platform.startswith('linux'):
        path_str = path_str.replace('\\', '/')

    return path_str


def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, confidence=0.95, clip_ids=None):
    """Calculate bootstrap CI for balanced accuracy."""
    np.random.seed(42)
    scores = []

    if clip_ids is not None:
        # Bootstrap at clip level
        unique_clips = pd.Series(clip_ids).unique()
        for _ in range(n_bootstrap):
            sampled_clips = np.random.choice(unique_clips, size=len(unique_clips), replace=True)
            mask = pd.Series(clip_ids).isin(sampled_clips).values
            scores.append(balanced_accuracy_score(
                np.array(y_true)[mask],
                np.array(y_pred)[mask]
            ))
    else:
        # Bootstrap at sample level
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            scores.append(balanced_accuracy_score(
                np.array(y_true)[indices],
                np.array(y_pred)[indices]
            ))

    lower = np.percentile(scores, (1 - confidence) / 2 * 100)
    upper = np.percentile(scores, (1 + confidence) / 2 * 100)

    return lower, upper


def evaluate_prompt(prompt: str, manifest_path: Path, output_dir: Path):
    """Evaluate prompt on full dev set."""

    print("="*80)
    print("EVALUATING PROMPT ON FULL DEV SET")
    print("="*80)
    print(f"Prompt: {prompt}")
    print()

    # Load manifest
    script_dir = manifest_path.parent.parent.parent
    manifest_df = pd.read_parquet(str(manifest_path))
    manifest_df = manifest_df[manifest_df["split"] == "dev"].reset_index(drop=True)

    # Convert paths
    manifest_df["audio_path"] = manifest_df["audio_path"].apply(
        lambda x: resolve_audio_path(x, script_dir)
    )

    print(f"Loaded {len(manifest_df)} dev samples")
    print(f"Variant types: {manifest_df['variant_type'].unique().tolist()}")
    print(f"Label distribution: {manifest_df['label'].value_counts().to_dict()}")
    print()

    # Load evaluator
    print("Loading Qwen2-Audio evaluator...")
    evaluator = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda",
        load_in_4bit=True,
    )
    evaluator.set_prompt(user_prompt=prompt)
    print("[OK] Evaluator loaded\n")

    # Evaluate all samples
    print("Evaluating all samples...")
    predictions = []

    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Evaluating"):
        try:
            result = evaluator.predict(row["audio_path"])

            # Convert label
            if result.label == "SPEECH":
                y_pred = 1
            elif result.label == "NONSPEECH":
                y_pred = 0
            else:
                continue  # Skip UNKNOWN

            # Convert ground truth
            y_true = 1 if row["label"] == "SPEECH" else 0

            predictions.append({
                "clip_id": row["clip_id"],
                "audio_path": row["audio_path"],
                "variant_type": row["variant_type"],
                "y_true": y_true,
                "y_pred": y_pred,
                "label_true": row["label"],
                "label_pred": result.label,
            })
        except Exception as e:
            print(f"\nError on {row['audio_path']}: {e}")
            continue

    pred_df = pd.DataFrame(predictions)

    print(f"\n[OK] Evaluated {len(pred_df)} samples successfully")
    print()

    # Calculate metrics
    print("="*80)
    print("RESULTS")
    print("="*80)

    # 1. Clip-level BA (majority vote)
    clip_agg = (
        pred_df.groupby("clip_id")
        .agg({
            "y_true": "first",
            "y_pred": lambda x: x.mode()[0] if len(x.mode()) > 0 else 0
        })
        .reset_index()
    )

    ba_clip = balanced_accuracy_score(clip_agg["y_true"], clip_agg["y_pred"])
    ci_lower, ci_upper = bootstrap_ci(
        clip_agg["y_true"].values,
        clip_agg["y_pred"].values,
        clip_ids=None,
        n_bootstrap=1000
    )

    print(f"\nClip-level BA: {ba_clip:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

    # 2. Per-condition BA
    print("\nPer-condition BA:")
    condition_results = {}

    for condition in pred_df["variant_type"].unique():
        cond_df = pred_df[pred_df["variant_type"] == condition]

        # Clip-level for this condition
        cond_clip_agg = (
            cond_df.groupby("clip_id")
            .agg({
                "y_true": "first",
                "y_pred": lambda x: x.mode()[0] if len(x.mode()) > 0 else 0
            })
            .reset_index()
        )

        if len(cond_clip_agg) > 0:
            ba = balanced_accuracy_score(
                cond_clip_agg["y_true"],
                cond_clip_agg["y_pred"]
            )
            ci_low, ci_high = bootstrap_ci(
                cond_clip_agg["y_true"].values,
                cond_clip_agg["y_pred"].values,
                n_bootstrap=1000
            )

            condition_results[condition] = {
                "ba": ba,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "n_clips": len(cond_clip_agg)
            }

            print(f"  {condition:12s}: {ba:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}]) n={len(cond_clip_agg)}")

    # 3. Macro-averaged BA across conditions
    macro_ba = np.mean([res["ba"] for res in condition_results.values()])
    print(f"\nMacro-BA (conditions): {macro_ba:.4f}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    pred_file = output_dir / "dev_predictions.parquet"
    pred_df.to_parquet(pred_file)
    print(f"\n[OK] Predictions saved to: {pred_file}")

    # Save metrics
    metrics = {
        "prompt": prompt,
        "ba_clip": float(ba_clip),
        "ba_clip_ci": [float(ci_lower), float(ci_upper)],
        "macro_ba_conditions": float(macro_ba),
        "per_condition": {
            k: {
                "ba": float(v["ba"]),
                "ci": [float(v["ci_lower"]), float(v["ci_upper"])],
                "n_clips": int(v["n_clips"])
            }
            for k, v in condition_results.items()
        },
        "n_samples": len(pred_df),
        "n_clips": len(clip_agg)
    }

    metrics_file = output_dir / "dev_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Metrics saved to: {metrics_file}")
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


def main():
    # Best prompt from OPRO optimization
    best_prompt = "Based on the audio file, is it SPEECH or NON-SPEECH?"

    # Paths
    script_dir = Path(__file__).parent.parent
    manifest_path = script_dir / "data" / "processed" / "conditions_final" / "conditions_manifest_split.parquet"
    output_dir = script_dir / "results" / "prompt_opt_local" / "dev_evaluation"

    evaluate_prompt(best_prompt, manifest_path, output_dir)


if __name__ == "__main__":
    main()
