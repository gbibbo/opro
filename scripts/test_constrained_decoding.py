"""
Test constrained decoding: compare SPEECH/NONSPEECH outputs with and without constraints.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import balanced_accuracy_score

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from qsm.models.qwen_audio import Qwen2AudioClassifier


def resolve_audio_path(raw_path: str, script_dir: Path) -> str:
    """Convert to absolute path and handle WSL/Windows conversion."""
    p = Path(str(raw_path))
    if not p.is_absolute():
        p = (script_dir / p).resolve()
    path_str = str(p)
    if sys.platform.startswith('linux') and ':' in path_str and '\\' in path_str:
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        path_str = f"/mnt/{drive}{rest}"
    elif sys.platform.startswith('linux'):
        path_str = path_str.replace('\\', '/')
    return path_str


def test_constrained_vs_unconstrained():
    """Compare constrained vs unconstrained decoding."""

    print("="*80)
    print("TESTING CONSTRAINED DECODING")
    print("="*80)
    print()

    # Load manifest
    script_dir = Path(__file__).parent.parent
    manifest_path = script_dir / "data" / "processed" / "conditions_final" / "conditions_manifest_split.parquet"
    manifest_df = pd.read_parquet(str(manifest_path))
    manifest_df = manifest_df[manifest_df["split"] == "dev"].reset_index(drop=True)

    # Convert paths
    manifest_df["audio_path"] = manifest_df["audio_path"].apply(
        lambda x: resolve_audio_path(x, script_dir)
    )

    # Sample 50 random files
    test_df = manifest_df.sample(n=50, random_state=42)

    print(f"Testing on {len(test_df)} random samples")
    print()

    # Best prompt from OPRO
    prompt = "Based on the audio file, is it SPEECH or NON-SPEECH?"

    # Test WITHOUT constrained decoding
    print("-"*80)
    print("1. WITHOUT CONSTRAINED DECODING (baseline)")
    print("-"*80)

    evaluator_unconstrained = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda",
        load_in_4bit=True,
        constrained_decoding=False,  # No constraints
    )
    evaluator_unconstrained.set_prompt(user_prompt=prompt)

    predictions_unconstrained = []
    for _, row in test_df.iterrows():
        try:
            result = evaluator_unconstrained.predict(row["audio_path"])
            y_pred = 1 if result.label == "SPEECH" else (0 if result.label == "NONSPEECH" else -1)
            y_true = 1 if row["label"] == "SPEECH" else 0

            predictions_unconstrained.append({
                "y_true": y_true,
                "y_pred": y_pred,
                "label_pred": result.label,
                "raw_output": result.raw_output,
            })
        except Exception as e:
            print(f"Error: {e}")

    pred_df_unconstrained = pd.DataFrame(predictions_unconstrained)

    # Calculate BA (excluding UNKNOWN)
    valid_mask = pred_df_unconstrained["y_pred"] != -1
    valid_preds = pred_df_unconstrained[valid_mask]

    ba_unconstrained = balanced_accuracy_score(
        valid_preds["y_true"],
        valid_preds["y_pred"]
    ) if len(valid_preds) > 0 else 0.0

    n_unknown = len(pred_df_unconstrained) - len(valid_preds)

    print(f"\n[RESULTS] Unconstrained:")
    print(f"  BA_clip: {ba_unconstrained:.4f}")
    print(f"  Valid predictions: {len(valid_preds)}/{len(pred_df_unconstrained)}")
    print(f"  UNKNOWN predictions: {n_unknown}")
    print()
    print("Sample outputs:")
    for i, row in pred_df_unconstrained.head(5).iterrows():
        print(f"  {i+1}. {row['label_pred']:10s} | {row['raw_output'][:50]}")
    print()

    # Unload model
    del evaluator_unconstrained
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # Test WITH constrained decoding
    print("-"*80)
    print("2. WITH CONSTRAINED DECODING")
    print("-"*80)

    evaluator_constrained = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda",
        load_in_4bit=True,
        constrained_decoding=True,  # Enable constraints
    )
    evaluator_constrained.set_prompt(user_prompt=prompt)

    predictions_constrained = []
    for _, row in test_df.iterrows():
        try:
            result = evaluator_constrained.predict(row["audio_path"])
            y_pred = 1 if result.label == "SPEECH" else (0 if result.label == "NONSPEECH" else -1)
            y_true = 1 if row["label"] == "SPEECH" else 0

            predictions_constrained.append({
                "y_true": y_true,
                "y_pred": y_pred,
                "label_pred": result.label,
                "raw_output": result.raw_output,
            })
        except Exception as e:
            print(f"Error: {e}")

    pred_df_constrained = pd.DataFrame(predictions_constrained)

    # Calculate BA (excluding UNKNOWN)
    valid_mask_c = pred_df_constrained["y_pred"] != -1
    valid_preds_c = pred_df_constrained[valid_mask_c]

    ba_constrained = balanced_accuracy_score(
        valid_preds_c["y_true"],
        valid_preds_c["y_pred"]
    ) if len(valid_preds_c) > 0 else 0.0

    n_unknown_c = len(pred_df_constrained) - len(valid_preds_c)

    print(f"\n[RESULTS] Constrained:")
    print(f"  BA_clip: {ba_constrained:.4f}")
    print(f"  Valid predictions: {len(valid_preds_c)}/{len(pred_df_constrained)}")
    print(f"  UNKNOWN predictions: {n_unknown_c}")
    print()
    print("Sample outputs:")
    for i, row in pred_df_constrained.head(5).iterrows():
        print(f"  {i+1}. {row['label_pred']:10s} | {row['raw_output'][:50]}")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Unconstrained BA: {ba_unconstrained:.4f} ({n_unknown} UNKNOWN)")
    print(f"Constrained BA:   {ba_constrained:.4f} ({n_unknown_c} UNKNOWN)")
    print(f"Improvement:      {ba_constrained - ba_unconstrained:+.4f}")
    print("="*80)


if __name__ == "__main__":
    test_constrained_vs_unconstrained()
