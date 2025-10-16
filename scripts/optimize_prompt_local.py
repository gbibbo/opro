"""
Optimizador de Prompts Local Simplificado (Sin DSPy, Sin APIs).

Usa Qwen2.5-3B para generar candidatos y tu evaluador para scoring.
100% local, 100% gratis, código simple.

Usage:
    python scripts/optimize_prompt_local.py
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import local modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from qsm.models.qwen_audio import Qwen2AudioClassifier


def sanitize_prompt(prompt: str) -> tuple[str, bool]:
    """Sanitize and validate prompt."""
    forbidden_tokens = [
        '<|audio_bos|>', '<|AUDIO|>', '<|audio_eos|>',
        '<|im_start|>', '<|im_end|>',
        '<audio>', '</audio>',
    ]

    cleaned = prompt.strip()
    for token in forbidden_tokens:
        if token in cleaned:
            return cleaned, False

    if len(cleaned) < 10 or len(cleaned) > 300:
        return cleaned, False

    upper = cleaned.upper()
    has_speech = 'SPEECH' in upper
    has_nonspeech = 'NON-SPEECH' in upper or 'NONSPEECH' in upper

    if not (has_speech and has_nonspeech):
        return cleaned, False

    return cleaned, True


def score_prompt(prompt: str, evaluator: Qwen2AudioClassifier, manifest_df: pd.DataFrame,
                 subset_size: int = 150) -> dict:
    """Score a prompt on subset of dev set."""

    # Sanitize
    cleaned, is_valid = sanitize_prompt(prompt)
    if not is_valid:
        print(f"DEBUG: Prompt validation FAILED")
        print(f"  Prompt: {prompt[:100]}")
        print(f"  Cleaned: {cleaned[:100]}")
        upper = cleaned.upper()
        has_speech = 'SPEECH' in upper
        has_nonspeech = 'NON-SPEECH' in upper or 'NONSPEECH' in upper
        print(f"  has_speech={has_speech}, has_nonspeech={has_nonspeech}")
        print(f"  length={len(cleaned)}")
        return {"ba_clip": 0.0, "is_valid": False}

    # Subset stratified by variant type and label
    valid_types = manifest_df["variant_type"].dropna().unique().tolist()
    valid_labels = manifest_df["label"].dropna().unique().tolist()  # Use actual labels from manifest

    # Calculate samples per category (balance by type AND label)
    samples_per_category = max(1, subset_size // (len(valid_types) * len(valid_labels)))

    sampled_dfs = []
    for vtype in valid_types:
        for label in valid_labels:
            df_subset = manifest_df[
                (manifest_df["variant_type"] == vtype) &
                (manifest_df["label"] == label)
            ]
            if len(df_subset) >= samples_per_category:
                sampled_dfs.append(df_subset.sample(n=samples_per_category, random_state=42))
            elif len(df_subset) > 0:
                sampled_dfs.append(df_subset)

    subset_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    # Verify balance (check for at least 2 different labels)
    label_dist = subset_df["label"].value_counts().to_dict()
    if len(label_dist) < 2:
        print(f"[ERROR] Unbalanced subset: {label_dist}")
        return {"ba_clip": 0.0, "is_valid": False}

    # Evaluate
    evaluator.set_prompt(user_prompt=cleaned)

    predictions = []
    debug_labels = []
    for idx, row in subset_df.iterrows():
        try:
            result = evaluator.predict(row["audio_path"])

            # Debug: track what labels we're getting
            result_label = result.label if hasattr(result, 'label') else str(result)
            debug_labels.append(result_label)

            # Convert PredictionResult to int (0 or 1)
            if result.label == "SPEECH":
                y_pred = 1
            elif result.label == "NONSPEECH":
                y_pred = 0
            else:
                continue  # Skip UNKNOWN

            # Convert label to int if it's a string
            label = row["label"]
            if isinstance(label, str):
                label = 1 if label.upper() in ["SPEECH", "1"] else 0

            predictions.append({
                "clip_id": row["clip_id"],
                "y_true": label,
                "y_pred": y_pred,
            })
        except Exception as e:
            debug_labels.append(f"ERROR:{str(e)[:20]}")
            pass

    if len(predictions) == 0:
        print(f"DEBUG: No valid predictions! Evaluated {len(subset_df)} samples but got 0 predictions")
        # Show what labels we got
        from collections import Counter
        label_counts = Counter(debug_labels)
        print(f"DEBUG: Actual labels returned by evaluator:")
        for lbl, cnt in label_counts.most_common(10):
            print(f"  '{lbl}': {cnt} times")
        return {"ba_clip": 0.0, "is_valid": False}

    pred_df = pd.DataFrame(predictions)

    # Clip-level BA (majority vote)
    clip_agg = (
        pred_df.groupby("clip_id")
        .agg({"y_true": "first", "y_pred": lambda x: x.mode()[0] if len(x.mode()) > 0 else 0})
        .reset_index()
    )
    ba_clip = balanced_accuracy_score(clip_agg["y_true"], clip_agg["y_pred"])

    return {"ba_clip": ba_clip, "is_valid": True}


def generate_candidates(optimizer_llm, optimizer_tokenizer, baseline_prompt: str,
                        history: list, n_candidates: int = 6) -> list[str]:
    """Generate prompt candidates using Qwen2.5-3B."""

    # Build meta-prompt with explicit examples
    best_score = history[0]['score'] if history else 0.0
    meta_prompt = f"""You are a prompt optimization assistant. Generate EXACTLY {n_candidates} prompt variations for audio classification.

TASK: Given an audio file, classify it as either SPEECH or NON-SPEECH.

CURRENT BEST PROMPT (score={best_score:.3f}):
"{baseline_prompt}"

REQUIREMENTS for each prompt:
- Must contain the words "SPEECH" and "NON-SPEECH" (or "NONSPEECH")
- Must ask the model to classify audio
- Length: 10-300 characters
- NO special tokens like <|audio_bos|> or <|AUDIO|>

EXAMPLES of good prompts:
1. "Does this audio contain speech? Answer SPEECH or NON-SPEECH."
2. "Classify this audio: SPEECH or NONSPEECH?"
3. "Is there human speech in this audio? Reply SPEECH or NON-SPEECH only."

Now generate {n_candidates} NEW prompt variations (different from baseline). Output one prompt per line, NO numbers, NO quotes:"""

    # Generate
    inputs = optimizer_tokenizer(meta_prompt, return_tensors="pt").to(optimizer_llm.device)

    with torch.no_grad():
        outputs = optimizer_llm.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=optimizer_tokenizer.eos_token_id,
        )

    generated_text = optimizer_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove input prompt
    if generated_text.startswith(meta_prompt):
        generated_text = generated_text[len(meta_prompt):].strip()

    # Parse candidates (one per line)
    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
    candidates = []

    for line in lines[:n_candidates * 3]:  # Check more lines to find valid ones
        # Remove numbering if present
        if line and line[0].isdigit() and '.' in line[:5]:
            line = line.split('.', 1)[1].strip()

        # Remove quotes if present
        line = line.strip('"\'')

        # Skip meta-instructions or too-long lines
        if not line or len(line) > 300:
            continue

        # Skip lines that are meta-commentary
        if line.lower().startswith(('here are', 'to ensure', 'do not', 'these are')):
            continue

        if line:
            candidates.append(line)

        if len(candidates) >= n_candidates:
            break

    return candidates[:n_candidates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--n_iterations", type=int, default=3)
    parser.add_argument("--n_candidates", type=int, default=6)
    parser.add_argument("--subset_size", type=int, default=150)
    parser.add_argument("--output_dir", default="results/prompt_opt_local")
    parser.add_argument("--baseline", default="Does this audio contain human speech?\nReply with ONLY one word: SPEECH or NON-SPEECH.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("PROMPT OPTIMIZER - 100% LOCAL")
    print("="*60)
    print(f"Optimizer: {args.optimizer_model}")
    print(f"Iterations: {args.n_iterations}")
    print(f"Candidates/iter: {args.n_candidates}")
    print(f"Subset size: {args.subset_size}")
    print("="*60)

    # Load manifest
    print("\nLoading manifest...")

    # Get script directory and find data
    script_dir = Path(__file__).parent.parent
    manifest_path = script_dir / "data" / "processed" / "conditions_final" / "conditions_manifest_split.parquet"

    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found at: {manifest_path}")
        sys.exit(1)

    manifest_df = pd.read_parquet(str(manifest_path))
    manifest_df = manifest_df[manifest_df["split"] == "dev"].reset_index(drop=True)

    # Convert paths to absolute and handle WSL/Windows
    def resolve_audio_path(raw_path: str) -> str:
        """Convert to absolute path and handle WSL/Windows conversion."""
        p = Path(str(raw_path))

        # Make absolute if relative
        if not p.is_absolute():
            p = (script_dir / p).resolve()

        # Convert Windows path to WSL if needed (C:\ -> /mnt/c/)
        path_str = str(p)
        if sys.platform.startswith('linux') and ':' in path_str and '\\' in path_str:
            # Windows path in WSL - convert C:\... -> /mnt/c/...
            drive = path_str[0].lower()
            rest = path_str[2:].replace('\\', '/')
            path_str = f"/mnt/{drive}{rest}"
        elif sys.platform.startswith('linux'):
            # Already linux path, ensure forward slashes
            path_str = path_str.replace('\\', '/')

        return path_str

    manifest_df["audio_path"] = manifest_df["audio_path"].apply(resolve_audio_path)

    # Verify files exist
    manifest_df["file_exists"] = manifest_df["audio_path"].apply(lambda x: Path(x).exists())
    missing = manifest_df[~manifest_df["file_exists"]]
    if len(missing) > 0:
        print(f"[WARNING] {len(missing)} audio files not found:")
        print(missing[["audio_path"]].head(10).to_string(index=False))
        print(f"\nRemoving {len(missing)} missing files from manifest...")
        manifest_df = manifest_df[manifest_df["file_exists"]].drop(columns=["file_exists"]).reset_index(drop=True)
    else:
        manifest_df = manifest_df.drop(columns=["file_exists"])

    print(f"[OK] {len(manifest_df)} dev variants")

    # Evaluate baseline (load evaluator ONLY)
    print(f"\n{'='*60}")
    print("BASELINE")
    print("="*60)
    print(f"Prompt: {args.baseline}")
    print("\nLoading evaluator (Qwen2-Audio)...")
    evaluator = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda",
        load_in_4bit=True,
    )
    print("[OK] Evaluator loaded")

    # Smoke test: verify evaluator can load audio files
    print("\nRunning smoke test on 5 audio files...")
    smoke_samples = manifest_df.sample(n=min(5, len(manifest_df)), random_state=42)
    smoke_ok = 0
    evaluator.set_prompt(user_prompt=args.baseline)

    for idx, row in smoke_samples.iterrows():
        try:
            result = evaluator.predict(row["audio_path"])
            if hasattr(result, 'label'):
                print(f"  OK: {row['audio_path'][-60:]} -> {result.label}")
                smoke_ok += 1
            else:
                print(f"  WARN: {row['audio_path'][-60:]} -> {result} (no label attr)")
        except Exception as e:
            print(f"  ERROR: {row['audio_path'][-60:]} -> {str(e)[:50]}")

    if smoke_ok == 0:
        print("\n[FATAL] Smoke test failed! Evaluator cannot load ANY audio files.")
        print("Check paths and file permissions.")
        sys.exit(1)

    print(f"\n[OK] Smoke test passed: {smoke_ok}/{len(smoke_samples)} files loaded successfully")

    baseline_result = score_prompt(args.baseline, evaluator, manifest_df, args.subset_size)
    baseline_score = baseline_result["ba_clip"]
    print(f"BA_clip: {baseline_score:.4f}")

    # Unload evaluator before loading optimizer
    print("\nUnloading evaluator...")
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)

    # Load optimizer LLM
    print(f"\nLoading optimizer ({args.optimizer_model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.optimizer_model, trust_remote_code=True)

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    optimizer_llm = AutoModelForCausalLM.from_pretrained(
        args.optimizer_model,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    optimizer_llm.eval()
    print("[OK] Optimizer loaded")

    # History
    history = [{"prompt": args.baseline, "score": baseline_score}]
    best_prompt = args.baseline
    best_score = baseline_score

    # Optimization loop
    start_time = datetime.now()

    evaluator = None  # Track evaluator state

    for iteration in range(1, args.n_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{args.n_iterations}")
        print("="*60)

        # Generate candidates
        print(f"\nGenerating {args.n_candidates} candidates...")

        # Unload evaluator if loaded
        if evaluator is not None:
            del evaluator
            evaluator = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)

        candidates = generate_candidates(
            optimizer_llm, tokenizer, best_prompt, history, args.n_candidates
        )

        print(f"Generated {len(candidates)} candidates")
        for i, c in enumerate(candidates, 1):
            print(f"  {i}. {c[:80]}...")

        # Unload optimizer before loading evaluator
        print("\nUnloading optimizer...")
        del optimizer_llm
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)

        # Reload evaluator
        print("Loading evaluator...")
        evaluator = Qwen2AudioClassifier(
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
            device="cuda",
            load_in_4bit=True,
        )

        # Evaluate candidates
        print("\nEvaluating candidates...")
        for i, candidate in enumerate(candidates, 1):
            result = score_prompt(candidate, evaluator, manifest_df, args.subset_size)

            if result["is_valid"]:
                score = result["ba_clip"]
                print(f"  {i}. BA={score:.4f} | {candidate[:60]}...")

                history.append({"prompt": candidate, "score": score})

                if score > best_score:
                    best_score = score
                    best_prompt = candidate
                    print(f"    [NEW BEST!] (+{score - baseline_score:.4f})")
            else:
                print(f"  {i}. INVALID | {candidate[:60]}...")

        # Sort history
        history = sorted(history, key=lambda x: x["score"], reverse=True)

        print(f"\nBest so far: BA={best_score:.4f} (+{best_score - baseline_score:.4f})")

        # Unload evaluator before next iteration
        del evaluator
        evaluator = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)

        # Reload optimizer for next iteration (if not last)
        if iteration < args.n_iterations:
            print("\nReloading optimizer...")
            tokenizer = AutoTokenizer.from_pretrained(args.optimizer_model, trust_remote_code=True)
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            optimizer_llm = AutoModelForCausalLM.from_pretrained(
                args.optimizer_model,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
            optimizer_llm.eval()

    duration = (datetime.now() - start_time).total_seconds()

    # Final summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"\nBaseline: {baseline_score:.4f}")
    print(f"Best:     {best_score:.4f}")
    print(f"Improvement: +{best_score - baseline_score:.4f}")
    print(f"\nBest prompt:")
    print(f'  "{best_prompt}"')

    # Save results
    results = {
        "baseline_prompt": args.baseline,
        "baseline_score": baseline_score,
        "best_prompt": best_prompt,
        "best_score": best_score,
        "improvement": best_score - baseline_score,
        "duration_seconds": duration,
        "history": history[:10],  # Top 10
        "config": vars(args),
    }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    prompt_file = output_dir / "best_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(best_prompt)

    print(f"\n[OK] Results: {results_file}")
    print(f"[OK] Best prompt: {prompt_file}")
    print("\n" + "="*60)
    print("DONE! ($0 cost, 100% local)")
    print("="*60)


if __name__ == "__main__":
    main()
