#!/usr/bin/env python3
"""
Lightweight Prompt Testing - Extreme Memory Optimization

This version uses maximum CPU offloading and minimal batch processing
to work on systems with limited VRAM/RAM.

Usage:
    python scripts/test_prompt_lightweight.py \
        --checkpoint checkpoints/ablations/LORA_attn_mlp/seed_42/final \
        --test_csv data/processed/grouped_split/dev_metadata.csv \
        --num_samples 10
"""

import argparse
import json
import pandas as pd
import torch
import gc
import os
from pathlib import Path
from datetime import datetime
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel
import soundfile as sf

# Prompt templates (same as before)
PROMPT_TEMPLATES = [
    "Is this audio SPEECH (A) or NON-SPEECH (B)? Answer with a single letter:",

    "Is this audio speech or non-speech?\nA) SPEECH\nB) NONSPEECH\n\nAnswer:",

    "Listen carefully. Choose:\nA) SPEECH (human voice speaking)\nB) NONSPEECH (music, noise, silence, animals)\n\nAnswer A or B.",

    "Does this audio contain human speech?\nA) Yes (SPEECH)\nB) No (NONSPEECH)\n\nAnswer A or B only.",

    "A) SPEECH\nB) NONSPEECH\n\nWhich one?",
]


def get_ab_token_ids(tokenizer):
    """Get all token IDs that represent 'A' or 'B'."""
    ids_A = []
    ids_B = []

    ids_A.extend(tokenizer.encode('A', add_special_tokens=False))
    ids_B.extend(tokenizer.encode('B', add_special_tokens=False))

    ids_A_space = tokenizer.encode(' A', add_special_tokens=False)
    ids_B_space = tokenizer.encode(' B', add_special_tokens=False)

    for id_val in ids_A_space:
        if id_val not in ids_A:
            ids_A.append(id_val)
    for id_val in ids_B_space:
        if id_val not in ids_B:
            ids_B.append(id_val)

    return ids_A, ids_B


def evaluate_sample_logits(model, processor, audio_path, ids_A, ids_B, user_prompt):
    """Evaluate single sample with extreme memory optimization."""
    # Load audio
    audio, sr = sf.read(audio_path)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Create conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    # Process
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audios=[audio], sampling_rate=sr, return_tensors="pt", padding=True)

    # Move to device (model handles device mapping)
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits
    logits = outputs.logits[0, -1, :]

    # Get A/B logits
    logits_A = logits[ids_A]
    logits_B = logits[ids_B]

    # Aggregate
    logit_A = torch.logsumexp(logits_A, dim=0).item()
    logit_B = torch.logsumexp(logits_B, dim=0).item()

    # Compute probability
    logit_diff = logit_A - logit_B
    prob_A = torch.sigmoid(torch.tensor(logit_diff)).item()

    # Prediction
    prediction = 'A' if prob_A > 0.5 else 'B'

    # Clean up immediately
    del outputs, inputs, logits
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'prediction': prediction,
        'prob_A': prob_A
    }


def load_model_extreme_offload(checkpoint_path):
    """Load model with maximum CPU offloading to minimize VRAM/RAM usage."""
    base_model_id = "Qwen/Qwen2-Audio-7B-Instruct"

    print("Loading with 8-bit quantization and aggressive CPU offloading...")

    # Use 8-bit instead of 4-bit (more stable on low memory)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Maximum offloading
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory={0: "2GB", "cpu": "4GB"},  # Very conservative limits for 8GB system
        offload_folder="offload_tmp",  # Use disk offloading
    )

    # Load LoRA
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    return model


def main():
    parser = argparse.ArgumentParser(description="Lightweight prompt testing")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples (use small number for low memory)')
    parser.add_argument('--output_dir', type=str, default='results/prompt_test_lightweight')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LIGHTWEIGHT PROMPT TESTING (Extreme Memory Optimization)")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Samples: {args.num_samples}")
    print(f"\nNOTE: Using 8-bit quantization + CPU offloading for low memory systems")

    # Load model
    print(f"\n[1/4] Loading model (this may take 5-10 minutes)...")
    try:
        model = load_model_extreme_offload(args.checkpoint)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"\nERROR: Could not load model")
        print(f"Error: {e}")
        print("\nIf you see 'Killed', your system ran out of memory.")
        print("Try one of these solutions:")
        print("  1. Reduce --num_samples to 5")
        print("  2. Close all other programs")
        print("  3. Run on a machine with more RAM (>32GB recommended)")
        print("  4. Use cloud GPU (Colab, Lambda, RunPod)")
        return

    # Load processor
    print(f"\n[2/4] Loading processor...")
    base_model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    ids_A, ids_B = get_ab_token_ids(processor.tokenizer)
    print(f"Token IDs: A={ids_A}, B={ids_B}")

    # Load data
    print(f"\n[3/4] Loading test data...")
    test_df = pd.read_csv(args.test_csv)

    if len(test_df) > args.num_samples:
        test_df = test_df.sample(n=args.num_samples, random_state=42)

    print(f"Using {len(test_df)} samples")

    # Prepare samples
    samples = []
    for _, row in test_df.iterrows():
        label_map = {'SPEECH': 'A', 'NONSPEECH': 'B'}
        label_col = 'ground_truth' if 'ground_truth' in test_df.columns else 'label'
        samples.append({
            'audio_path': row['audio_path'],
            'ground_truth_token': label_map[row[label_col]]
        })

    # Test prompts
    print(f"\n[4/4] Testing {len(PROMPT_TEMPLATES)} prompts...")
    print("="*80)

    results = []

    for i, prompt in enumerate(PROMPT_TEMPLATES):
        print(f"\n[{i+1}/{len(PROMPT_TEMPLATES)}] Testing:")
        print(f"  {prompt[:60]}..." if len(prompt) > 60 else f"  {prompt}")

        correct = 0
        for j, sample in enumerate(samples):
            try:
                result = evaluate_sample_logits(
                    model, processor,
                    sample['audio_path'],
                    ids_A, ids_B,
                    prompt
                )

                if result['prediction'] == sample['ground_truth_token']:
                    correct += 1

                # Progress indicator
                if (j + 1) % 5 == 0:
                    print(f"    Progress: {j+1}/{len(samples)}")

            except Exception as e:
                print(f"  Error on sample {j}: {e}")
                continue

        accuracy = correct / len(samples)

        results.append({
            'prompt_id': i,
            'prompt': prompt,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(samples)
        })

        print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(samples)})")

    # Cleanup
    print(f"\n[5/5] Cleaning up...")
    del model, processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Clean up offload folder
    if os.path.exists("offload_tmp"):
        import shutil
        shutil.rmtree("offload_tmp")

    print("Memory cleared")

    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)

    for _, row in results_df.iterrows():
        print(f"{row['accuracy']:.1%} - {row['prompt'][:50]}...")

    best = results_df.iloc[0]
    print(f"\nBest prompt ({best['accuracy']:.1%}):")
    print(best['prompt'])

    # Save
    results_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_file, index=False)

    best_file = output_dir / "best_prompt.txt"
    best_file.write_text(best['prompt'])

    print(f"\nResults saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
