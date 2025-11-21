#!/usr/bin/env python3
"""
Memory-Efficient Prompt Testing: Test Multiple Prompts Sequentially

Strategy: Load model once → test all prompts → unload properly
This avoids OOM issues by not keeping multiple models in memory.

Usage:
    python scripts/test_prompt_templates.py \
        --checkpoint checkpoints/ablations/LORA_attn_mlp/seed_42/final \
        --test_csv data/processed/grouped_split/dev_metadata.csv \
        --output_dir results/prompt_testing \
        --num_samples 10
"""

import argparse
import json
import pandas as pd
import torch
import gc
from pathlib import Path
from datetime import datetime
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import PeftModel
import sys

# Add scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))


# Prompt templates from OPRO
PROMPT_TEMPLATES = [
    # Baseline
    "Choose one:\nA) SPEECH (human voice)\nB) NONSPEECH (music/noise/silence/animals)\n\nAnswer with A or B ONLY.",

    # Emphasis variations
    "Listen carefully. Choose:\nA) SPEECH (human voice speaking)\nB) NONSPEECH (music, noise, silence, animals)\n\nAnswer A or B.",

    "Classify this audio:\nA) SPEECH - human voice, talking, speaking\nB) NONSPEECH - music, noise, environmental sounds, animals\n\nYour answer (A or B):",

    # Task framing
    "Your task: Detect if this audio contains speech.\nA) SPEECH (any human voice)\nB) NONSPEECH (no human voice)\n\nAnswer:",

    # Decision-focused
    "Does this audio contain human speech?\nA) Yes (SPEECH)\nB) No (NONSPEECH)\n\nAnswer A or B only.",

    # Feature-focused
    "Analyze the audio features. Is there human vocal activity?\nA) SPEECH detected\nB) NONSPEECH (music/noise/silence/other)\n\nAnswer:",

    # Detailed
    "Audio classification task:\n- A: SPEECH (human voice, talking, vocalizations)\n- B: NONSPEECH (music, environmental sounds, noise, silence, non-human)\n\nYour classification:",

    # Simple
    "A) SPEECH\nB) NONSPEECH\n\nWhich one?",

    # Question format
    "Is this audio speech or non-speech?\nA) SPEECH\nB) NONSPEECH\n\nAnswer:",

    # Instruction format
    "Identify the audio type. Select one:\nA) SPEECH (human speaking)\nB) NONSPEECH (other sounds)\n\nSelection:",
]


def get_ab_token_ids(tokenizer):
    """Get all token IDs that represent 'A' or 'B'."""
    ids_A = []
    ids_B = []

    # Variant 1: No space
    ids_A.extend(tokenizer.encode('A', add_special_tokens=False))
    ids_B.extend(tokenizer.encode('B', add_special_tokens=False))

    # Variant 2: Leading space
    ids_A_space = tokenizer.encode(' A', add_special_tokens=False)
    ids_B_space = tokenizer.encode(' B', add_special_tokens=False)

    # Add space variants (avoiding duplicates)
    for id_val in ids_A_space:
        if id_val not in ids_A:
            ids_A.append(id_val)
    for id_val in ids_B_space:
        if id_val not in ids_B:
            ids_B.append(id_val)

    return ids_A, ids_B


def evaluate_sample_logits(model, processor, audio_path, ids_A, ids_B, user_prompt):
    """Evaluate single sample with custom prompt."""
    import soundfile as sf

    # Load audio
    audio, sr = sf.read(audio_path)

    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Create conversation format
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

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward pass (no generation)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits for last position
    logits = outputs.logits[0, -1, :]

    # Get logits for A and B tokens
    logits_A = logits[ids_A]
    logits_B = logits[ids_B]

    # Aggregate using logsumexp
    logit_A = torch.logsumexp(logits_A, dim=0).item()
    logit_B = torch.logsumexp(logits_B, dim=0).item()

    # Compute probabilities
    logit_diff = logit_A - logit_B
    prob_A = torch.sigmoid(torch.tensor(logit_diff)).item()

    # Prediction
    prediction = 'A' if prob_A > 0.5 else 'B'

    return {
        'prediction': prediction,
        'confidence': max(prob_A, 1.0 - prob_A),
        'prob_A': prob_A
    }


def evaluate_prompt(model, processor, ids_A, ids_B, samples, prompt):
    """Evaluate a prompt on a set of samples."""
    correct = 0
    total = len(samples)

    for sample in samples:
        try:
            result = evaluate_sample_logits(
                model, processor,
                sample['audio_path'],
                ids_A, ids_B,
                prompt
            )

            if result['prediction'] == sample['ground_truth_token']:
                correct += 1
        except Exception as e:
            print(f"  Error on {sample['audio_path']}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Test multiple prompt templates")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned model checkpoint')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='CSV with test samples')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test per prompt (for speed)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEMORY-EFFICIENT PROMPT TESTING")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Test CSV: {args.test_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Samples per prompt: {args.num_samples}")

    # Load model ONCE
    print(f"\n[1/4] Loading model...")
    base_model_id = "Qwen/Qwen2-Audio-7B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )

    try:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model.eval()

        print(f"Model loaded on: {model.device}")

        # Freeze all parameters (no training)
        for param in model.parameters():
            param.requires_grad = False

    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model")
        print(f"Error: {e}")
        print("\nThis likely means insufficient VRAM. Try:")
        print("  1. Close other programs using GPU")
        print("  2. Run on CPU (very slow): add device_map='cpu'")
        print("  3. Use smaller model or fewer samples")
        return

    # Load processor
    print(f"\n[2/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Get token IDs
    ids_A, ids_B = get_ab_token_ids(processor.tokenizer)
    print(f"Token IDs: A={ids_A}, B={ids_B}")

    # Load data
    print(f"\n[3/4] Loading test data...")
    test_df = pd.read_csv(args.test_csv)
    print(f"Total samples: {len(test_df)}")

    # Sample subset
    if len(test_df) > args.num_samples:
        test_df = test_df.sample(n=args.num_samples, random_state=42)
        print(f"Using {args.num_samples} samples for testing")

    # Prepare samples
    samples = []
    for _, row in test_df.iterrows():
        label_map = {'SPEECH': 'A', 'NONSPEECH': 'B'}
        label_col = 'ground_truth' if 'ground_truth' in test_df.columns else 'label'
        samples.append({
            'audio_path': row['audio_path'],
            'ground_truth_token': label_map[row[label_col]]
        })

    print(f"  SPEECH:    {sum(1 for s in samples if s['ground_truth_token'] == 'A')}")
    print(f"  NONSPEECH: {sum(1 for s in samples if s['ground_truth_token'] == 'B')}")

    # Test all prompts
    print(f"\n[4/4] Testing {len(PROMPT_TEMPLATES)} prompt templates...")
    print("=" * 80)

    results = []
    best_accuracy = 0.0
    best_prompt = None

    for i, prompt in enumerate(PROMPT_TEMPLATES):
        print(f"\n[{i+1}/{len(PROMPT_TEMPLATES)}] Testing prompt:")
        print(f"  {prompt[:70]}..." if len(prompt) > 70 else f"  {prompt}")

        accuracy = evaluate_prompt(model, processor, ids_A, ids_B, samples, prompt)

        results.append({
            'prompt_id': i,
            'prompt': prompt,
            'accuracy': accuracy,
            'correct': int(accuracy * len(samples)),
            'total': len(samples)
        })

        print(f"  Accuracy: {accuracy:.1%} ({int(accuracy * len(samples))}/{len(samples)})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_prompt = prompt
            print(f"  [NEW BEST]")

    # Unload model properly
    print(f"\n[5/5] Cleaning up...")
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("Model unloaded, memory cleared")

    # Save results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('accuracy', ascending=False)

    print(f"\nTop 5 prompts:")
    for i, row in results_df.head(5).iterrows():
        print(f"{row['accuracy']:.1%} - {row['prompt'][:60]}...")

    print(f"\nBest prompt (accuracy: {best_accuracy:.1%}):")
    print(f"{best_prompt}")

    # Save to files
    results_file = output_dir / f"prompt_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")

    best_prompt_file = output_dir / "best_prompt.txt"
    best_prompt_file.write_text(best_prompt)
    print(f"Best prompt saved to: {best_prompt_file}")

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'best_accuracy': float(best_accuracy),
            'best_prompt': best_prompt,
            'num_prompts_tested': len(PROMPT_TEMPLATES),
            'num_samples': len(samples),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"Summary saved to: {summary_file}")

    print(f"\n{'='*80}")
    print("[DONE] Testing complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
