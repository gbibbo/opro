#!/usr/bin/env python3
"""
Evaluate fine-tuned model using DIRECT LOGITS (no generate).

This is faster and more stable than generate() for binary A/B classification.
We compute the forward pass once and extract logits for A and B tokens directly.

Advantages over generate():
1. Faster: No sampling/decoding overhead
2. More stable: Deterministic (no temperature/sampling issues)
3. Same result: For constrained binary tasks, logits are sufficient
4. Enables calibration: Easy to apply temperature scaling

CRITICAL: This evaluation script uses the SAME prompt format as training:

"Choose one:
A) SPEECH (human voice)
B) NONSPEECH (music/noise/silence/animals)

Answer with A or B ONLY."

Model outputs token "A" (mapped to SPEECH) or "B" (mapped to NONSPEECH).

References:
- Guo et al. (2017): "On Calibration of Modern Neural Networks"
- Platt scaling / temperature scaling for binary classification
"""

import argparse
import pandas as pd
import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel


def get_ab_token_ids(tokenizer):
    """
    Get token IDs for 'A' and 'B'.

    CRITICAL: Model was trained to output "A" (for SPEECH) or "B" (for NONSPEECH).
    We extract these tokens to match the training format exactly.

    This matches the training prompt:
    "Choose one:
     A) SPEECH (human voice)
     B) NONSPEECH (music/noise/silence/animals)

     Answer with A or B ONLY."

    Returns:
        ids_A, ids_B: Lists containing single token IDs
    """
    # Get tokenizations for A and B
    tokens_a = tokenizer.encode('A', add_special_tokens=False)
    tokens_b = tokenizer.encode('B', add_special_tokens=False)

    print(f"Tokenization of 'A': {tokens_a}")
    print(f"Tokenization of 'B': {tokens_b}")

    # Use first token (should be single token anyway)
    id_A = tokens_a[0]
    id_B = tokens_b[0]

    # Return as single-element lists for compatibility
    ids_A = [id_A]
    ids_B = [id_B]

    print(f"Using A/B tokens to match training format:")
    print(f"  A token (SPEECH): {id_A}")
    print(f"  B token (NONSPEECH): {id_B}")

    # Verify no overlap
    if id_A == id_B:
        raise ValueError(f"ERROR: A and B have the same token: {id_A}")

    return ids_A, ids_B


def evaluate_sample_logits(model, processor, audio_path, ids_A, ids_B, temperature=1.0, user_prompt=None):
    """
    Evaluate a single sample using direct logits (no generate).

    Args:
        model: The fine-tuned model
        processor: The Qwen2-Audio processor
        audio_path: Path to audio file
        ids_A: List of token IDs for 'A' (maps to SPEECH)
        ids_B: List of token IDs for 'B' (maps to NONSPEECH)
        temperature: Temperature for scaling logits (default 1.0 = no scaling)
        user_prompt: Custom prompt text (default: matches training prompt)

    Returns:
        dict with prediction, confidence, and raw logits
    """
    # Load audio
    audio, sr = sf.read(audio_path)
    target_sr = processor.feature_extractor.sampling_rate

    # Resample if needed
    if sr != target_sr:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(torch.tensor(audio)).numpy()

    # Use custom prompt or default (MUST MATCH TRAINING PROMPT)
    if user_prompt is None:
        user_prompt = """Choose one:
A) SPEECH (human voice)
B) NONSPEECH (music/noise/silence/animals)

Answer with A or B ONLY."""

    # Create prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]

    # Process
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text_prompt],
        audio=[audio],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward pass (no generate)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits for the LAST token position (where answer would be)
    # Shape: (batch_size, seq_len, vocab_size)
    logits = outputs.logits[0, -1, :]  # Last position

    # Extract logits for A and B tokens
    logits_A = logits[ids_A]  # Token A (SPEECH)
    logits_B = logits[ids_B]  # Token B (NONSPEECH)

    # Apply temperature scaling
    logits_A = logits_A / temperature
    logits_B = logits_B / temperature

    # Aggregate (should be single token, but use logsumexp for consistency)
    logit_A = torch.logsumexp(logits_A, dim=0).item()
    logit_B = torch.logsumexp(logits_B, dim=0).item()

    # Compute probabilities using softmax over {A, B}
    logit_diff = logit_A - logit_B
    prob_A = torch.sigmoid(torch.tensor(logit_diff)).item()
    prob_B = 1.0 - prob_A

    # Map A/B to SPEECH/NONSPEECH for output
    prediction = 'SPEECH' if prob_A > prob_B else 'NONSPEECH'
    confidence = max(prob_A, prob_B)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'prob_A': prob_A,
        'prob_B': prob_B,
        'prob_SPEECH': prob_A,  # For compatibility
        'prob_NONSPEECH': prob_B,  # For compatibility
        'logit_A': logit_A,
        'logit_B': logit_B,
        'logit_SPEECH': logit_A,  # For compatibility
        'logit_NONSPEECH': logit_B,  # For compatibility
        'logit_diff': logit_diff
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate with direct logits (no generate)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned checkpoint (required unless --no-lora)"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Evaluate base model without LoRA adapter (baseline)"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Path to test metadata CSV"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for logit scaling (1.0 = no scaling)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt text (default: standard A/B prompt)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save detailed predictions (optional)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.no_lora and args.checkpoint is None:
        parser.error("--checkpoint is required unless --no-lora is specified")

    print("="*60)
    if args.no_lora:
        print("LOGIT-BASED EVALUATION - BASE MODEL (No LoRA)")
    else:
        print("LOGIT-BASED EVALUATION - FINETUNED MODEL")
    print("="*60)

    # Load model
    base_model_id = "Qwen/Qwen2-Audio-7B-Instruct"

    if args.no_lora:
        print(f"\nLoading BASE model (no LoRA): {base_model_id}")
    else:
        print(f"\nLoading finetuned model from {args.checkpoint}")

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapter only if not using base model
    if not args.no_lora:
        model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    print(f"Model loaded on: {model.device}")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Get A/B token IDs (matches training format)
    ids_A, ids_B = get_ab_token_ids(processor.tokenizer)

    # Load test data
    print(f"\nLoading test data from {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    print(f"Test samples: {len(test_df)}")

    # Determine label column
    label_col = 'ground_truth' if 'ground_truth' in test_df.columns else 'label'

    # Evaluate
    results = []
    correct = 0
    total = 0

    print(f"\nEvaluating with temperature = {args.temperature}...")
    if args.prompt:
        print(f"Using custom prompt: {args.prompt[:80]}..." if len(args.prompt) > 80 else f"Using custom prompt: {args.prompt}")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        audio_path = row['audio_path']
        # Prepend 'data/' if path doesn't start with it
        if not audio_path.startswith('data/'):
            audio_path = 'data/' + audio_path
        ground_truth = row[label_col]

        result = evaluate_sample_logits(
            model, processor, audio_path,
            ids_A, ids_B,
            temperature=args.temperature,
            user_prompt=args.prompt
        )

        is_correct = (result['prediction'] == ground_truth)
        correct += is_correct
        total += 1

        results.append({
            'clip_id': row['clip_id'],
            'audio_path': audio_path,
            'ground_truth': ground_truth,
            'prediction': result['prediction'],
            'correct': is_correct,
            'confidence': result['confidence'],
            'prob_A': result['prob_A'],
            'prob_B': result['prob_B'],
            'prob_SPEECH': result['prob_SPEECH'],  # Compatibility
            'prob_NONSPEECH': result['prob_NONSPEECH'],  # Compatibility
            'logit_A': result['logit_A'],
            'logit_B': result['logit_B'],
            'logit_SPEECH': result['logit_SPEECH'],  # Compatibility
            'logit_NONSPEECH': result['logit_NONSPEECH'],  # Compatibility
            'logit_diff': result['logit_diff']
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate metrics
    accuracy = correct / total * 100

    # Per-class accuracy
    speech_results = results_df[results_df['ground_truth'] == 'SPEECH']
    nonspeech_results = results_df[results_df['ground_truth'] == 'NONSPEECH']

    speech_acc = speech_results['correct'].mean() * 100 if len(speech_results) > 0 else 0
    nonspeech_acc = nonspeech_results['correct'].mean() * 100 if len(nonspeech_results) > 0 else 0

    # Confidence statistics
    correct_conf = results_df[results_df['correct']]['confidence'].mean()
    wrong_conf = results_df[~results_df['correct']]['confidence'].mean()

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"\nPer-class accuracy:")
    print(f"  SPEECH:    {speech_results['correct'].sum()}/{len(speech_results)} = {speech_acc:.1f}%")
    print(f"  NONSPEECH: {nonspeech_results['correct'].sum()}/{len(nonspeech_results)} = {nonspeech_acc:.1f}%")

    print(f"\nConfidence statistics:")
    print(f"  Overall:  {results_df['confidence'].mean():.3f}")
    print(f"  Correct:  {correct_conf:.3f}")
    print(f"  Wrong:    {wrong_conf:.3f}" if not pd.isna(wrong_conf) else "  Wrong:    N/A (all correct)")
    print(f"  Gap:      {correct_conf - wrong_conf:.3f}" if not pd.isna(wrong_conf) else "  Gap:      N/A")

    # Logit difference statistics
    print(f"\nLogit difference (SPEECH - NONSPEECH) statistics:")
    print(f"  Mean: {results_df['logit_diff'].mean():.3f}")
    print(f"  Std:  {results_df['logit_diff'].std():.3f}")
    print(f"  Min:  {results_df['logit_diff'].min():.3f}")
    print(f"  Max:  {results_df['logit_diff'].max():.3f}")

    # Show errors
    errors = results_df[~results_df['correct']]
    if len(errors) > 0:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)} total)")
        print(f"{'='*60}\n")

        for _, error in errors.iterrows():
            print(f"clip_id: {error['clip_id']}")
            print(f"  Ground truth: {error['ground_truth']}")
            print(f"  Prediction:   {error['prediction']}")
            print(f"  Confidence:   {error['confidence']:.3f}")
            print(f"  Logit diff:   {error['logit_diff']:.3f}")
            print()

    # Save detailed predictions
    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nDetailed predictions saved to: {args.output_csv}")

    print(f"\n{'='*60}")
    print(f"[DONE] Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
