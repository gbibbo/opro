#!/usr/bin/env python3
"""
Evaluate fine-tuned model using DIRECT LOGITS (no generate).

This is faster and more stable than generate() for binary A/B classification.
We compute the forward pass once and extract logits for tokens A and B directly.

Advantages over generate():
1. Faster: No sampling/decoding overhead
2. More stable: Deterministic (no temperature/sampling issues)
3. Same result: For constrained A/B tasks, logits are sufficient
4. Enables calibration: Easy to apply temperature scaling

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
    """Get all token IDs that represent 'A' or 'B'."""
    # Tokenize 'A' and 'B' to get their token IDs
    ids_A = tokenizer.encode('A', add_special_tokens=False)
    ids_B = tokenizer.encode('B', add_special_tokens=False)

    print(f"Token IDs for 'A': {ids_A}")
    print(f"Token IDs for 'B': {ids_B}")

    return ids_A, ids_B


def evaluate_sample_logits(model, processor, audio_path, ids_A, ids_B, temperature=1.0):
    """
    Evaluate a single sample using direct logits (no generate).

    Args:
        model: The fine-tuned model
        processor: The Qwen2-Audio processor
        audio_path: Path to audio file
        ids_A: List of token IDs for 'A'
        ids_B: List of token IDs for 'B'
        temperature: Temperature for scaling logits (default 1.0 = no scaling)

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

    # Create prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": "Is this audio SPEECH (A) or NON-SPEECH (B)? Answer with a single letter:"}
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
    logits_A = logits[ids_A]  # Could be multiple tokens
    logits_B = logits[ids_B]

    # Apply temperature scaling
    logits_A = logits_A / temperature
    logits_B = logits_B / temperature

    # Aggregate if multiple tokens per option (take max)
    logit_A = logits_A.max().item()
    logit_B = logits_B.max().item()

    # Compute probabilities using softmax over {A, B}
    logit_diff = logit_A - logit_B
    prob_A = torch.sigmoid(torch.tensor(logit_diff)).item()
    prob_B = 1.0 - prob_A

    # Prediction
    prediction = 'A' if prob_A > prob_B else 'B'
    confidence = max(prob_A, prob_B)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'prob_A': prob_A,
        'prob_B': prob_B,
        'logit_A': logit_A,
        'logit_B': logit_B,
        'logit_diff': logit_diff
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate with direct logits (no generate)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint"
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
        "--output_csv",
        type=str,
        default=None,
        help="Path to save detailed predictions (optional)"
    )

    args = parser.parse_args()

    print("="*60)
    print("LOGIT-BASED EVALUATION (No Generate)")
    print("="*60)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    base_model_id = "Qwen/Qwen2-Audio-7B-Instruct"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    print(f"Model loaded on: {model.device}")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Get A/B token IDs
    ids_A, ids_B = get_ab_token_ids(processor.tokenizer)

    # Load test data
    print(f"\nLoading test data from {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    print(f"Test samples: {len(test_df)}")

    # Label mapping
    label_map = {'SPEECH': 'A', 'NONSPEECH': 'B'}

    # Evaluate
    results = []
    correct = 0
    total = 0

    print(f"\nEvaluating with temperature = {args.temperature}...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        audio_path = row['audio_path']
        ground_truth = row['label']
        ground_truth_token = label_map[ground_truth]

        result = evaluate_sample_logits(
            model, processor, audio_path,
            ids_A, ids_B,
            temperature=args.temperature
        )

        is_correct = (result['prediction'] == ground_truth_token)
        correct += is_correct
        total += 1

        results.append({
            'clip_id': row['clip_id'],
            'audio_path': audio_path,
            'ground_truth': ground_truth,
            'ground_truth_token': ground_truth_token,
            'prediction': result['prediction'],
            'correct': is_correct,
            'confidence': result['confidence'],
            'prob_A': result['prob_A'],
            'prob_B': result['prob_B'],
            'logit_A': result['logit_A'],
            'logit_B': result['logit_B'],
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
    print(f"  SPEECH (A):    {speech_results['correct'].sum()}/{len(speech_results)} = {speech_acc:.1f}%")
    print(f"  NONSPEECH (B): {nonspeech_results['correct'].sum()}/{len(nonspeech_results)} = {nonspeech_acc:.1f}%")

    print(f"\nConfidence statistics:")
    print(f"  Overall:  {results_df['confidence'].mean():.3f}")
    print(f"  Correct:  {correct_conf:.3f}")
    print(f"  Wrong:    {wrong_conf:.3f}" if not pd.isna(wrong_conf) else "  Wrong:    N/A (all correct)")
    print(f"  Gap:      {correct_conf - wrong_conf:.3f}" if not pd.isna(wrong_conf) else "  Gap:      N/A")

    # Logit difference statistics
    print(f"\nLogit difference (A - B) statistics:")
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
            print(f"  Ground truth: {error['ground_truth']} ({error['ground_truth_token']})")
            print(f"  Prediction:   {error['prediction']}")
            print(f"  Confidence:   {error['confidence']:.3f}")
            print(f"  Logit diff:   {error['logit_diff']:.3f}")
            print()

    # Save detailed predictions
    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
        print(f"\nDetailed predictions saved to: {args.output_csv}")

    print(f"\n{'='*60}")
    print(f"✓ Evaluation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
