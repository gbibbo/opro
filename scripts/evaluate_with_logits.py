#!/usr/bin/env python3
"""
Evaluate fine-tuned model using DIRECT LOGITS (no generate).

This is faster and more stable than generate() for binary SPEECH/NONSPEECH classification.
We compute the forward pass once and extract logits for SPEECH and NONSPEECH tokens directly.

Advantages over generate():
1. Faster: No sampling/decoding overhead
2. More stable: Deterministic (no temperature/sampling issues)
3. Same result: For constrained binary tasks, logits are sufficient
4. Enables calibration: Easy to apply temperature scaling

IMPORTANT: This evaluation script uses the SAME prompt format as training:
"Does this audio contain human speech? Answer SPEECH or NONSPEECH."

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


def get_speech_nonspeech_token_ids(tokenizer):
    """
    Get all token IDs that represent 'SPEECH' or 'NONSPEECH'.

    Important: Handles both variants (with and without leading space)
    to avoid tokenization bias. Qwen tokenizer may produce different tokens
    depending on context.

    Returns:
        ids_SPEECH, ids_NONSPEECH: Lists of token IDs (may contain multiple variants)
    """
    # Tokenize 'SPEECH' and 'NONSPEECH' with and without leading space
    ids_SPEECH = []
    ids_NONSPEECH = []

    # Variant 1: No space
    ids_SPEECH.extend(tokenizer.encode('SPEECH', add_special_tokens=False))
    ids_NONSPEECH.extend(tokenizer.encode('NONSPEECH', add_special_tokens=False))

    # Variant 2: Leading space
    ids_SPEECH_space = tokenizer.encode(' SPEECH', add_special_tokens=False)
    ids_NONSPEECH_space = tokenizer.encode(' NONSPEECH', add_special_tokens=False)

    # Add space variants (avoiding duplicates)
    for id_val in ids_SPEECH_space:
        if id_val not in ids_SPEECH:
            ids_SPEECH.append(id_val)

    for id_val in ids_NONSPEECH_space:
        if id_val not in ids_NONSPEECH:
            ids_NONSPEECH.append(id_val)

    print(f"Token IDs for 'SPEECH' (including ' SPEECH'): {ids_SPEECH}")
    print(f"Token IDs for 'NONSPEECH' (including ' NONSPEECH'): {ids_NONSPEECH}")

    return ids_SPEECH, ids_NONSPEECH


def evaluate_sample_logits(model, processor, audio_path, ids_SPEECH, ids_NONSPEECH, temperature=1.0, user_prompt=None):
    """
    Evaluate a single sample using direct logits (no generate).

    Args:
        model: The fine-tuned model
        processor: The Qwen2-Audio processor
        audio_path: Path to audio file
        ids_SPEECH: List of token IDs for 'SPEECH'
        ids_NONSPEECH: List of token IDs for 'NONSPEECH'
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
        user_prompt = "Does this audio contain human speech? Answer SPEECH or NONSPEECH."

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

    # Extract logits for SPEECH and NONSPEECH tokens
    logits_SPEECH = logits[ids_SPEECH]  # Could be multiple tokens
    logits_NONSPEECH = logits[ids_NONSPEECH]

    # Apply temperature scaling
    logits_SPEECH = logits_SPEECH / temperature
    logits_NONSPEECH = logits_NONSPEECH / temperature

    # Aggregate multiple token variants using logsumexp (sum probabilities)
    # This is more robust than max() for handling tokenization variants
    # logsumexp([a,b]) ≈ log(exp(a) + exp(b)) = log(p_SPEECH + p_SPEECH_space)
    logit_SPEECH = torch.logsumexp(logits_SPEECH, dim=0).item()
    logit_NONSPEECH = torch.logsumexp(logits_NONSPEECH, dim=0).item()

    # Compute probabilities using softmax over {SPEECH, NONSPEECH}
    # After logsumexp, these represent the total probability mass for each option
    logit_diff = logit_SPEECH - logit_NONSPEECH
    prob_SPEECH = torch.sigmoid(torch.tensor(logit_diff)).item()
    prob_NONSPEECH = 1.0 - prob_SPEECH

    # Prediction
    prediction = 'SPEECH' if prob_SPEECH > prob_NONSPEECH else 'NONSPEECH'
    confidence = max(prob_SPEECH, prob_NONSPEECH)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'prob_SPEECH': prob_SPEECH,
        'prob_NONSPEECH': prob_NONSPEECH,
        'logit_SPEECH': logit_SPEECH,
        'logit_NONSPEECH': logit_NONSPEECH,
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

    print("="*60)
    print("LOGIT-BASED EVALUATION (No Generate)")
    print("="*60)

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    base_model_id = "Qwen/Qwen2-Audio-7B-Instruct"

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

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    print(f"Model loaded on: {model.device}")

    # Load processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Get SPEECH/NONSPEECH token IDs
    ids_SPEECH, ids_NONSPEECH = get_speech_nonspeech_token_ids(processor.tokenizer)

    # Load test data
    print(f"\nLoading test data from {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    print(f"Test samples: {len(test_df)}")

    # Determine label column
    label_col = 'ground_truth' if 'ground_truth' in test_df.columns else 'label'

    # No label mapping needed - model now outputs SPEECH/NONSPEECH directly

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
            ids_SPEECH, ids_NONSPEECH,
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
            'prob_SPEECH': result['prob_SPEECH'],
            'prob_NONSPEECH': result['prob_NONSPEECH'],
            'logit_SPEECH': result['logit_SPEECH'],
            'logit_NONSPEECH': result['logit_NONSPEECH'],
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
