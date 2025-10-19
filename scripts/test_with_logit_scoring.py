"""Evaluate model using direct logit scoring (no generation).

This approach is faster and more stable than generate():
1. Forward pass with prompt
2. Extract logits for tokens 'A' and 'B' at first output position
3. Apply softmax to get probabilities
4. Optionally apply temperature scaling for calibration
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import torch
import soundfile as sf
from transformers import (
    Qwen2AudioForConditionalGeneration,
    Qwen2AudioProcessor,
)
from peft import PeftModel
import numpy as np

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


@dataclass
class LogitScoreResult:
    """Results from logit-based scoring."""

    filename: str
    true_label: str
    predicted_label: str
    prob_a: float
    prob_b: float
    logit_a: float
    logit_b: float
    correct: bool


def get_ab_token_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Get all token IDs that represent 'A' and 'B'."""

    def get_single_token_variants(char: str) -> List[int]:
        variants = [char, f" {char}", f"\n{char}"]
        valid_ids = []
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                decoded = tokenizer.decode([ids[0]])
                if char in decoded.upper():
                    valid_ids.append(ids[0])
        return list(set(valid_ids))

    ids_a = get_single_token_variants("A")
    ids_b = get_single_token_variants("B")

    return ids_a, ids_b


def score_with_logits(
    model,
    processor,
    audio_path: Path,
    question: str,
    ids_a: List[int],
    ids_b: List[int],
    temperature: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Score audio using direct logit extraction.

    Args:
        model: Fine-tuned model
        processor: Qwen2AudioProcessor
        audio_path: Path to audio file
        question: Question prompt
        ids_a: Token IDs for 'A'
        ids_b: Token IDs for 'B'
        temperature: Temperature for calibration (1.0 = no scaling)

    Returns:
        prob_a, prob_b, logit_a, logit_b
    """
    # Load audio
    audio, sr = sf.read(audio_path)
    target_sr = processor.feature_extractor.sampling_rate

    # Resample if needed
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Create conversation
    conversation = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": question}]}]

    # Apply chat template
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    # Process inputs
    inputs = processor(
        text=text,
        audio=[audio],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True,
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass (no generation)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get logits at first output position
    # Shape: [batch_size, seq_len, vocab_size]
    logits = outputs.logits[0, -1, :]  # Last position of sequence

    # Extract logits for A and B tokens
    logits_a = logits[ids_a]  # Shape: [num_a_variants]
    logits_b = logits[ids_b]  # Shape: [num_b_variants]

    # Aggregate logits (take max or mean)
    # Max is more interpretable: "best token variant for A/B"
    logit_a = logits_a.max().item()
    logit_b = logits_b.max().item()

    # Apply temperature scaling
    logit_a_scaled = logit_a / temperature
    logit_b_scaled = logit_b / temperature

    # Compute probabilities via softmax over {A, B}
    logits_ab = torch.tensor([logit_a_scaled, logit_b_scaled])
    probs_ab = torch.softmax(logits_ab, dim=0)

    prob_a = probs_ab[0].item()
    prob_b = probs_ab[1].item()

    return prob_a, prob_b, logit_a, logit_b


def evaluate_with_logit_scoring(
    model_path: Path,
    test_csv: Path,
    temperature: float = 1.0,
) -> List[LogitScoreResult]:
    """
    Evaluate model using logit scoring.

    Args:
        model_path: Path to fine-tuned model checkpoint
        test_csv: Path to test metadata CSV
        temperature: Temperature for calibration

    Returns:
        List of LogitScoreResult
    """
    print(f"\n{'=' * 80}")
    print("LOGIT-BASED EVALUATION (No Generation)")
    print(f"{'=' * 80}\n")

    print(f"Model: {model_path}")
    print(f"Test set: {test_csv}")
    print(f"Temperature: {temperature}")

    # Load test data
    df = pd.read_csv(test_csv)
    print(f"\nTest samples: {len(df)}")

    # Load model
    print("\nLoading model...")
    base_model_name = "Qwen/Qwen2-Audio-7B-Instruct"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load LoRA weights
    if (model_path / "adapter_config.json").exists():
        print(f"Loading LoRA weights from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference

    processor = Qwen2AudioProcessor.from_pretrained(base_model_name)

    # Get A/B token IDs
    ids_a, ids_b = get_ab_token_ids(processor.tokenizer)
    print(f"\nToken IDs:")
    print(f"  A: {ids_a} -> {[processor.tokenizer.decode([i]) for i in ids_a]}")
    print(f"  B: {ids_b} -> {[processor.tokenizer.decode([i]) for i in ids_b]}")

    # Evaluate
    print(f"\n{'=' * 80}")
    print("RUNNING EVALUATION")
    print(f"{'=' * 80}\n")

    results = []
    for idx, row in df.iterrows():
        audio_path = Path(row["audio_path"])
        true_label = row["label"]

        # Create question
        question = (
            "Listen carefully to this audio clip. Does it contain human speech?\n\n"
            "A) Yes, contains speech\n"
            "B) No, no speech detected\n\n"
            "Answer with only the letter (A or B):"
        )

        # Score with logits
        prob_a, prob_b, logit_a, logit_b = score_with_logits(
            model, processor, audio_path, question, ids_a, ids_b, temperature
        )

        # Predict based on higher probability
        predicted_label = "SPEECH" if prob_a > prob_b else "NONSPEECH"
        correct = predicted_label == true_label

        result = LogitScoreResult(
            filename=audio_path.name,
            true_label=true_label,
            predicted_label=predicted_label,
            prob_a=prob_a,
            prob_b=prob_b,
            logit_a=logit_a,
            logit_b=logit_b,
            correct=correct,
        )
        results.append(result)

        # Print result
        status = "[OK]" if correct else "[FAIL]"
        print(f"{status} {audio_path.name}")
        print(f"      Expected: {true_label}, Predicted: {predicted_label}")
        print(f"      P(A)={prob_a:.3f}, P(B)={prob_b:.3f} "
              f"(logits: A={logit_a:.2f}, B={logit_b:.2f})")

    return results


def main():
    """Run logit-based evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate with logit scoring")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            project_root
            / "checkpoints"
            / "qwen2_audio_speech_detection_normalized"
            / "final"
        ),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=str(project_root / "data" / "clean_clips_normalized" / "test_metadata.csv"),
        help="Path to test CSV",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for calibration (1.0 = no scaling)",
    )
    parser.add_argument(
        "--save_predictions",
        type=str,
        help="Save predictions to JSON file",
    )

    args = parser.parse_args()

    # Evaluate
    results = evaluate_with_logit_scoring(
        Path(args.checkpoint),
        Path(args.test_csv),
        temperature=args.temperature,
    )

    # Compute metrics
    correct = [r.correct for r in results]
    accuracy = np.mean(correct)

    speech_results = [r for r in results if r.true_label == "SPEECH"]
    nonspeech_results = [r for r in results if r.true_label == "NONSPEECH"]

    speech_acc = np.mean([r.correct for r in speech_results]) if speech_results else 0
    nonspeech_acc = (
        np.mean([r.correct for r in nonspeech_results]) if nonspeech_results else 0
    )

    # Confidence analysis
    correct_results = [r for r in results if r.correct]
    wrong_results = [r for r in results if not r.correct]

    conf_correct = (
        np.mean([max(r.prob_a, r.prob_b) for r in correct_results])
        if correct_results
        else 0
    )
    conf_wrong = (
        np.mean([max(r.prob_a, r.prob_b) for r in wrong_results]) if wrong_results else 0
    )

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"RESULT: {sum(correct)}/{len(results)} = {accuracy:.1%}")
    print(f"{'=' * 80}\n")

    print("Breakdown:")
    print(f"  SPEECH:    {sum(r.correct for r in speech_results)}/{len(speech_results)} "
          f"= {speech_acc:.1%}")
    print(f"  NONSPEECH: {sum(r.correct for r in nonspeech_results)}/{len(nonspeech_results)} "
          f"= {nonspeech_acc:.1%}")

    print(f"\nConfidence:")
    print(f"  Overall avg:  {np.mean([max(r.prob_a, r.prob_b) for r in results]):.3f}")
    print(f"  Correct avg:  {conf_correct:.3f}")
    print(f"  Wrong avg:    {conf_wrong:.3f}")
    print(f"  Gap:          {conf_correct - conf_wrong:.3f}")

    print(f"\n{'=' * 80}\n")

    # Save predictions if requested
    if args.save_predictions:
        import json

        predictions_data = {
            "temperature": args.temperature,
            "accuracy": accuracy,
            "predictions": [int(r.predicted_label == "SPEECH") for r in results],
            "ground_truth": [int(r.true_label == "SPEECH") for r in results],
            "probabilities": [
                {"prob_speech": r.prob_a, "prob_nonspeech": r.prob_b} for r in results
            ],
            "details": [
                {
                    "filename": r.filename,
                    "true_label": r.true_label,
                    "predicted_label": r.predicted_label,
                    "prob_a": r.prob_a,
                    "prob_b": r.prob_b,
                    "logit_a": r.logit_a,
                    "logit_b": r.logit_b,
                    "correct": r.correct,
                }
                for r in results
            ],
        }

        save_path = Path(args.save_predictions)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(predictions_data, f, indent=2)

        print(f"Predictions saved to: {save_path}")


if __name__ == "__main__":
    main()
