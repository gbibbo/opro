"""Test fine-tuned model on NORMALIZED clips with constrained A/B decoding."""

import sys
from pathlib import Path
import pandas as pd
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel
import soundfile as sf
import librosa
import numpy as np

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def get_ab_token_ids(tokenizer):
    """Get all single-token variants for A and B (with/without leading space/newline)."""
    def get_single_token_variants(char: str):
        """Get all single-token IDs that decode to variants of the character."""
        variants = [char, f" {char}", f"\n{char}"]
        valid_ids = []
        for variant in variants:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                # Verify it decodes back to something containing the char
                decoded = tokenizer.decode([ids[0]])
                if char in decoded.upper():
                    valid_ids.append(ids[0])
        return list(set(valid_ids))  # Remove duplicates

    ids_A = get_single_token_variants("A")
    ids_B = get_single_token_variants("B")

    if not ids_A or not ids_B:
        raise ValueError(f"Could not find single tokens for A/B. A: {ids_A}, B: {ids_B}")

    return ids_A, ids_B


def make_ab_prefix_fn(ids_A, ids_B):
    """Create prefix_allowed_tokens_fn that only allows A or B on first step."""
    allowed = ids_A + ids_B
    def prefix_fn(batch_id, input_ids):
        return allowed
    return prefix_fn


def load_finetuned_model(checkpoint_dir: Path, base_model_name: str):
    """Load fine-tuned model with LoRA weights."""
    print(f"Loading base model: {base_model_name}")
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"Loading LoRA weights from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_dir))

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test fine-tuned model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(project_root / "checkpoints" / "qwen2_audio_speech_detection_normalized" / "final"),
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default=str(project_root / "data" / "processed" / "normalized_clips" / "test_metadata.csv"),
        help="Path to test CSV",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("TESTING FINE-TUNED MODEL (Normalized Audio)")
    print("=" * 80)

    # Paths - use normalized clips
    checkpoint_dir = Path(args.checkpoint)
    test_csv = Path(args.test_csv)

    if not checkpoint_dir.exists():
        print(f"\n[ERROR] Checkpoint not found: {checkpoint_dir}")
        return

    if not test_csv.exists():
        print(f"\n[ERROR] Test data not found: {test_csv}")
        return

    # Load test data
    print(f"\nLoading test data...")
    df_test = pd.read_csv(test_csv)
    print(f"  Test samples: {len(df_test)}")
    print(f"    SPEECH:    {(df_test['ground_truth'] == 'SPEECH').sum()}")
    print(f"    NONSPEECH: {(df_test['ground_truth'] == 'NONSPEECH').sum()}")

    # Load fine-tuned model
    print(f"\nLoading fine-tuned model...")
    model = load_finetuned_model(
        checkpoint_dir,
        base_model_name="Qwen/Qwen2-Audio-7B-Instruct"
    )

    # Load processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model.eval()

    # Setup constrained A/B decoding
    print(f"\nSetting up constrained A/B decoding...")
    ids_A, ids_B = get_ab_token_ids(processor.tokenizer)
    prefix_fn = make_ab_prefix_fn(ids_A, ids_B)

    print(f"  Tokens for 'A': {ids_A} -> {[repr(processor.tokenizer.decode([tid])) for tid in ids_A]}")
    print(f"  Tokens for 'B': {ids_B} -> {[repr(processor.tokenizer.decode([tid])) for tid in ids_B]}")

    print("\n" + "=" * 80)
    print("RUNNING EVALUATION (with constrained A/B decoding)")
    print("=" * 80)

    # System and user prompts
    system_prompt = "You classify audio content."
    user_prompt = (
        "Choose one:\n"
        "A) SPEECH (human voice)\n"
        "B) NONSPEECH (music/noise/silence/animals)\n\n"
        "Answer with A or B ONLY."
    )

    correct = 0
    total = 0
    results = []

    for idx, row in df_test.iterrows():
        audio_path = project_root / row['audio_path']

        if not audio_path.exists():
            print(f"[SKIP] {audio_path.name}")
            continue

        expected = row['ground_truth'].upper()

        # Load and process audio
        audio, sr = sf.read(audio_path)
        target_sr = processor.feature_extractor.sampling_rate
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype('float32')

        # Prepare conversation
        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # Process
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=[audio], sampling_rate=target_sr, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

        # Generate with constrained decoding and scores
        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                prefix_allowed_tokens_fn=prefix_fn,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Extract sequences and scores
        generated_sequences = gen_output.sequences
        scores = gen_output.scores[0]  # Logits for first (and only) generated token

        # Decode
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_sequences[:, input_length:]
        output_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        # Compute confidence from logits (softmax over A and B only)
        # Get logits for A and B tokens
        logits_A = scores[0, ids_A]  # Shape: [len(ids_A)]
        logits_B = scores[0, ids_B]  # Shape: [len(ids_B)]

        # Combine all A and B logits
        all_logits = torch.cat([logits_A, logits_B])  # Shape: [len(ids_A) + len(ids_B)]
        probs_all = torch.softmax(all_logits, dim=0)

        # Sum probabilities for each class (in case multiple token variants)
        prob_A = probs_all[:len(ids_A)].sum().item()
        prob_B = probs_all[len(ids_A):].sum().item()

        # Determine prediction based on generated token
        generated_token_id = generated_tokens[0, 0].item()
        is_A = generated_token_id in ids_A
        predicted = "SPEECH" if is_A else "NONSPEECH"
        confidence = prob_A if is_A else prob_B

        is_correct = predicted == expected
        correct += int(is_correct)
        total += 1

        status = "[OK]" if is_correct else "[FAIL]"
        print(f"{status} {audio_path.name}")
        print(f"      Expected: {expected}, Predicted: {predicted} (conf: {confidence:.3f}, raw: '{output_text}')")

        results.append({
            'file': audio_path.name,
            'expected': expected,
            'predicted': predicted,
            'confidence': confidence,
            'prob_A': prob_A,
            'prob_B': prob_B,
            'correct': is_correct
        })

    accuracy = 100 * correct / total if total > 0 else 0

    print(f"\n{'=' * 80}")
    print(f"RESULT: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'=' * 80}")

    # Breakdown by class
    speech_results = [r for r in results if r['expected'] == 'SPEECH']
    nonspeech_results = [r for r in results if r['expected'] == 'NONSPEECH']

    speech_correct = sum(1 for r in speech_results if r['correct'])
    nonspeech_correct = sum(1 for r in nonspeech_results if r['correct'])

    print(f"\nBreakdown:")
    print(f"  SPEECH:    {speech_correct}/{len(speech_results)} = {100*speech_correct/len(speech_results) if speech_results else 0:.1f}%")
    print(f"  NONSPEECH: {nonspeech_correct}/{len(nonspeech_results)} = {100*nonspeech_correct/len(nonspeech_results) if nonspeech_results else 0:.1f}%")

    # Confidence stats
    avg_confidence = np.mean([r['confidence'] for r in results])
    avg_conf_correct = np.mean([r['confidence'] for r in results if r['correct']])
    avg_conf_wrong = np.mean([r['confidence'] for r in results if not r['correct']]) if any(not r['correct'] for r in results) else 0

    print(f"\nConfidence:")
    print(f"  Overall avg:  {avg_confidence:.3f}")
    print(f"  Correct avg:  {avg_conf_correct:.3f}")
    print(f"  Wrong avg:    {avg_conf_wrong:.3f}")

    if accuracy > 50:
        improvement = accuracy - 50
        print(f"\n[SUCCESS] Improved by {improvement:.1f}% over baseline (50%)!")
        if accuracy >= 75:
            print("  This is a good result with normalized audio!")
    elif accuracy == 50:
        print("\n[NOTE] Still at 50%. The model hasn't learned to distinguish.")
        print("  Possible reasons:")
        print("  - Need more training data (currently only 32 samples)")
        print("  - Need more epochs or different hyperparameters")
        print("  - The audio might still not be distinguishable")
    else:
        print(f"\n[WARNING] Accuracy dropped below 50%.")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
