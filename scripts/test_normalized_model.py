"""Test fine-tuned model on NORMALIZED clips."""

import sys
from pathlib import Path
import pandas as pd
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel
import soundfile as sf
import librosa

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


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
    print("=" * 80)
    print("TESTING FINE-TUNED MODEL (Normalized Audio)")
    print("=" * 80)

    # Paths - use normalized clips
    checkpoint_dir = project_root / "checkpoints" / "qwen2_audio_speech_detection_normalized" / "final"
    test_csv = project_root / "data" / "processed" / "normalized_clips" / "test_metadata.csv"

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

    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
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
        inputs = processor(text=text, audio=[audio], return_tensors="pt")
        inputs = inputs.to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
            )

        # Decode
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_length:]
        output_text = processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

        # Parse
        predicted = "SPEECH" if output_text.upper() in ["A", "SPEECH"] else "NONSPEECH"

        is_correct = predicted == expected
        correct += int(is_correct)
        total += 1

        status = "[OK]" if is_correct else "[FAIL]"
        print(f"{status} {audio_path.name}")
        print(f"      Expected: {expected}, Predicted: {predicted} (raw: '{output_text}')")

        results.append({
            'file': audio_path.name,
            'expected': expected,
            'predicted': predicted,
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
