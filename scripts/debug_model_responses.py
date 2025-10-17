"""Debug script to inspect raw model outputs and parser behavior."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.qsm.models.qwen_audio import Qwen2AudioClassifier


def main():
    """Test model on a few samples and show raw outputs."""

    # Load metadata
    metadata_path = project_root / "data" / "processed" / "snr_duration_crossed" / "metadata.csv"
    df = pd.read_csv(metadata_path)

    # Take 10 SPEECH and 10 NONSPEECH samples
    speech_samples = df[df['ground_truth'] == 'SPEECH'].head(10)
    nonspeech_samples = df[df['ground_truth'].str.upper().isin(['NONSPEECH', 'NON-SPEECH'])].head(10)

    test_df = pd.concat([speech_samples, nonspeech_samples])

    print("="*80)
    print("MODEL RESPONSE DEBUGGING")
    print("="*80)

    # Test both prompts
    prompts = {
        "Baseline": "Is this audio clip SPEECH or NON-SPEECH?",
        "Optimized": "Based on the audio file, is it SPEECH or NON-SPEECH?"
    }

    for prompt_name, prompt_text in prompts.items():
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt_name}")
        print(f"Text: {prompt_text}")
        print(f"{'='*80}\n")

        # Load model
        model = Qwen2AudioClassifier(
            model_name="Qwen/Qwen2-Audio-7B-Instruct",
            device="cuda",
            load_in_4bit=True,
            constrained_decoding=False
        )

        # Set prompt
        model.set_prompt(user_prompt=prompt_text)

        # Test on samples
        correct = 0
        total = 0

        for idx, row in test_df.iterrows():
            audio_path = project_root / row['audio_path']
            true_label = row['ground_truth'].strip().upper()
            if true_label not in ['SPEECH', 'NONSPEECH', 'NON-SPEECH']:
                continue

            if true_label in ['NONSPEECH', 'NON-SPEECH']:
                true_label = 'NONSPEECH'

            # Predict
            result = model.predict(audio_path)

            # Show result
            is_correct = result.label == true_label
            correct += int(is_correct)
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"{status} TRUE: {true_label:<10} | PRED: {result.label:<10} | CONF: {result.confidence:.2f}")
            print(f"  RAW OUTPUT: '{result.raw_output[:100]}'")
            print(f"  Audio: {row['audio_path']}")
            print()

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'-'*80}")
        print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"{'-'*80}\n")

        # Clean up
        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("DEBUGGING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
