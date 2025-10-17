"""Test A/B format with constrained decoding."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.qsm.models.qwen_audio import Qwen2AudioClassifier


def main():
    """Test A/B format on easy samples."""

    # Load metadata
    metadata_path = project_root / "data" / "processed" / "snr_duration_crossed" / "metadata.csv"
    df = pd.read_csv(metadata_path)

    # Select EASY samples only (>=500ms, >=+10dB)
    easy = df[(df['duration_ms'] >= 500) & (df['snr_db'] >= 10)]

    speech_samples = easy[easy['ground_truth'] == 'SPEECH'].head(5)
    nonspeech_samples = easy[easy['ground_truth'].str.upper().isin(['NONSPEECH', 'NON-SPEECH'])].head(5)

    test_df = pd.concat([speech_samples, nonspeech_samples])

    print("="*80)
    print("A/B FORMAT TEST - Constrained Decoding")
    print("="*80)
    print(f"\nTesting on {len(test_df)} EASY samples (>=500ms, >=+10dB)")
    print("Expected: ~100% accuracy\n")

    # Test with A/B prompt and constrained decoding
    print("Loading model with constrained decoding...")
    model = Qwen2AudioClassifier(
        model_name="Qwen/Qwen2-Audio-7B-Instruct",
        device="cuda",
        load_in_4bit=True,
        constrained_decoding=True  # A/B format with max_new_tokens=3
    )

    # Model already has A/B prompt as default
    print(f"\nPrompt:\n{model.user_prompt}\n")

    correct = 0
    total = 0

    for idx, row in test_df.iterrows():
        audio_path = project_root / row['audio_path']
        true_label = row['ground_truth'].strip().upper()

        if true_label in ['NONSPEECH', 'NON-SPEECH']:
            true_label = 'NONSPEECH'

        # Predict
        result = model.predict(audio_path)

        is_correct = result.label == true_label
        correct += int(is_correct)
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} TRUE: {true_label:<10} | PRED: {result.label:<10} | CONF: {result.confidence:.2f}")
        print(f"  RAW OUTPUT: '{result.raw_output}'")
        print(f"  Duration: {row['duration_ms']:.0f}ms | SNR: {row['snr_db']:+.0f}dB")
        print()

    accuracy = correct / total if total > 0 else 0
    print("="*80)
    print(f"RESULT: {correct}/{total} = {accuracy:.1%}")
    print("="*80)

    if accuracy >= 0.9:
        print("\n✓ SUCCESS: A/B format with constrained decoding works correctly!")
        print("  Recommendation: Use this for gate evaluation")
    elif accuracy >= 0.7:
        print("\n⚠️  PARTIAL SUCCESS: Acceptable but not optimal")
        print("  Recommendation: Review failures and tune constraints")
    else:
        print("\n✗ FAILURE: A/B format not working as expected")
        print("  Recommendation: Debug constrained decoding implementation")


if __name__ == "__main__":
    main()
