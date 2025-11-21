#!/usr/bin/env python3
"""
Quick test of multiple prompts on a small sample to find which works best.
"""

import argparse
import pandas as pd
import torch
import soundfile as sf
from pathlib import Path
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

PROMPTS = {
    "base": """Task: speech presence classification.
Classes: A = SPEECH (human voice present), B = NONSPEECH (no human voice).
Rules: Listen to the clip and output only a single uppercase letter: A or B. Do not include any other text.""",

    "simple": """Does this audio contain human speech? Answer A for yes (speech present) or B for no (no speech).""",

    "direct": """Listen to this audio clip. Is there human voice or speech in it?
A = Yes, human speech is present
B = No, only noise/music/silence
Answer with just A or B.""",

    "training": """Choose one:
A) SPEECH (human voice)
B) NONSPEECH (music/noise/silence/animals)

Answer with A or B ONLY.""",

    "question": """What do you hear in this audio?
A) Human speech or voice
B) Non-speech sounds (music, noise, silence, animals)
Reply with A or B only.""",
}


def get_ab_token_ids(tokenizer):
    tokens_a = tokenizer.encode('A', add_special_tokens=False)
    tokens_b = tokenizer.encode('B', add_special_tokens=False)
    return [tokens_a[0]], [tokens_b[0]]


def evaluate_sample(model, processor, audio_path, ids_A, ids_B, prompt):
    audio, sr = sf.read(audio_path)
    target_sr = processor.feature_extractor.sampling_rate

    if sr != target_sr:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(torch.tensor(audio)).numpy()

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text_prompt],
        audio=[audio],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]
    logit_A = logits[ids_A[0]].item()
    logit_B = logits[ids_B[0]].item()

    logit_diff = logit_A - logit_B
    prob_A = torch.sigmoid(torch.tensor(logit_diff)).item()

    prediction = 'SPEECH' if prob_A > 0.5 else 'NONSPEECH'
    return prediction, prob_A, logit_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=str, default="data/processed/experimental_variants/dev_metadata.csv")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples per class to test")
    args = parser.parse_args()

    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
    ids_A, ids_B = get_ab_token_ids(processor.tokenizer)

    # Load and sample test data
    df = pd.read_csv(args.test_csv)
    label_col = 'ground_truth' if 'ground_truth' in df.columns else 'label'

    # Balanced sample
    speech_samples = df[df[label_col] == 'SPEECH'].sample(n=min(args.n_samples, len(df[df[label_col] == 'SPEECH'])), random_state=42)
    nonspeech_samples = df[df[label_col] == 'NONSPEECH'].sample(n=min(args.n_samples, len(df[df[label_col] == 'NONSPEECH'])), random_state=42)
    test_df = pd.concat([speech_samples, nonspeech_samples])

    print(f"\nTesting on {len(test_df)} samples ({len(speech_samples)} SPEECH, {len(nonspeech_samples)} NONSPEECH)")
    print("=" * 70)

    results = {}
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n>>> Testing prompt: {prompt_name}")
        correct = 0
        speech_correct = 0
        nonspeech_correct = 0
        speech_total = 0
        nonspeech_total = 0

        for idx, row in test_df.iterrows():
            audio_path = row['audio_path']
            if not audio_path.startswith('data/'):
                audio_path = 'data/' + audio_path
            ground_truth = row[label_col]

            pred, prob_A, logit_diff = evaluate_sample(model, processor, audio_path, ids_A, ids_B, prompt_text)

            is_correct = (pred == ground_truth)
            correct += is_correct

            if ground_truth == 'SPEECH':
                speech_total += 1
                speech_correct += is_correct
            else:
                nonspeech_total += 1
                nonspeech_correct += is_correct

        acc = correct / len(test_df) * 100
        speech_acc = speech_correct / speech_total * 100 if speech_total > 0 else 0
        nonspeech_acc = nonspeech_correct / nonspeech_total * 100 if nonspeech_total > 0 else 0

        results[prompt_name] = {
            'accuracy': acc,
            'speech_acc': speech_acc,
            'nonspeech_acc': nonspeech_acc
        }

        print(f"    Overall: {acc:.1f}% | SPEECH: {speech_acc:.1f}% | NONSPEECH: {nonspeech_acc:.1f}%")

    print("\n" + "=" * 70)
    print("SUMMARY - Sorted by overall accuracy:")
    print("=" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{name:12s}: {r['accuracy']:5.1f}% overall | SPEECH: {r['speech_acc']:5.1f}% | NONSPEECH: {r['nonspeech_acc']:5.1f}%")


if __name__ == "__main__":
    main()
