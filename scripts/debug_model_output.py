#!/usr/bin/env python3
"""
Debug script to see what the model ACTUALLY outputs (using generate, not logits).
"""

import pandas as pd
import torch
import soundfile as sf
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

def main():
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

    # Load a few test samples
    df = pd.read_csv("data/processed/experimental_variants/dev_metadata.csv")
    label_col = 'ground_truth' if 'ground_truth' in df.columns else 'label'

    # Get 3 SPEECH and 3 NONSPEECH samples
    speech_samples = df[df[label_col] == 'SPEECH'].head(3)
    nonspeech_samples = df[df[label_col] == 'NONSPEECH'].head(3)

    prompts_to_test = [
        # Simple direct question
        "What do you hear in this audio? Describe it briefly.",

        # Binary choice
        "Does this audio contain human speech? Answer YES or NO only.",

        # A/B format
        "Is there human voice in this audio?\nA) Yes\nB) No\nAnswer A or B only.",
    ]

    print("\n" + "="*70)
    print("TESTING MODEL ACTUAL OUTPUTS (using generate)")
    print("="*70)

    for prompt in prompts_to_test:
        print(f"\n>>> PROMPT: {prompt[:60]}...")
        print("-"*70)

        for label, samples in [("SPEECH", speech_samples), ("NONSPEECH", nonspeech_samples)]:
            print(f"\n  [{label} samples]")
            for idx, row in samples.iterrows():
                audio_path = row['audio_path']
                if not audio_path.startswith('data/'):
                    audio_path = 'data/' + audio_path

                # Load audio
                audio, sr = sf.read(audio_path)
                target_sr = processor.feature_extractor.sampling_rate

                if sr != target_sr:
                    import torchaudio.transforms as T
                    resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                    audio = resampler(torch.tensor(audio)).numpy()

                # Create conversation
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

                # Generate response
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False,
                    )

                # Decode only the new tokens
                input_len = inputs['input_ids'].shape[1]
                response = processor.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

                clip_id = row['clip_id']
                print(f"    {clip_id}: \"{response.strip()}\"")

    print("\n" + "="*70)
    print("DEBUG COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
