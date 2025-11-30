#!/usr/bin/env python3
"""Quick verification of SPEECH samples using Silero VAD."""
import pandas as pd
import torch
import torchaudio
from pathlib import Path

# Load Silero VAD
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
(get_speech_timestamps, _, read_audio, _, _) = utils

def check_speech_ratio(audio_path, threshold=0.8):
    """Check if speech is present in at least threshold% of the clip."""
    try:
        wav = read_audio(audio_path, sampling_rate=16000)
        duration_samples = len(wav)

        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

        speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        speech_ratio = speech_samples / duration_samples if duration_samples > 0 else 0

        return speech_ratio, speech_ratio >= threshold
    except Exception as e:
        print(f"Error {audio_path}: {e}")
        return 0, False

# Load test data
df = pd.read_csv('data/processed/expanded_4conditions/test_metadata.csv')
speech_df = df[df['ground_truth'] == 'SPEECH']

print(f"Total SPEECH samples: {len(speech_df)}")
print(f"Threshold: 80% speech duration")
print()

passed = 0
failed_samples = []

for idx, row in speech_df.iterrows():
    audio_path = row['audio_path']
    if not Path(audio_path).exists():
        audio_path = 'data/' + audio_path if not audio_path.startswith('data/') else audio_path

    ratio, passed_check = check_speech_ratio(audio_path)

    if passed_check:
        passed += 1
    else:
        failed_samples.append((audio_path, ratio, row.get('duration_ms', 'N/A')))

print(f"Passed (>=80% speech): {passed}/{len(speech_df)} ({100*passed/len(speech_df):.1f}%)")
print(f"Failed (<80% speech): {len(failed_samples)}")

if failed_samples:
    print("\nFailed samples (first 20):")
    for path, ratio, dur in failed_samples[:20]:
        print(f"  {Path(path).name}: {ratio*100:.1f}% speech, duration={dur}ms")
