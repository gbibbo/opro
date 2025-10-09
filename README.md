# Qwen Speech Minimum (QSM)

**Status:** ✅ Sprint 3 Complete | 🚀 VAD Baseline Established

Temporal threshold measurement and optimization for speech detection in Qwen models.

## 📊 Project Status

- **Sprint 0 (Infrastructure):** ✅ COMPLETE
- **Sprint 1 (Dataset Ingestion):** ✅ COMPLETE
- **Sprint 2 (Segment Extraction):** ✅ COMPLETE
- **Sprint 3 (VAD Baseline):** ✅ COMPLETE - Silero-VAD baseline established
- **Sprint 4 (Model Inference):** 🔜 Ready to start
- **Dataset:** ✅ 1,016 clean segments (640 SPEECH + 376 NONSPEECH)

## 🎉 Sprint 3 Complete - VAD Baseline Established

### Silero-VAD Performance
- **SPEECH (AVA + VoxConverse):** 95-100% accuracy
- **NONSPEECH (ESC-50 Clean):** 98-100% accuracy on segments ≥100ms
- **Latency:** ~10-20ms (suitable for real-time applications)

### Dataset Composition (1,016 segments total)

**SPEECH Segments (640 total)**
- AVA-Speech: 320 segments (8 durations × 40 each)
- VoxConverse: 320 segments (8 durations × 40 each)

**NONSPEECH Segments (376 total - cleaned)**
- ESC-50: 376 segments (8 durations × 47 each)
- **Ambiguous sounds removed:** Human sounds (breathing, clapping, sneezing, etc.) and animal vocalizations (cat, bird, pig, etc.) eliminated for clear binary classification

**Durations:** 20, 40, 60, 80, 100, 200, 500, 1000 ms

**Clean categories (23 total):**
- Mechanical: airplane, car_horn, chainsaw, engine, helicopter, train, vacuum_cleaner, washing_machine
- Environmental: crackling_fire, rain, sea_waves, thunderstorm, water_drops, wind
- Urban: clock_alarm, clock_tick, glass_breaking, keyboard_typing, siren, toilet_flush
- Actions: can_opening, door_wood_creaks, pouring_water

### Safety Buffer Implementation
- 1-second safety buffers at start/end of all ground truth intervals
- Ensures 100% label confidence (no boundary ambiguity)

## Quick Start

### 1. Environment Setup

```bash
conda activate opro
pip install -e .
```

### 2. Generate Segments

```bash
# AVA-Speech (320 SPEECH segments)
python scripts/make_segments_ava.py --durations 20 40 60 80 100 200 500 1000 --max-per-duration 40

# VoxConverse (320 SPEECH segments)
python scripts/make_segments_voxconverse.py --durations 20 40 60 80 100 200 500 1000 --max-per-duration 40

# ESC-50 Clean (376 NONSPEECH segments - ambiguous sounds removed)
python scripts/make_segments_esc50.py --durations 20 40 60 80 100 200 500 1000 --max-per-duration 80
```

### 3. Run VAD Baseline

```bash
# Evaluate on SPEECH
python scripts/run_vad_baseline.py --segments-dir data/segments/ava_speech/train

# Evaluate on NONSPEECH
python scripts/run_vad_baseline.py --segments-dir data/segments/esc50/nonspeech

# Custom threshold
python scripts/run_vad_baseline.py --segments-dir data/segments/voxconverse/dev --threshold 0.5
```

### 4. Validate Segments

```bash
# Interactive player (WSL-compatible)
python scripts/play_segments_interactive.py --segments-dir data/segments/ava_speech/train --min-duration 500
```

### 5. Verify Dataset

```python
import pandas as pd

ava = pd.read_parquet('data/segments/ava_speech/train/segments_metadata.parquet')
vox = pd.read_parquet('data/segments/voxconverse/dev/segments_metadata.parquet')
esc = pd.read_parquet('data/segments/esc50/nonspeech/segments.parquet')

print(f"SPEECH: {len(ava) + len(vox)}, NONSPEECH: {len(esc)}")
# Output: SPEECH: 640, NONSPEECH: 376

# Check ESC-50 categories (should be 23 clean categories)
print(f"NONSPEECH categories: {esc['condition'].nunique()}")
print(esc['condition'].value_counts().sort_index())
```

## Project Structure

```
qwen-speech-min/
├── scripts/
│   ├── make_segments_ava.py          # Generate AVA segments
│   ├── make_segments_voxconverse.py  # Generate VoxConverse segments
│   ├── make_segments_esc50.py        # Generate ESC-50 NONSPEECH (cleaned)
│   ├── clean_esc50_dataset.py        # Remove ambiguous sounds
│   ├── run_vad_baseline.py           # Evaluate Silero-VAD
│   └── play_segments_interactive.py  # Validate segments (WSL-compatible)
├── src/qsm/
│   ├── data/
│   │   ├── loaders.py   # RTTM, AVA-Speech loaders
│   │   └── slicing.py   # Segment extraction with safety buffers
│   └── vad/
│       ├── base.py      # VAD abstract interface
│       └── silero.py    # Silero-VAD implementation
├── data/segments/
│   ├── ava_speech/train/      # 320 AVA SPEECH segments
│   ├── voxconverse/dev/       # 320 VoxConverse SPEECH segments
│   └── esc50/nonspeech/       # 376 ESC-50 NONSPEECH segments (cleaned)
└── results/vad_baseline/      # Silero-VAD evaluation results
```

## Datasets

### AVA-Speech
- Frame-level annotations (25fps, ~40ms resolution)
- Conditions: clean, music, noise
- Usage: 320 SPEECH segments

### VoxConverse
- RTTM annotations
- Multi-speaker conversations
- Usage: 320 SPEECH segments

### ESC-50 (Cleaned)
- Environmental Sound Classification
- **23 clean categories** (41.2% of original dataset removed)
- **Removed:** Human sounds, animal vocalizations, ambiguous sounds
- 100% guaranteed no speech or speech-like sounds
- Usage: 376 NONSPEECH segments

## VAD Baseline

### Silero-VAD
- **Neural network** based VAD (superior to WebRTC for balanced classification)
- Frame duration: ~32ms (window_size=512 samples at 16kHz)
- Threshold: 0.5 (configurable)
- **Performance:**
  - SPEECH: 95-100% accuracy
  - NONSPEECH: 98-100% accuracy (segments ≥100ms)
  - Latency: 10-20ms

### Why Silero over WebRTC?
WebRTC-VAD was designed for telephony with high recall optimization (catch all speech), resulting in 87% false positive rate on clean NONSPEECH. Silero-VAD provides balanced precision and recall suitable for binary classification.

## Safety Buffers

All segments extracted with 1-second safety buffers:

```
Ground truth interval: [10.0s, 15.0s] (5 seconds SPEECH)
Safety buffer exclusions: [10.0, 11.0] and [14.0, 15.0]
Valid extraction zone: [11.0, 14.0] (3 seconds)
```

This guarantees high-confidence labels for psychometric threshold measurement.

## Architecture

### Qwen2-Audio Temporal Resolution
- Mel spectrogram: 25ms window / 10ms hop
- Pooling: ×2 stride
- Effective resolution: ~40ms per output frame

### Target Durations
Based on Qwen2-Audio's ~40ms frame resolution:
```
[20, 40, 60, 80, 100, 200, 500, 1000] ms
```

## Dependencies

Core:
- `torch`, `torchaudio` (Silero-VAD)
- `pandas`, `pyannote.core`, `soundfile`
- `sklearn` (metrics)

See [pyproject.toml](pyproject.toml) for complete list.

## Documentation

- [INSTALL.md](INSTALL.md) - Installation guide
- [QUICK_START.md](QUICK_START.md) - Segment generation guide
- [SPRINT_0_COMPLETE.md](SPRINT_0_COMPLETE.md) - Sprint 0 summary

## Results

VAD baseline results available in `results/vad_baseline/`:
- AVA-Speech: [results/vad_baseline/ava_speech/](results/vad_baseline/ava_speech/)
- VoxConverse: [results/vad_baseline/voxconverse/](results/vad_baseline/voxconverse/)
- ESC-50 Clean: [results/vad_baseline/esc50/](results/vad_baseline/esc50/)

## References

- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- [AVA-Speech](https://research.google.com/ava/download.html)
- [VoxConverse](https://github.com/joonson/voxconverse)
- [ESC-50](https://github.com/karolpiczak/ESC-50)
- [Silero-VAD](https://github.com/snakers4/silero-vad)

## License

Apache-2.0

---

**Sprint 3 Complete:** Silero-VAD baseline established with 95-100% accuracy on clean dataset. Ready for Sprint 4 (Qwen2-Audio Model Inference).
