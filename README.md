# Qwen Speech Minimum (QSM)

**Status:** ✅ Segments Complete | 🚀 Ready for Model Inference

Temporal threshold measurement and optimization for speech detection in Qwen models.

## 📊 Project Status

- **Sprint 0 (Infrastructure):** ✅ COMPLETE
- **Sprint 1 (Dataset Ingestion):** ✅ COMPLETE  
- **Sprint 2 (Segment Extraction):** ✅ COMPLETE - 1,280 balanced segments ready
- **Sprint 3 (Model Inference):** 🔜 Ready to start
- **Segments Generated:** ✅ 1,280 segments (640 SPEECH + 640 NONSPEECH)

## 🎉 Sprint 2 Complete - Balanced Dataset Ready

### Dataset Composition (1,280 segments total)

**SPEECH Segments (640 total)**
- AVA-Speech: 320 segments (8 durations × 40 each)
- VoxConverse: 320 segments (8 durations × 40 each)

**NONSPEECH Segments (640 total)**
- ESC-50: 640 segments (8 durations × 80 each)

**Durations:** 20, 40, 60, 80, 100, 200, 500, 1000 ms

**Per-duration balance:** 80 SPEECH vs 80 NONSPEECH for each duration

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

# ESC-50 (640 NONSPEECH segments)
python scripts/make_segments_esc50.py --durations 20 40 60 80 100 200 500 1000 --max-per-duration 80
```

### 3. Validate Segments

```bash
# Interactive player (WSL-compatible)
python scripts/play_segments_interactive.py --segments-dir data/segments/ava_speech/train --min-duration 500
```

### 4. Verify Balance

```python
import pandas as pd

ava = pd.read_parquet('data/segments/ava_speech/train/segments_metadata.parquet')
vox = pd.read_parquet('data/segments/voxconverse/dev/segments_metadata.parquet')
esc = pd.read_parquet('data/segments/esc50/nonspeech/segments.parquet')

print(f"SPEECH: {len(ava) + len(vox)}, NONSPEECH: {len(esc)}")
# Output: SPEECH: 640, NONSPEECH: 640
```

## Project Structure

```
qwen-speech-min/
├── scripts/
│   ├── make_segments_ava.py          # Generate AVA segments
│   ├── make_segments_voxconverse.py  # Generate VoxConverse segments  
│   ├── make_segments_esc50.py        # Generate ESC-50 NONSPEECH
│   └── play_segments_interactive.py  # Validate segments (WSL-compatible)
├── src/qsm/data/
│   ├── loaders.py   # RTTM, AVA-Speech loaders
│   └── slicing.py   # Segment extraction with safety buffers
└── data/segments/
    ├── ava_speech/train/      # 320 AVA SPEECH segments
    ├── voxconverse/dev/       # 320 VoxConverse SPEECH segments
    └── esc50/nonspeech/       # 640 ESC-50 NONSPEECH segments
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

### ESC-50
- Environmental Sound Classification
- 50 categories (rain, wind, animals, urban sounds, etc.)
- 100% guaranteed no speech
- Usage: 640 NONSPEECH segments

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

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - 1-minute reference
- [INSTALL.md](INSTALL.md) - Installation guide
- [QUICK_START.md](QUICK_START.md) - Segment generation guide
- [SPRINT_0_COMPLETE.md](SPRINT_0_COMPLETE.md) - Sprint 0 summary

## References

- [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)
- [AVA-Speech](https://research.google.com/ava/download.html)
- [VoxConverse](https://github.com/joonson/voxconverse)
- [ESC-50](https://github.com/karolpiczak/ESC-50)

## License

Apache-2.0

---

**Sprint 2 Complete:** 1,280 segments generated and balanced. Ready for Sprint 3 (Model Inference).
