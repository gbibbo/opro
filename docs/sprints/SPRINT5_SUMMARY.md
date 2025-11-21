# Sprint 5 Summary: Psychoacoustic Condition Generators

**Status:** ✅ COMPLETE
**Date:** 2025-10-10

---

## What Was Built

Three audio manipulation modules + CLI for generating psychoacoustic condition variants:

### 1. **SNR / White Noise Module** (`src/qsm/audio/noise.py`)
- ✅ Adds calibrated white noise at target SNR levels
- ✅ Computes RMS over **effective segment** (excludes padding)
- ✅ Noise applied to **entire 2000 ms container** (no boundary cues)
- ✅ Deterministic via seed

### 2. **Band-Limited Filtering Module** (`src/qsm/audio/filters.py`)
- ✅ Telephony band-pass (300-3400 Hz, ITU-T standard)
- ✅ Ablations: LP 3400 Hz, HP 300 Hz
- ✅ Zero-phase Butterworth IIR (4th order)
- ✅ Validated: -31 dB @ 200 Hz, 0 dB @ 1000 Hz, -48 dB @ 5000 Hz

### 3. **Reverberation Module** (`src/qsm/audio/reverb.py`)
- ✅ RIR convolution with energy normalization
- ✅ RIR database loader with T60 filtering
- ✅ Supports OpenSLR SLR28 RIRS_NOISES dataset

### 4. **CLI: `scripts/build_conditions.py`**
- ✅ Generates full condition matrix: **dur × SNR × band × RIR**
- ✅ Parallel processing (multiprocessing)
- ✅ Output: JSONL + Parquet manifests

### 5. **Supporting Scripts**
- ✅ `scripts/download_rirs.py` - Download OpenSLR SLR28 dataset
- ✅ `scripts/test_audio_manipulations.py` - Validation tests

---

## Test Results

All modules validated:

```
Audio Manipulation Module Tests
============================================================
Testing SNR mixing...
  Target SNR: 0.00 dB
  Actual SNR: 0.00 dB
  ✓ SNR test passed

Testing band-pass filters...
  200 Hz attenuation: -31.51 dB (expect < -20 dB)
  1000 Hz attenuation: -0.00 dB (expect ~ 0 dB)
  5000 Hz attenuation: -48.64 dB (expect < -20 dB)
  ✓ Band filter test passed

Testing RIR convolution...
  RIR length: 16000 samples (1.00 s)
  Output length: 8000 samples
  ✓ RIR convolution test passed

All tests passed! ✓
```

---

## Usage Example

```bash
# 1. Download RIR dataset (once)
python scripts/download_rirs.py --output_dir data/external/RIRS_NOISES

# 2. Build conditions
python scripts/build_conditions.py \
    --input_manifest data/processed/qsm_dev_padded.jsonl \
    --output_dir data/processed/conditions/ \
    --snr_levels -10,-5,0,5,10,20 \
    --band_filters none,telephony,lp3400,hp300 \
    --rir_root data/external/RIRS_NOISES/RIRS_NOISES \
    --rir_t60_bins 0.0-0.4,0.4-0.8,0.8-1.5 \
    --n_workers 4
```

**Output structure:**
```
data/processed/conditions/
├── snr/          # SNR sweep variants
├── band/         # Band-limited variants
├── rir/          # Reverberation variants
├── conditions_manifest.jsonl
└── conditions_manifest.parquet
```

---

## Key Design Decisions

### 1. SNR Computation Strategy
- **RMS computed on effective segment only** (excludes 2000 ms padding)
- **Noise added to entire container** to avoid giving model "cues"
- Prevents inflating SNR estimates while maintaining ecological validity

### 2. Band-Limiting Rationale
- **Telephony (300-3400 Hz):** Simulates phone/VoIP quality (ITU-T standard)
- **Ablations (LP/HP):** Isolate contributions of low vs high-frequency cues
- **Zero-phase filtering:** Preserves temporal envelope

### 3. RIR T60 Bins
- **0.0-0.4 s:** Dry / small rooms
- **0.4-0.8 s:** Medium rooms
- **0.8-1.5 s:** Large halls / reverberant spaces

### 4. Reproducibility
- All operations deterministic via `seed` parameter
- Metadata tracks all parameters (SNR, filter type, RIR ID, T60)

---

## Alignment with Qwen2-Audio Temporal Resolution

- **Qwen2-Audio encoder:** ~40 ms per frame (25 ms window, 10 ms hop, 2× pooling)
- **Our duration grid:** {20, 40, 60, 80, 100, 200, 500, 1000} ms
- **Strong improvements at ≥80 ms** align with **2-3 encoder frames** of context
- This psychoacoustic sweep will reveal how manipulations degrade performance **relative to this ~40 ms temporal quantum**

---

## Next Steps (Sprint 6)

1. **Create `evaluate_conditions.py`** (adapt from `evaluate_extended.py`)
   - Run Qwen2-Audio on all condition variants
   - Maintain multiple-choice prompt format (A/B/C/D)
   - Output: `results/qwen_conditions.parquet`

2. **Run Silero-VAD on same conditions** for comparison
   - Output: `results/silero_conditions.parquet`

3. **Unified analysis** (Sprint 7):
   - P(SPEECH) vs SNR curves (per duration)
   - P(SPEECH) vs duration curves (per SNR)
   - Band-limiting effects
   - T60 effects

---

## Documentation

- **Detailed guide:** [docs/SPRINT5_PSYCHOACOUSTIC_CONDITIONS.md](docs/SPRINT5_PSYCHOACOUSTIC_CONDITIONS.md)
- **Module docs:** See docstrings in `src/qsm/audio/*.py`

---

## Files Created

```
src/qsm/audio/
├── __init__.py
├── noise.py          # SNR / white noise
├── filters.py        # Band-limited filtering
└── reverb.py         # RIR convolution

scripts/
├── build_conditions.py           # Main CLI
├── download_rirs.py              # RIR dataset downloader
└── test_audio_manipulations.py  # Validation tests

docs/
└── SPRINT5_PSYCHOACOUSTIC_CONDITIONS.md  # Full documentation
```

---

**Sprint 5 Status:** ✅ COMPLETE
**All tests passing:** ✅
**Ready for Sprint 6:** ✅
