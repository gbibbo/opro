# Sprint 5: Psychoacoustic Condition Generators

**Status:** ✅ Complete
**Date:** 2025-10-10

## Overview

This sprint implements **deterministic and reproducible** psychoacoustic manipulations for QSM evaluation:

1. **SNR sweep** (white noise at different SNR levels)
2. **Band-limited filtering** (telephony 300-3400 Hz, LP, HP)
3. **Reverberation** (RIR convolution with T60 sweep)

All manipulations are applied to **padded 2000 ms containers** (validated in Sprint 4).

---

## Modules

### 1. `src/qsm/audio/noise.py`
White noise / SNR mixing.

**Key features:**
- Computes RMS over **effective segment** (excludes padding) to avoid inflating SNR
- Adds noise to **entire container** to avoid boundary cues
- Deterministic via `seed` parameter

**Example:**
```python
from qsm.audio import mix_at_snr
import soundfile as sf

audio, sr = sf.read("padded_2000ms.wav")
noisy, meta = mix_at_snr(
    audio,
    snr_db=0.0,
    sr=sr,
    padding_ms=2000,
    effective_dur_ms=100,  # 100 ms effective segment
    seed=42,
)
print(meta)  # {"snr_db": 0.0, "rms_signal": ..., "rms_noise": ..., "seed": 42}
```

---

### 2. `src/qsm/audio/filters.py`
Band-limited filtering (telephony, LP, HP).

**Key features:**
- Zero-phase Butterworth IIR filters (4th order)
- Telephony band: **300-3400 Hz** (ITU-T standard)
- Ablations: LP 3400 Hz, HP 300 Hz

**Example:**
```python
from qsm.audio import apply_bandpass, apply_lowpass, apply_highpass
import soundfile as sf

audio, sr = sf.read("input.wav")

# Telephony band (300-3400 Hz)
telephony = apply_bandpass(audio, sr, lowcut=300, highcut=3400)

# Low-pass only
lp = apply_lowpass(audio, sr, highcut=3400)

# High-pass only
hp = apply_highpass(audio, sr, lowcut=300)
```

**Validation:**
- 200 Hz: -31 dB attenuation ✓
- 1000 Hz: 0 dB (passband) ✓
- 5000 Hz: -48 dB attenuation ✓

---

### 3. `src/qsm/audio/reverb.py`
Reverberation via RIR convolution.

**Key features:**
- Uses **OpenSLR SLR28 RIRS_NOISES** dataset
- RIR database with T60 filtering
- Energy-normalized convolution (preserves RMS)

**Example:**
```python
from qsm.audio import load_rir_database, apply_rir
import soundfile as sf

# Load RIR database
rir_db = load_rir_database("data/external/RIRS_NOISES/RIRS_NOISES")
print(f"Loaded {len(rir_db.list_all())} RIRs")

# Get RIRs in T60 range [0.4, 0.8] seconds
rir_ids = rir_db.get_by_t60(t60_min=0.4, t60_max=0.8)
rir_audio = rir_db.get_rir(rir_ids[0], sr=16000)

# Apply RIR
audio, sr = sf.read("input.wav")
reverb = apply_rir(audio, rir_audio, normalize=True)
```

---

## CLI: `scripts/build_conditions.py`

Generate full condition matrix: **dur × SNR × band × RIR**.

### Usage

```bash
# Download RIR dataset (once)
python scripts/download_rirs.py --output_dir data/external/RIRS_NOISES

# Build conditions
python scripts/build_conditions.py \
    --input_manifest data/processed/qsm_dev_padded.jsonl \
    --output_dir data/processed/conditions/ \
    --snr_levels -10,-5,0,5,10,20 \
    --band_filters none,telephony,lp3400,hp300 \
    --rir_root data/external/RIRS_NOISES/RIRS_NOISES \
    --rir_metadata data/external/RIRS_NOISES/rir_metadata.json \
    --rir_t60_bins 0.0-0.4,0.4-0.8,0.8-1.5 \
    --n_workers 4
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_manifest` | Input JSONL manifest (padded 2000ms) | *required* |
| `--output_dir` | Output directory for variants | *required* |
| `--snr_levels` | Comma-separated SNR levels (dB) | `-10,-5,0,5,10,20` |
| `--band_filters` | Comma-separated band filters | `none,telephony,lp3400,hp300` |
| `--rir_root` | Root directory of RIR dataset | `None` |
| `--rir_metadata` | Optional RIR metadata JSON with T60 | `None` |
| `--rir_t60_bins` | T60 bins (e.g., `0.0-0.4,0.4-0.8`) | `none` |
| `--n_workers` | Number of parallel workers | `4` |
| `--seed` | Base random seed | `42` |

### Output Structure

```
data/processed/conditions/
├── snr/
│   ├── clip001_snr-10.wav
│   ├── clip001_snr+0.wav
│   └── ...
├── band/
│   ├── clip001_bandtelephony.wav
│   ├── clip001_bandlp3400.wav
│   └── ...
├── rir/
│   ├── clip001_rir_T60_0.4-0.8.wav
│   └── ...
├── conditions_manifest.jsonl  # Metadata (JSONL)
└── conditions_manifest.parquet  # Metadata (Parquet)
```

### Metadata Schema

Each variant has:
- `clip_id`: Original clip ID
- `original_path`: Path to original padded audio
- `duration_ms`: Effective duration (ms)
- `label`: Ground truth label (SPEECH/NON-SPEECH)
- `variant_type`: Manipulation type (`snr`, `band`, `rir`)
- `snr_db`: SNR level (if applicable)
- `band_filter`: Filter type (`none`, `telephony`, `lp3400`, `hp300`)
- `rir_id`: RIR identifier (if applicable)
- `T60`: Reverberation time (sec, if applicable)
- `audio_path`: Path to variant audio file

---

## Testing

Run validation tests:

```bash
python scripts/test_audio_manipulations.py
```

**Expected output:**
```
============================================================
Audio Manipulation Module Tests
============================================================
Testing SNR mixing...
  Target SNR: 0.00 dB
  Actual SNR: 0.00 dB
  RMS signal: 0.706886
  RMS noise: 0.706886
✓ SNR test passed
Testing band-pass filters...
  200 Hz attenuation: -31.51 dB (expect < -20 dB)
  1000 Hz attenuation: -0.00 dB (expect ~ 0 dB)
  5000 Hz attenuation: -48.64 dB (expect < -20 dB)
✓ Band filter test passed
Testing RIR convolution...
  RIR length: 16000 samples (1.00 s)
  Output length: 8000 samples
  Max amplitude: 1.0000
✓ RIR convolution test passed
============================================================
All tests passed! ✓
============================================================
```

---

## Notes

### SNR Computation
- **Critical:** SNR is computed relative to the **effective segment** (excludes padding), but noise is added to the **entire 2000 ms container**.
- This avoids inflating SNR estimates and prevents giving the model "cues" about segment boundaries.

### Band-Limiting Rationale
- **Telephony band (300-3400 Hz):** ITU-T standard, simulates phone/VoIP quality
- **Ablations (LP/HP):** Isolate contributions of low-frequency vs high-frequency cues

### RIR Dataset (OpenSLR SLR28)
- **Simulated RIRs:** Parametric room models with known T60
- **Real RIRs:** Measured impulse responses from real rooms
- **T60 bins:** Categorize RIRs by reverberation time (e.g., 0.0-0.4, 0.4-0.8, 0.8-1.5 sec)

### Duration Grid Alignment
- Effective durations: `{20, 40, 60, 80, 100, 200, 500, 1000} ms`
- **~40 ms per encoder frame** (Qwen2-Audio: 25 ms window, 10 ms hop, 2× pooling)
- Strong improvements at ≥80 ms align with **2-3 frames** of context

---

## Next Steps (Sprint 6)

1. **Evaluate conditions** with Qwen2-Audio and Silero-VAD:
   ```bash
   python scripts/evaluate_conditions.py \
       --conditions_manifest data/processed/conditions/conditions_manifest.jsonl \
       --output_dir results/qwen_conditions/
   ```

2. **Compare Qwen vs Silero** across SNR/band/RIR sweeps

3. **Generate psychometric curves** (Sprint 7):
   - P(SPEECH) vs SNR
   - P(SPEECH) vs duration
   - Effects of band-limiting and T60

---

## References

1. **SNR / MTF:** Houtgast & Steeneken (1985). "A review of the MTF concept in room acoustics"
2. **Telephony band:** ITU-T Recommendation G.711 (300-3400 Hz)
3. **RIR dataset:** OpenSLR SLR28 RIRS_NOISES - https://www.openslr.org/28/
4. **Qwen2-Audio temporal resolution:** Chu et al. (2024). "Qwen2-Audio Technical Report" - https://arxiv.org/abs/2407.10759

---

**Status:** ✅ All modules tested and validated
**Author:** Claude Code
**Date:** 2025-10-10
