# Sprint 5 - Estado de Implementación

**Fecha**: 2025-10-11
**Status**: 🔄 EN PROGRESO (95% completado)

---

## Objetivo

Implementar generadores de condiciones psicoacústicas para las **4 manipulaciones principales**:

1. ✅ **SNR/Ruido blanco** (varios niveles)
2. ✅ **Duración** (ya implementado en sprints anteriores)
3. ✅ **Banda limitada** (telefonía + ablaciones)
4. 🔄 **Reverberación** (RIR con bins T60)

---

## ✅ Completado

### 1. Módulos de Audio Implementados

#### `src/qsm/audio/noise.py`
- ✅ Mezcla a SNR objetivo usando ruido blanco
- ✅ Cálculo de RMS sobre **segmento efectivo** (excluye padding)
- ✅ Aplicación de ruido a **todo el container de 2000ms** (sin dar pistas)
- ✅ Reproducibilidad vía `seed`
- ✅ Metadata: `snr_db`, `rms_signal`, `rms_noise`, `seed`

#### `src/qsm/audio/filters.py`
- ✅ Band-pass telefonía: 300-3400 Hz (estándar ITU-T)
- ✅ Ablaciones: Low-pass 3400 Hz, High-pass 300 Hz
- ✅ Implementación: Butterworth IIR 4º orden, zero-phase
- ✅ Validado: Response correcta en frecuencias clave

#### `src/qsm/audio/reverb.py`
- ✅ Convolución con RIR (FFT-based)
- ✅ Normalización de energía (preserva RMS)
- ✅ Database loader con soporte OpenSLR SLR28
- ✅ Filtrado por T60 range
- ✅ Metadata: `rir_id`, `T60`, `T60_bin`

### 2. Dataset RIR (OpenSLR SLR28)

- ✅ **Descargado**: 1.3 GB (`rirs_noises.zip`)
- ✅ **Descomprimido**: `data/external/RIRS_NOISES/RIRS_NOISES/`
- ✅ **Estructura**:
  - `simulated_rirs/` - RIRs simulados
  - `real_rirs_isotropic_noises/` - RIRs reales
  - `pointsource_noises/` - Ruidos de fondo
- ✅ **Total RIRs**: 60,417 archivos WAV

### 3. Script de Extracción de T60

#### `scripts/extract_rir_t60.py`
- ✅ Método Schroeder integration (backward energy curve)
- ✅ Estimación T30 (fit -5 a -35 dB, extrapolado a -60 dB)
- ✅ Procesamiento en batch con `tqdm`
- ✅ Output: `data/external/RIRS_NOISES/rir_metadata.json`
- 🔄 **Status**: Ejecutando (1,158 / 60,417 completados)

### 4. CLI de Generación de Condiciones

#### `scripts/build_conditions.py`
- ✅ Entrada: Manifest JSONL de audios padded a 2000ms
- ✅ Salida: Matriz completa `dur × SNR × band × RIR`
- ✅ Multiprocessing con `ProcessPoolExecutor`
- ✅ Output formats: JSONL + Parquet
- ✅ Flags:
  - `--snr_levels`: Comma-separated SNR (dB)
  - `--band_filters`: none, telephony, lp3400, hp300
  - `--rir_root`: Path to RIR dataset
  - `--rir_metadata`: Path to T60 metadata JSON
  - `--rir_t60_bins`: Ranges like "0.0-0.4,0.4-0.8,0.8-1.5"
  - `--n_workers`: Parallel workers

### 5. Condiciones Ya Generadas

#### Condiciones Existentes (`data/processed/conditions/`)
- **Total**: 6,264 variantes
- **SNR**: -10, -5, 0, +5, +10, +20 dB (4,176 variantes)
- **Band**: none, telephony, lp3400, hp300 (2,088 variantes)
- **Labels**: 3,384 NON-SPEECH + 2,880 SPEECH
- **Archivos**: `conditions_manifest.jsonl`, `conditions_manifest.parquet`

#### Condiciones de Alto Ruido (`data/processed/conditions_high_noise/`)
- **Total**: 1,392 variantes
- **SNR**: +40, +60 dB (ruido extremo)
- **Labels**: 752 NON-SPEECH + 640 SPEECH
- **Archivos**: `conditions_manifest.jsonl`, `conditions_manifest.parquet`

---

## 🔄 En Progreso

### 1. Extracción de T60 de RIRs
- **Proceso**: `scripts/extract_rir_t60.py` corriendo en background
- **Progreso**: ~2% (1,158 / 60,417 RIRs)
- **ETA**: ~15-20 minutos
- **Output**: `data/external/RIRS_NOISES/rir_metadata.json`

**Observación inicial** (muestra de 100 RIRs):
- Todos los RIRs tienen **T60 muy alto** (4.7 - 6.8 s)
- Distribución sesgada hacia espacios muy reverberantes
- Posible necesidad de ajustar bins T60 sugeridos

---

## ⏳ Pendiente

### 1. Completar Extracción de T60
- Esperar a que termine el proceso en background
- Analizar distribución completa de T60
- Ajustar bins T60 según distribución real

### 2. Generar Variantes con Reverberación
Una vez que tengamos el metadata de T60:

```bash
cd "c:\VS code projects\OPRO Qwen"

python scripts/build_conditions.py \
    --input_manifest data/processed/qsm_dev_padded.jsonl \
    --output_dir data/processed/conditions_with_rir/ \
    --snr_levels none \
    --band_filters none \
    --rir_root data/external/RIRS_NOISES/RIRS_NOISES \
    --rir_metadata data/external/RIRS_NOISES/rir_metadata.json \
    --rir_t60_bins "2.0-4.0,4.0-5.0,5.0-7.0" \
    --n_workers 4
```

**Nota**: Ajustar bins según distribución real observada

### 3. Subset de 4 Condiciones Principales

Crear subset balanceado para evaluación rápida:
- **Condición 1**: SNR -10 dB (10 samples)
- **Condición 2**: SNR +60 dB (10 samples)
- **Condición 3**: Telephony band (10 samples)
- **Condición 4**: RIR T60 medio (10 samples) - **PENDING**

Total: 40 muestras (20 SPEECH + 20 NON-SPEECH)

### 4. Evaluación de 4 Condiciones
- Qwen2-Audio en 4 condiciones
- Silero-VAD en 4 condiciones
- Análisis comparativo

---

## 📊 Estructura de Archivos Generados

```
data/
├── external/
│   └── RIRS_NOISES/
│       ├── rirs_noises.zip (1.3 GB)
│       ├── RIRS_NOISES/
│       │   ├── simulated_rirs/
│       │   ├── real_rirs_isotropic_noises/
│       │   └── pointsource_noises/
│       └── rir_metadata.json (🔄 en progreso)
│
├── processed/
│   ├── conditions/
│   │   ├── snr/ (4,176 WAVs)
│   │   ├── band/ (2,088 WAVs)
│   │   ├── conditions_manifest.jsonl
│   │   └── conditions_manifest.parquet
│   │
│   ├── conditions_high_noise/
│   │   ├── snr/ (1,392 WAVs)
│   │   ├── conditions_manifest.jsonl
│   │   └── conditions_manifest.parquet
│   │
│   └── conditions_with_rir/ (⏳ pendiente)
│       ├── rir/
│       ├── conditions_manifest.jsonl
│       └── conditions_manifest.parquet

src/qsm/audio/
├── __init__.py
├── noise.py ✅
├── filters.py ✅
└── reverb.py ✅

scripts/
├── build_conditions.py ✅
├── extract_rir_t60.py ✅
├── download_rirs.py ✅
└── test_audio_manipulations.py ✅
```

---

## 🔬 Observaciones Técnicas

### SNR Computation
- **RMS calculado sobre segmento efectivo** (no padding) para evitar inflar SNR
- **Ruido aplicado al container completo** para evitar cues
- Validación: SNR accuracy < ±0.5 dB en segmentos ≥200ms

### Band-Limiting
- Validación espectral confirmada:
  - 200 Hz: -31 dB (stopband)
  - 1000 Hz: 0 dB (passband)
  - 5000 Hz: -48 dB (stopband)

### RIR Processing
- **Método T60**: Schroeder integration + T30 extrapolation
- **Normalización**: Preserva RMS de señal original
- **Deterministic**: RIR selection por bin usando seed

---

## 🎯 Próximos Pasos Inmediatos

1. ✅ Esperar completitud de `extract_rir_t60.py` (~15 min)
2. 📊 Analizar distribución completa de T60
3. 🔧 Ajustar bins T60 según datos reales
4. 🎬 Generar variantes con RIR
5. 📋 Crear subset final de 4 condiciones
6. 🧪 Ejecutar evaluaciones (Qwen + Silero)
7. 📈 Análisis comparativo

---

## 📝 Notas

- **Padding de 2000ms** se mantiene consistente en todas las variantes
- Todas las manipulaciones son **deterministas** (seed-based)
- Metadata completa guardada en JSONL/Parquet para trazabilidad
- Compatible con framework de evaluación existente (Sprint 6)

---

**Status General Sprint 5**: 95% ✅
**Bloqueador Principal**: Esperar completitud de extracción T60 (~15 min)
**Siguiente Sprint**: Sprint 6 - Evaluación unificada en 4 condiciones
