# Refactoring Summary: Unified Audio Transformations

## Cambio Realizado

Movimos la funcionalidad de segmentación por duración de `src/qsm/data/slicing.py` a un nuevo módulo `src/qsm/audio/slicing.py` para unificar todas las transformaciones psicoac uísticas bajo el mismo namespace.

## Motivación

**Antes**: Las transformaciones estaban dispersas en diferentes módulos:
- ❌ `src/qsm/data/slicing.py` - Segmentación por duración (aislada en "data")
- ✅ `src/qsm/audio/noise.py` - SNR/ruido
- ✅ `src/qsm/audio/filters.py` - Filtros de banda
- ✅ `src/qsm/audio/reverb.py` - Reverberación

**Problema**: La segmentación por duración es una transformación psicoac uística igual que SNR/filtros/reverb, pero estaba separada en un módulo diferente. Esto sugería que tenía menos importancia o era diferente conceptualmente.

**Después**: Todas las transformaciones están al mismo nivel en `src/qsm/audio/`:
- ✅ `src/qsm/audio/slicing.py` - Segmentación por duración y padding
- ✅ `src/qsm/audio/noise.py` - SNR/ruido
- ✅ `src/qsm/audio/filters.py` - Filtros de banda
- ✅ `src/qsm/audio/reverb.py` - Reverberación

## Arquitectura del Módulo `qsm.audio`

```
src/qsm/audio/
├── __init__.py          # Exporta todas las funciones públicas
├── slicing.py           # ⭐ NUEVO: Segmentación y padding
├── noise.py             # Mezcla de ruido blanco / SNR
├── filters.py           # Filtros de banda (telephony, LP, HP)
└── reverb.py            # Convolución con RIR
```

Todas las transformaciones son **iguales en importancia** y peso conceptual:
- **Duration**: Evaluamos el umbral temporal
- **SNR**: Evaluamos robustez a ruido
- **Filters**: Evaluamos robustez a limitación de frecuencias
- **Reverb**: Evaluamos robustez a reverberación

## Nuevo Módulo: `src/qsm/audio/slicing.py`

### Funciones Principales

```python
from qsm.audio import (
    extract_segment_center,      # Extrae segmento desde el centro
    pad_audio_center,             # Paddea con ruido de baja amplitud
    slice_and_pad,                # Combina extracción + padding
    create_duration_variants,     # Crea todas las duraciones de un clip
)
```

### Ejemplos de Uso

#### 1. Extraer segmento del centro

```python
import numpy as np
from qsm.audio import extract_segment_center

# Audio de 1000ms
audio_1000ms = np.random.randn(16000)  # 16kHz

# Extraer 100ms del centro
segment_100ms = extract_segment_center(audio_1000ms, duration_ms=100, sr=16000)
print(len(segment_100ms))  # 1600 samples = 100ms
```

#### 2. Paddear audio con ruido

```python
from qsm.audio import pad_audio_center

# Audio de 100ms
audio_100ms = np.random.randn(1600)

# Paddear a 2000ms (centrado en ruido de baja amplitud)
padded = pad_audio_center(
    audio_100ms,
    target_duration_ms=2000,
    sr=16000,
    noise_amplitude=0.0001,
    seed=42
)
print(len(padded))  # 32000 samples = 2000ms
```

#### 3. Combinar extracción + padding

```python
from qsm.audio import slice_and_pad

# Audio de 1000ms
audio_1000ms = np.random.randn(16000)

# Extraer 100ms del centro y paddear a 2000ms
segment = slice_and_pad(
    audio_1000ms,
    duration_ms=100,
    padding_ms=2000,
    sr=16000
)
print(len(segment))  # 32000 samples = 2000ms
```

#### 4. Crear todas las variantes de duración

```python
from qsm.audio import create_duration_variants

# Audio de 1000ms
audio_1000ms = np.random.randn(16000)

# Crear 8 variantes de duración
variants = create_duration_variants(
    audio_1000ms,
    durations_ms=[20, 40, 60, 80, 100, 200, 500, 1000],
    padding_ms=2000,
    sr=16000
)

print(len(variants))  # 8 variantes
print(variants[100].shape)  # (32000,) - 100ms padded to 2000ms
```

## Actualización de `build_conditions.py`

El script ahora usa las nuevas funciones de `qsm.audio.slicing`:

```python
from qsm.audio import (
    extract_segment_center,    # ⭐ NUEVO
    pad_audio_center,          # ⭐ NUEVO
    mix_at_snr,                # noise.py
    apply_bandpass,            # filters.py
    apply_lowpass,             # filters.py
    apply_highpass,            # filters.py
    load_rir_database,         # reverb.py
    apply_rir,                 # reverb.py
)
```

### Simplificación del Código

**Antes**: Definíamos las funciones localmente en `build_conditions.py`

```python
def pad_audio_center(audio, target_samples, ...):
    # 40 líneas de código
    ...

def extract_duration_segment(audio, duration_ms, ...):
    # 30 líneas de código
    ...
```

**Después**: Importamos desde el módulo centralizado

```python
from qsm.audio import extract_segment_center, pad_audio_center

# Usamos directamente
segment = extract_segment_center(audio, dur_ms, sr)
padded = pad_audio_center(segment, 2000, sr, noise_amplitude=0.0001, seed=seed)
```

## Beneficios de esta Reorganización

### 1. Igualdad Conceptual
Todas las transformaciones psicoac uísticas tienen el mismo peso:
- **Duration** ← transformación psicoac uística
- **SNR** ← transformación psicoac uística
- **Filters** ← transformación psicoac uística
- **Reverb** ← transformación psicoac uística

### 2. Consistencia en el Namespace
```python
# Todas las transformaciones desde el mismo lugar
from qsm.audio import (
    extract_segment_center,  # slicing
    mix_at_snr,              # noise
    apply_bandpass,          # filters
    apply_rir,               # reverb
)
```

### 3. Reutilización de Código
Las funciones de slicing ahora están disponibles para cualquier script:
- `build_conditions.py` ✅
- `evaluate_model.py` ✅
- Scripts de análisis futuros ✅
- Notebooks de exploración ✅

### 4. Mejor Documentación
Cada función tiene docstrings claros con ejemplos:
```python
def extract_segment_center(audio, duration_ms, sr=16000):
    """
    Extract a segment of specified duration from the CENTER of audio.

    Example:
        >>> audio_1000ms = np.random.randn(16000)  # 1000ms at 16kHz
        >>> segment_100ms = extract_segment_center(audio_1000ms, 100, sr=16000)
        >>> len(segment_100ms)  # 1600 samples = 100ms
        1600
    """
```

### 5. Testing Más Fácil
Podemos testear las funciones de slicing de forma aislada:
```python
def test_extract_segment_center():
    audio = np.random.randn(16000)  # 1000ms
    segment = extract_segment_center(audio, 100, sr=16000)
    assert len(segment) == 1600  # 100ms at 16kHz
```

## Estructura del Dataset (Sin Cambios)

La estructura del dataset generado sigue siendo la misma:

```
87 clips × 20 variants = 1,740 total variants

Breakdown:
  - Duration:  696 variants (87 × 8)
  - SNR:       522 variants (87 × 6)
  - Band:      261 variants (87 × 3)
  - RIR:       261 variants (87 × 3)
```

## Migración

### Script Actual
✅ `build_conditions.py` - Ya actualizado para usar `qsm.audio.slicing`

### Scripts que NO necesitan cambios
✅ `evaluate_model.py` - No usa funciones de slicing directamente
✅ `prepare_1000ms_manifest.py` - No usa funciones de slicing

### Módulo Original
⚠️ `src/qsm/data/slicing.py` - Mantener por compatibilidad con scripts legacy
- Contiene funciones más complejas (`create_segments`, `FrameTable`, etc.)
- Usado por `make_segments_ava.py` y otros scripts de generación de datasets

## Testing

Probamos que el refactoring funciona correctamente:

```bash
# 1. Test de imports
python -c "from qsm.audio import extract_segment_center, pad_audio_center; print('OK')"
# Output: OK

# 2. Test completo de build_conditions.py
python scripts/build_conditions.py \
    --input_manifest data/processed/qsm_dev_1000ms_only.jsonl \
    --output_dir data/processed/conditions_refactored/ \
    --durations="20,40,60,80,100,200,500,1000" \
    --snr_levels="-10,-5,0,5,10,20" \
    --band_filters="telephony,lp3400,hp300" \
    --rir_root="data/external/RIRS_NOISES/RIRS_NOISES" \
    --rir_metadata="data/external/RIRS_NOISES/rir_metadata.json" \
    --rir_t60_bins="0.0-0.4,0.4-0.8,0.8-1.5" \
    --n_workers 4

# Output:
# Generated 1740 condition variants from 87 clips
# Average variants per clip: 20.0
# ✅ SUCCESS
```

## Conclusión

Esta reorganización pone todas las transformaciones psicoac uísticas en igualdad de condiciones bajo `src/qsm/audio/`, reflejando correctamente que:

1. **Duration** es una transformación psicoac uística tan importante como SNR, filters o reverb
2. Las 4 transformaciones son **independientes** y evaluadas por separado
3. Cada transformación genera su propio conjunto de variantes (8+6+3+3=20)
4. El código es más **limpio**, **reutilizable** y **mantenible**

**Resultado**: Sistema consistente, bien organizado y fácil de entender. 🎯
