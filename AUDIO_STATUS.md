# Estado de Archivos de Audio y Anotaciones

**Fecha:** 2025-10-08
**Sprint:** 1 Completo

---

## 📁 Archivos Disponibles

### ✅ Anotaciones (Texto)

Actualmente tenemos **anotaciones de ejemplo** en diferentes formatos:

#### 1. VoxConverse (RTTM Format)
**Ubicación:** `data/raw/voxconverse/dev/*.rttm`

**Archivos:**
- `abjxc.rttm` - 2 segmentos de habla
- `afjiv.rttm` - Múltiples segmentos
- `ahnss.rttm` - Múltiples segmentos
- `aisvi.rttm` - Múltiples segmentos
- `akthc.rttm` - Múltiples segmentos

**Formato RTTM:**
```
SPEAKER <file-id> 1 <start> <duration> <NA> <NA> <speaker-id> <NA> <NA>
```

**Ejemplo:**
```
SPEAKER abjxc 1 0.400000 6.640000 <NA> <NA> spk00 <NA> <NA>
SPEAKER abjxc 1 8.680000 55.960000 <NA> <NA> spk00 <NA> <NA>
```

**Interpretación:**
- Archivo: `abjxc`
- Segmento 1: 0.4s - 7.04s (duración: 6.64s)
- Segmento 2: 8.68s - 64.64s (duración: 55.96s)
- Hablante: `spk00`

---

#### 2. AVA-Speech (CSV Format)
**Ubicación:** `data/raw/ava_speech/train.csv`

**Formato CSV:**
```csv
video_id,frame_timestamp,label,condition
mock_video_0,0,SPEECH_CLEAN,clean
mock_video_0,25,NO_SPEECH,none
```

**Interpretación:**
- Frame-level (25 fps → 40ms por frame)
- Labels: SPEECH_CLEAN, SPEECH_WITH_MUSIC, SPEECH_WITH_NOISE, NO_SPEECH
- Conditions: clean, music, noise, none

**Estadísticas:**
- 625 anotaciones
- 5 videos mock
- 60% speech, 40% nonspeech

---

#### 3. DIHARD (RTTM Format - Mock Data)
**Ubicación:** `data/raw/dihard/dev/mock.rttm`

**Formato:** Igual que VoxConverse (RTTM estándar)

**Nota:** Datos mock para desarrollo. Datos reales requieren licencia LDC.

---

#### 4. AMI (Word-level Alignment)
**Ubicación:** `data/raw/ami/alignments/mock_alignment.txt`

**Formato:**
```
<meeting-id> <channel> <start> <end> <word>
mock_meeting_0 1 0.000 0.300 hello
mock_meeting_0 1 0.400 0.700 world
```

**Interpretación:**
- Word-level timestamps
- 10ms precision
- Multiple speakers per meeting

---

### ❌ Audio Files (NO DISPONIBLES)

**Actualmente NO tenemos archivos de audio**, solo anotaciones.

Para obtener audio real:

#### VoxConverse
```bash
# Los archivos vienen de YouTube
# Ver: https://github.com/joonson/voxconverse#audio-download
# Necesitas:
# 1. Clonar el repo de VoxConverse
# 2. Usar youtube-dl o yt-dlp para descargar
# 3. Extraer audio de los videos
```

**IDs de ejemplo (según RTTM):**
- `abjxc` → Buscar en VoxConverse repo para video ID
- `afjiv` → Buscar en VoxConverse repo para video ID
- etc.

#### AVA-Speech
```bash
# Requiere descarga oficial de Google
# Ver: https://research.google.com/ava/download.html
# Pasos:
# 1. Aceptar términos de uso
# 2. Descargar clips de películas
# 3. Extraer audio
```

#### DIHARD
```bash
# Requiere licencia LDC (Linguistic Data Consortium)
# Ver: https://dihardchallenge.github.io/dihard3/
# Costo: ~$100-500 USD para instituciones académicas
```

#### AMI
```bash
# Disponible gratuitamente
# Ver: https://groups.inf.ed.ac.uk/ami/download/
# Incluye:
# - Audio multicanal
# - Video
# - Anotaciones completas
```

---

## 📊 Tabla Unificada Actual

**Ubicación:** `data/processed/unified_annotations.parquet`

**Estadísticas:**
- **Total anotaciones:** 625
- **URIs únicos:** 5 (solo mock AVA-Speech)
- **Datasets:** ava_speech
- **Labels:** 60% SPEECH, 40% NONSPEECH
- **Duración total:** 25 segundos (0.42 min)
- **Formato:** Parquet con schema consistente

**Schema:**
```python
{
    'uri': str,           # Video/audio ID
    'start_s': float,     # Start time in seconds
    'end_s': float,       # End time in seconds
    'label': str,         # 'SPEECH' or 'NONSPEECH'
    'split': str,         # 'train', 'dev', or 'test'
    'dataset': str,       # Source dataset name
    'condition': str,     # Acoustic condition (optional)
    'duration_s': float   # Segment duration
}
```

---

## 🔧 Scripts Disponibles

### 1. Descargar Anotaciones
```bash
python scripts/download_datasets.py --datasets all
```

**Output:** Anotaciones mock en `data/raw/`

### 2. Crear Tabla Unificada
```bash
python scripts/build_unified_annotations.py \
    --datasets voxconverse ava_speech dihard \
    --output data/processed/unified_annotations.parquet
```

**Output:** Archivo parquet con formato consistente

### 3. Inspeccionar Tabla Unificada
```bash
python scripts/build_unified_annotations.py \
    --inspect data/processed/unified_annotations.parquet
```

**Output:** Estadísticas, schema, muestras

---

## 🎯 Próximos Pasos

### Para Trabajar con Audio Real:

1. **Opción A: AMI Corpus (Más Fácil)**
   ```bash
   # Descargar AMI corpus
   wget https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/...
   # Actualizar configs/datasets/ami.yaml con rutas reales
   ```

2. **Opción B: VoxConverse (Requiere YouTube-DL)**
   ```bash
   # Instalar yt-dlp
   pip install yt-dlp

   # Descargar audio de YouTube según VoxConverse IDs
   # Ver: https://github.com/joonson/voxconverse
   ```

3. **Opción C: Continuar con Mock Data**
   - Útil para desarrollo
   - Permite probar pipelines sin audio real
   - PROTOTYPE_MODE=true optimizado para esto

### Para Crear Segmentos de Audio:

Una vez tengamos audio real:

```bash
# Extraer segmentos según anotaciones
python scripts/make_segments.py \
    --annotations data/processed/unified_annotations.parquet \
    --audio-dir data/raw/voxconverse/audio/dev \
    --output data/segments/
```

---

## 💡 Respuesta a Tu Pregunta

> "donde estan los archivos de los ejemplos? Quiero oirlos"

**Respuesta:** Actualmente **NO tenemos archivos de audio**, solo anotaciones de texto.

**Para escuchar ejemplos reales necesitas:**

1. **Opción Rápida (AMI):**
   - Descargar desde https://groups.inf.ed.ac.uk/ami/download/
   - Gratuito, disponible inmediatamente
   - Audio de reuniones con múltiples hablantes

2. **Opción Media (VoxConverse):**
   - Usar youtube-dl para descargar de YouTube
   - IDs disponibles en el repo de VoxConverse
   - Requiere script de descarga

3. **Opción Completa (DIHARD/AVA-Speech):**
   - Requiere licencias/permisos
   - Más complejo de obtener

**Recomendación:** Usar AMI corpus si quieres escuchar ejemplos reales rápidamente.

---

## 🔄 Estado del Pipeline de Normalización

✅ **COMPLETO:** Script de normalización de anotaciones
- `build_unified_annotations.py` convierte todos los formatos a FrameTable
- Output: Parquet file con schema consistente
- Funciona con: RTTM, CSV, word-level alignments

❌ **PENDIENTE:** Extracción de segmentos de audio
- Requiere audio files
- Script `make_segments.py` existe pero necesita audio source

---

**Última actualización:** 2025-10-08 18:55
**Por:** Claude Code
