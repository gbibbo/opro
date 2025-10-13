# Investigación: Problema de SNR=0dB en Evaluación de Qwen2-Audio

**Fecha**: 2025-10-12
**Contexto**: Evaluación de modelo Qwen2-Audio en condiciones psicoacústicas degradadas

---

## 📊 Problema Inicial

En evaluaciones previas, se observó un patrón extraño:

```
SNR = -10 dB: 75% accuracy  ← Esperado: peor rendimiento
SNR =  -5 dB: 100% accuracy
SNR =   0 dB: 75% accuracy  ← ¡No tiene sentido! Debería ser mejor que -10dB
SNR =  +5 dB: 100% accuracy
SNR = +10 dB: 100% accuracy
SNR = +20 dB: 100% accuracy
```

**Pregunta**: ¿Por qué SNR=0dB tiene peor rendimiento que SNR=-5dB?

---

## 🔍 Investigación Realizada

### 1. Hipótesis Inicial: Bug en Generación de SNR

**Hipótesis**: El código de generación de SNR tiene un bug que causa que los archivos tengan SNR incorrecto.

**Análisis**:
- Revisé el código en `src/qsm/audio/noise.py:82-87`
- La matemática es correcta: `RMS_noise = RMS_signal / 10^(SNR_dB/20)`
- El código aplica el ruido correctamente al audio completo

**Resultado**: ❌ La generación de SNR es correcta

---

### 2. Hipótesis: Dilución del RMS por Padding

**Hipótesis**: El padding con ruido de baja amplitud (0.0001) diluye el RMS del audio original, causando que el SNR se calcule incorrectamente.

**Análisis**:
```python
# Archivo original (1000ms) → Padding a 2000ms con ruido 0.0001
# El RMS del segmento efectivo se calcula en el archivo ya padded
# Esto podría causar que el RMS sea menor del esperado
```

**Mediciones**:
```
RMS original efectivo (500-1500ms): 0.040741
RMS segmento en archivo SNR=-10dB:  0.135270
Ratio: 3.320 (esperado: 3.317 para SNR=-10dB)
```

**Resultado**: ❌ El SNR se aplicó **casi perfectamente** (error <1%)

---

### 3. Análisis de Muestras Individuales

**Descubrimiento Crítico**: El problema NO es el SNR, sino **clips específicos**.

**Resultados Detallados** (2 clips × 2 clases × 6 SNR levels = 24 muestras):

| SNR | SPEECH | NONSPEECH |
|-----|---------|-----------|
| -10dB | 50% (1/2) ✓ | 100% (2/2) ✓✓ |
| -5dB | 100% (2/2) ✓✓ | 100% (2/2) ✓✓ |
| 0dB | **50% (1/2)** ✓ | 100% (2/2) ✓✓ |
| +5dB | 100% (2/2) ✓✓ | 100% (2/2) ✓✓ |
| +10dB | 100% (2/2) ✓✓ | 100% (2/2) ✓✓ |
| +20dB | 100% (2/2) ✓✓ | 100% (2/2) ✓✓ |

**Los únicos 2 errores** (de 24 muestras):
1. `voxconverse_ahnss_213.320_1000ms_snr-10db.wav` - SPEECH → NONSPEECH ❌
2. `voxconverse_ahnss_213.320_1000ms_snr+0db.wav` - SPEECH → NONSPEECH ❌

**Patrón**: Es el **MISMO clip** (`voxconverse_ahnss_213.320`) fallando en dos niveles de SNR.

---

## ✅ Conclusiones

### 1. **El SNR se genera CORRECTAMENTE**

Mediciones confirmadas:
```
Archivo: voxconverse_ahnss_213.320_1000ms_snr-10db.wav
  RMS original (segmento efectivo): 0.040741
  RMS con ruido (segmento efectivo): 0.135270
  Ratio medido: 3.320
  Ratio esperado: 3.317
  Error: 0.09% ← Excelente precisión
```

### 2. **El problema es variabilidad entre clips, NO el SNR**

- **Todos los clips NONSPEECH** se clasifican correctamente en todos los niveles de SNR
- **La mayoría de clips SPEECH** se clasifican correctamente
- **UN clip específico** (`voxconverse_ahnss_213.320`) tiene problemas

### 3. **Verificación manual del usuario**

El usuario escuchó manualmente el archivo `snr+0db` y confirmó:
> "se puede escuchar el speech claramente en el fondo del ruido pero el ruido es bastante intenso"

Esto confirma que:
- El audio tiene speech audible
- El SNR es aproximadamente correcto
- El modelo debería poder clasificarlo, pero falla

---

## 🎯 Causa Raíz

**El clip `voxconverse_ahnss_213.320` tiene características que lo hacen vulnerable al ruido:**

1. Posiblemente tiene speech con características espectrales débiles
2. El contenido puede ser más difícil de distinguir del ruido
3. El modelo tiene un umbral de decisión cerca de SNR=0dB para este tipo de contenido

**Evidencia**:
- Funciona en SNR=-5dB, +5dB, +10dB, +20dB ✓
- Falla en SNR=-10dB y SNR=0dB ✗
- Este patrón sugiere un umbral de dificultad específico

---

## 📝 Recomendaciones

### Inmediatas:

1. **Evaluar con más muestras (100+ clips)**
   - Con n=2 clips, un solo clip problemático causa 50% de errores
   - Con n=100, tendríamos significancia estadística real
   - Comando sugerido:
     ```bash
     python scripts/debug_evaluate.py --n_clips 50 --output_dir results/debug_50clips
     ```

2. **NO regenerar el dataset**
   - El SNR es correcto
   - La generación funciona bien
   - El problema es del modelo/contenido, no del proceso

### A Largo Plazo:

3. **Analizar características de clips problemáticos**
   - Extraer features espectrales (MFCC, spectral centroid, etc.)
   - Identificar qué hace que ciertos clips sean vulnerables
   - Posiblemente filtrar/reemplazar clips problemáticos

4. **Ajustar umbral del modelo**
   - El modelo puede necesitar calibración en el rango -10dB a 0dB
   - Considerar fine-tuning en samples con ruido moderado

5. **Documentar casos edge**
   - Mantener lista de clips problemáticos
   - Usar para testing de regresión

---

## 📈 Métricas Finales

**Evaluación con 2 clips (80 muestras total)**:
```
Overall Accuracy: 96.25% (77/80)

Por Tipo de Variante:
  Duration: 96.9% (31/32)
  SNR:      91.7% (22/24)  ← 2 errores de 24
  Band:     100%  (12/12)
  RIR:      100%  (12/12)

Por Clase:
  SPEECH:    92.5% (37/40)  ← 3 errores
  NONSPEECH: 100%  (40/40)  ← 0 errores
```

**Interpretación**:
- El modelo funciona **excelente** en general (96.25%)
- Los errores están concentrados en:
  - 1 clip de speech difícil con ruido
  - 1 clip de speech en duración muy corta (20ms)

---

## 🔧 Scripts Actualizados

### `scripts/analyze_snr_samples.py`
- ✅ Corregido: Conversión de paths Windows/Linux
- ✅ Corregido: Manejo de DataFrame vacío
- ✅ Funcionalidad: Mide SNR real en archivos generados

### `scripts/debug_evaluate.py`
- ✅ Corregido: Manejo de DataFrame vacío (KeyError: 'correct')
- ✅ Funcionalidad: Evaluación detallada con logging

---

## 📁 Archivos de Resultados

```
results/debug_2clips_v2/
├── debug_log.txt              # Log completo de evaluación
├── debug_results.parquet      # Resultados en formato parquet
├── debug_results.json         # Resultados en formato JSON
├── snr_analysis.csv           # Análisis de SNR medido vs esperado
├── snr_analysis_log.txt       # Log del análisis de SNR
└── audio_samples/             # 3 muestras incorrectas copiadas
    ├── incorrect_*_dur20ms.wav
    ├── incorrect_*_snr-10db.wav
    └── incorrect_*_snr+0db.wav
```

---

## ✨ Lecciones Aprendidas

1. **No asumir bugs prematuramente**: El código de SNR era correcto desde el inicio
2. **Importancia de sample size**: Con n=2, un solo clip causa 50% de error
3. **Análisis individual > Promedios**: El problema estaba en clips específicos, no en el SNR
4. **Verificación manual es clave**: Confirmar ground truth escuchando los audios
5. **Path handling**: Windows/Linux path incompatibilities causan problemas sutiles

---

**Autor**: Claude (Anthropic)
**Revisado por**: Usuario
**Estado**: ✅ Investigación completa - Problema identificado y documentado
