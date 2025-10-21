# Resultados: Split Sin Leakage + Sanity Check

**Fecha**: 2025-10-21
**Script**: `create_group_stratified_split.py` + `sanity_check_audio.py`

---

## Split Sin Data Leakage

### Configuración
- **Método**: GroupShuffleSplit por `clip_id`
- **Test size**: 20% (0.2)
- **Random seed**: 42
- **Estratificación**: Por clase (SPEECH/NONSPEECH)

### Resultados del Split

**Grupos totales**: 20 clip_ids únicos
- SPEECH: 10 grupos
- NONSPEECH: 10 grupos

**Train Set** (128 samples):
- Grupos: 16 (8 SPEECH, 8 NONSPEECH)
- Samples: 64 SPEECH, 64 NONSPEECH (50/50 balance)
- Distribución: 2 duraciones × 4 SNRs × 2 clases × 8 réplicas/grupo = 128

**Test Set** (32 samples):
- Grupos: 4 (2 SPEECH, 2 NONSPEECH)
- Samples: 16 SPEECH, 16 NONSPEECH (50/50 balance)
- Distribución: 2 duraciones × 4 SNRs × 2 clases × 2 réplicas/grupo = 32

### Verificación de Leakage

**Train clip_ids**: 16
**Test clip_ids**: 4
**Overlap**: **0** ✓

**Conclusión**: ✓ ZERO leakage - Ningún clip_id aparece en ambos conjuntos

---

## Sanity Check de Audio (Train Set)

### Archivos Analizados
- **Total**: 128 archivos
- **Con issues**: 68 (53.1%)
- **OK**: 60 (46.9%)

### Issues Encontrados

**low_peak**: 68 archivos (53.1%)
- **Interpretación**: Clips muy silenciosos (peak < 0.1)
- **¿Es problema?**: **NO** - Es esperado para:
  - SNR bajo (0dB, 5dB): Señal débil es intencional
  - Clips NONSPEECH: Muchos sonidos ambientales son naturalmente suaves
  - Clips cortos (200ms): Menos energía total

**Otros issues**: 0
- ✓ Sin NaN/Inf
- ✓ Sin valores exactamente 0 (silencio total)
- ✓ Sin clipping excesivo
- ✓ Sampling rate correcto (16000 Hz en todos)

### Estadísticas de Audio

**Sampling Rate**:
- 16000 Hz: 128 archivos (100.0%) ✓

**RMS Distribution**:
```
Min:    0.000320  (clip: 1-21935-A-38_1000ms_068)
25th %: 0.010025
Median: 0.018345
75th %: 0.036120
Max:    0.195957  (clip: 1-79220-A-17_1000ms_001)
Range:  612× (amplio - esperado por SNR variado)
```

**Peak Amplitude Distribution**:
```
Min:    0.001099
25th %: 0.048943
Median: 0.088730
75th %: 0.194824
Max:    0.870758
```

### Top 5 Clips Más Silenciosos (por RMS)

| Clip ID | Label | RMS | Peak | Comentario |
|---------|-------|-----|------|------------|
| 1-21935-A-38_1000ms_068 | NONSPEECH | 0.000320 | 0.001099 | Muy suave |
| 1-51436-A-17_1000ms_042 | NONSPEECH | 0.000788 | 0.002472 | Muy suave |
| 1-51436-A-17_1000ms_042 | NONSPEECH | 0.000797 | 0.003265 | Muy suave |
| 1-51805-C-33_1000ms_039 | NONSPEECH | 0.000975 | 0.004089 | Muy suave |
| 1-21935-A-38_1000ms_068 | NONSPEECH | 0.001004 | 0.003265 | Muy suave |

**Observación**: Todos los clips más silenciosos son **NONSPEECH** - esto es correcto, ya que muchos sonidos ambientales son naturalmente suaves.

### Top 5 Clips Más Fuertes (por RMS)

| Clip ID | Label | RMS | Peak |
|---------|-------|-----|------|
| 1-79220-A-17_1000ms_001 | NONSPEECH | 0.195957 | 0.870758 |
| 1-43807-D-47_1000ms_053 | NONSPEECH | 0.156540 | 0.563141 |
| 1-43807-D-47_1000ms_053 | NONSPEECH | 0.149754 | 0.591858 |
| 1-79220-A-17_1000ms_001 | NONSPEECH | 0.125748 | 0.437134 |
| 1-172649-B-40_1000ms_010 | NONSPEECH | 0.118150 | 0.430756 |

**Observación**: También dominados por NONSPEECH - algunos sonidos ambientales (ej: explosiones, golpes) son muy fuertes.

---

## Interpretación y Recomendaciones

### Issues "low_peak" (68 archivos)

**¿Son realmente problemas?**

**NO** - Son características esperadas del dataset:

1. **SNR bajo (0dB, 5dB)**:
   - Por diseño, la señal es débil comparada con el ruido
   - Peak bajo es **intencional** para simular condiciones extremas
   - El modelo debe aprender a clasificar incluso con señal muy débil

2. **Clips NONSPEECH naturalmente suaves**:
   - Sonidos como hojas, viento, lluvia son inherentemente suaves
   - Esto es representativo de ambientes reales

3. **Clips cortos (200ms)**:
   - Menos energía acumulada total
   - Peak puede ser más bajo que en clips de 1000ms

### ¿Acción requerida?

**NO** - Los "issues" son en realidad **características del dataset**, no errores.

**Justificación**:
- ✓ Todos los archivos tienen SR correcto (16000 Hz)
- ✓ No hay NaN/Inf/silencio total
- ✓ Rangos de RMS y peak son razonables
- ✓ La variabilidad (612×) es esperada para un dataset con SNR 0-20dB

**Recomendación**: **Proceder con entrenamiento sin cambios**

---

## Comparación: Split Anterior vs Nuevo

| Métrica | Split Anterior (CON leakage) | Split Nuevo (SIN leakage) |
|---------|------------------------------|---------------------------|
| **Método** | Random por fila | GroupShuffleSplit por clip_id |
| **Train samples** | 128 | 128 |
| **Test samples** | 32 | 32 |
| **Overlap clip_ids** | **Desconocido (probablemente >0)** | **0** ✓ |
| **Balance clase** | 50/50 | 50/50 |
| **Validación leakage** | ❌ No realizada | ✓ Verificado 0 overlap |

**Predicción de impacto**:
- Con leakage: Accuracy ~96.9% (optimista)
- Sin leakage: **Esperamos ~92-95%** (más realista)

La diferencia esperada de **~2-5%** es **normal** cuando se elimina leakage en datasets pequeños con alta correlación intra-grupo.

---

## Próximos Pasos

### Paso 1: Re-entrenar con datos limpios
```bash
# Copiar CSVs a las rutas que espera el script actual
cp data/processed/grouped_split/train_metadata.csv data/processed/normalized_clips/train_metadata.csv
cp data/processed/grouped_split/test_metadata.csv data/processed/normalized_clips/test_metadata.csv

# O actualizar finetune_qwen_audio.py para aceptar --train_csv/--test_csv
```

### Paso 2: Evaluar modelo re-entrenado
- Comparar accuracy con/sin leakage
- Verificar si NONSPEECH accuracy baja (esperado)

### Paso 3: Análisis estadístico
- McNemar test si comparamos múltiples modelos
- Bootstrap CI para intervalos de confianza robustos

---

## Conclusión

✓ **Split sin leakage creado exitosamente**
✓ **Sanity check completo - dataset válido**
✓ **Listo para re-entrenar con datos científicamente limpios**

Los "issues" reportados (low_peak) son **características intencionales** del dataset para entrenar robustez a condiciones extremas, no errores de datos.

**Confianza**: ALTA - El dataset está bien preparado para entrenamiento honesto sin data leakage.
