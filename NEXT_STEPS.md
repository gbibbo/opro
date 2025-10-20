# Próximos Pasos Recomendados

**Fecha**: 2025-10-20
**Estado Actual**: Multi-seed training completado con varianza cero (96.9% en todos los seeds)

---

## Resumen de Resultados Actuales

### Multi-Seed Training (Attention-Only, Re-entrenado)

**Todos los 5 seeds dieron resultados idénticos:**

| Seed | Overall | SPEECH | NONSPEECH | Training Loss |
|------|---------|--------|-----------|---------------|
| 42   | 96.9%   | 93.8%  | 100.0%    | 0.2860        |
| 123  | 96.9%   | 93.8%  | 100.0%    | 0.2871        |
| 456  | 96.9%   | 93.8%  | 100.0%    | 0.2858        |
| 789  | 96.9%   | 93.8%  | 100.0%    | 0.2858        |
| 2024 | 96.9%   | 93.8%  | 100.0%    | 0.2839        |

**Estadísticas:**
- Media: 96.9% ± 0.0%
- 95% CI: [96.9%, 96.9%]
- Todos fallan en la misma muestra SPEECH (15/16 correctos)
- Todos aciertan todas las muestras NONSPEECH (16/16)

### ¿Por qué varianza cero?

**Causa:** Test set muy pequeño (32 muestras)
- Solo 16 SPEECH, 16 NONSPEECH
- Solo hay 1-2 "muestras difíciles" en SPEECH
- Todos los seeds convergen al mismo mínimo local
- 15/16 SPEECH = 93.8% (siempre)

**Esto NO es un bug** - es una consecuencia natural de:
1. Dataset pequeño (poca granularidad)
2. Arquitectura LoRA muy estable (loss masking funciona excelente)
3. Training muy determinista (mismo proceso → mismo resultado)

### Comparación: Attention-Only vs MLP (Seed 42)

| Config | Overall | SPEECH | NONSPEECH | Params | Tamaño |
|--------|---------|--------|-----------|--------|--------|
| **Attention-only (nuevo)** | **96.9%** | 93.8% | **100%** | 20.7M | 84MB |
| MLP (anterior) | 96.9% | **100%** | 93.8% | 43.9M | 168MB |

**Observación:**
- **Accuracy total idéntica** (96.9%)
- **Trade-off:** Attention-only mejor en NONSPEECH, MLP mejor en SPEECH
- Con solo 32 muestras, imposible distinguir estadísticamente

---

## Plan Recomendado (2 Carriles en Paralelo)

### 🎯 Carril A: Evaluación Robusta (PRIORITARIO)

**Objetivo:** Blindar la evaluación con estadística sólida

#### Paso 1: Ampliar Test Set (150-200 muestras)

**Por qué:**
- Con 32 muestras: 1 error = 3.1% cambio en accuracy
- Con 200 muestras: 1 error = 0.5% cambio en accuracy
- Poder estadístico suficiente para detectar diferencias de 2-3% con p<0.05

**Cómo ejecutar:**

```bash
# 1. Generar metadata con diseño factorial
python scripts/generate_extended_test_set.py \
    --voxconverse_csv data/raw/voxconverse_metadata.csv \
    --musan_csv data/raw/musan_metadata.csv \
    --output_dir data/processed/extended_test_clips \
    --n_samples_per_class 100 \
    --seed 42

# Esto genera:
# - 200 muestras totales (100 SPEECH, 100 NONSPEECH)
# - Diseño factorial: 5 duraciones × 5 SNRs × 4 samples/condición
# - Balanceado: 50% SPEECH, 50% NONSPEECH
# - Output: data/processed/extended_test_clips/extended_test_metadata.csv
```

**Nota:** Necesitarás implementar `scripts/process_extended_test_clips.py` para procesar el audio según la metadata.

#### Paso 2: Re-evaluar con Metodología Robusta

```bash
# Evaluar los mejores modelos con el test ampliado
python scripts/run_robust_evaluation.py \
    --models attention_only_seed42 mlp_seed42 baseline \
    --checkpoints \
        checkpoints/qwen2_audio_speech_detection_multiseed/seed_42/final \
        checkpoints/with_mlp_seed_42/final \
        Qwen/Qwen2-Audio-7B-Instruct \
    --test_csv data/processed/extended_test_clips/extended_test_metadata.csv \
    --results_dir results/extended_evaluation
```

#### Paso 3: Decisión Basada en Estadística

**Si attention_only ≈ MLP (p ≥ 0.05):**
- ✅ **Usar attention-only** (20.7M params, 84MB)
- Razón: Más simple, 2× más pequeño, misma performance
- Continuar con OPRO sobre attention-only

**Si MLP > attention_only (p < 0.05):**
- ✅ **Usar MLP** (43.9M params, 168MB)
- Razón: Mejora estadísticamente significativa
- Vale la pena el 2× en tamaño por +2-3% accuracy

---

### 🚀 Carril B: OPRO sobre Modelo Fine-Tuned (EN PARALELO)

**Objetivo:** Optimizar el prompt sobre el modelo fine-tuned congelado

**Hipótesis:** Fine-tuning + OPRO > Fine-tuning solo

#### Implementación

Necesitarás crear `scripts/opro_optimize_finetuned.py` (ver [README_ROBUST_EVALUATION.md](README_ROBUST_EVALUATION.md) para detalles).

**Comando de uso:**

```bash
# Optimizar prompt sobre modelo FT (attention-only seed 42)
python scripts/opro_optimize_finetuned.py \
    --checkpoint checkpoints/qwen2_audio_speech_detection_multiseed/seed_42/final \
    --dev_csv data/processed/normalized_clips/dev_metadata.csv \
    --test_csv data/processed/extended_test_clips/extended_test_metadata.csv \
    --n_iterations 50 \
    --output_dir results/opro_finetuned
```

---

## Resumen: ¿Qué Ejecutar Ahora?

### Orden Sugerido (Prioridad Alta → Baja)

1. **[URGENTE] Ampliar test set a 200 muestras**
   ```bash
   python scripts/generate_extended_test_set.py \
       --n_samples_per_class 100 \
       --output_dir data/processed/extended_test_clips
   ```
   - Implementar `process_extended_test_clips.py` (basado en `normalize_clips.py`)
   - Tiempo estimado: 2-3 horas desarrollo + 30 min procesamiento

2. **Re-evaluar con estadística robusta**
   ```bash
   python scripts/run_robust_evaluation.py \
       --models attention_only mlp \
       --test_csv data/processed/extended_test_clips/extended_test_metadata.csv
   ```
   - Tiempo estimado: 15-20 min (evaluación con logits es rápida)

3. **Decidir modelo final** (attention-only vs MLP)
   - Basado en McNemar p-value
   - Si p ≥ 0.05 → attention-only (más simple)
   - Si p < 0.05 y MLP mejor → MLP

4. **[OPCIONAL] OPRO sobre FT**
   - Solo si quieres exprimir últimos 1-2% de accuracy

5. **[OPCIONAL] Qwen3-Omni baseline**
   - Solo para comparación académica

---

## Timeline Estimado

### Plan Corto (1 día)
1. ✅ Ampliar test set (2-3 horas)
2. ✅ Re-evaluar attention vs MLP (20 min)
3. ✅ Decidir modelo final (lectura de estadísticas)
4. ✅ Documentar resultados

**Entregable:** Modelo validado estadísticamente con 96-97% accuracy ± CI

### Plan Medio (2-3 días)
1. Todo del plan corto
2. ✅ OPRO sobre modelo FT (6-8 horas training)
3. ✅ Validación final en test ampliado
4. ✅ Comparación con Qwen3-Omni

**Entregable:** Modelo optimizado con 97-98% accuracy + comparación SOTA

### Plan Completo (1 semana)
1. Todo del plan medio
2. ✅ Hyperparameter grid search (LoRA rank, alpha, LR)
3. ✅ Escalado de dataset (train: 500-1k samples)
4. ✅ Análisis estratificado (performance por duration, SNR)
5. ✅ Paper-ready results con tablas y estadísticas

**Entregable:** Publicación científica con resultados sólidos

---

## Archivos de Referencia

**Scripts creados (listos para usar):**
- ✅ `scripts/generate_extended_test_set.py` - Genera metadata de 200 samples
- ✅ `scripts/run_robust_evaluation.py` - Pipeline completo con bootstrap + McNemar
- ✅ `scripts/test_with_logit_scoring.py` - Evaluación rápida con logits
- ✅ `scripts/compare_models_mcnemar.py` - Test estadístico pareado

**Scripts por implementar:**
- ⏳ `scripts/process_extended_test_clips.py` - Procesar audio según metadata
- ⏳ `scripts/opro_optimize_finetuned.py` - OPRO sobre modelo congelado

**Documentación:**
- ✅ `README_ROBUST_EVALUATION.md` - Guía completa de metodología estadística
- ✅ `NEXT_STEPS.md` - Este documento
- ✅ `RESULTS_MLP_COMPARISON.md` - Resultados MLP vs attention-only

---

**Última actualización:** 2025-10-20
**Estado:** Infraestructura lista, esperando decisión de usuario
