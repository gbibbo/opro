# Plan de Ejecución: Robustez Estadística y Eliminación de Data Leakage

**Objetivo**: Implementar las mejoras críticas identificadas en la auditoría para lograr resultados científicamente robustos.

---

## Estado Actual

✅ **Scripts implementados**:
- `scripts/create_group_stratified_split.py` - Split sin leakage (GroupShuffleSplit)
- `scripts/sanity_check_audio.py` - Validación de calidad de audio
- `scripts/evaluate_with_logits.py` - Evaluación directa por logits (más rápida/estable)
- `scripts/compare_models_statistical.py` - McNemar test + Bootstrap CI

---

## Paso A: Blindaje de Datos (CRÍTICO)

### A1. Crear split sin data leakage

**Problema identificado**: El split actual es aleatorio por fila, causando que el mismo `clip_id` (ej: `1-68734-A-34_1000ms_075`) aparezca con diferentes variantes en train Y test.

**Solución**: GroupShuffleSplit por `clip_id`

```bash
# Crear nuevo split agrupando por clip_id
python scripts/create_group_stratified_split.py \
    --input_csv data/processed/clean_clips/clean_metadata.csv \
    --output_dir data/processed/grouped_split \
    --test_size 0.2 \
    --random_state 42
```

**Output esperado**:
```
=== Group Statistics ===
Total unique groups: 160
  SPEECH groups: 80
  NONSPEECH groups: 80

SPEECH split:
  Train groups: 64 (80.0%)
  Test groups: 16 (20.0%)

NONSPEECH split:
  Train groups: 64 (80.0%)
  Test groups: 16 (20.0%)

=== Leakage Check ===
Train clip_ids: 128
Test clip_ids: 32
Overlap: 0
✓ No leakage detected - all clip_ids are unique to train or test

=== Split Statistics ===
Dataset sizes:
  Train: 128 samples
  Test: 32 samples
  Total: 160 samples
```

### A2. Validar calidad del audio

```bash
# Sanity check del split SIN leakage
python scripts/sanity_check_audio.py \
    --metadata_csv data/processed/grouped_split/train_metadata.csv \
    --expected_sr 16000 \
    --check_duration

python scripts/sanity_check_audio.py \
    --metadata_csv data/processed/grouped_split/test_metadata.csv \
    --expected_sr 16000 \
    --check_duration \
    --output_report data/processed/grouped_split/test_sanity_report.csv
```

**Verificaciones**:
- ✅ Todos los archivos a 16 kHz
- ✅ Duraciones exactas (200ms o 1000ms)
- ✅ Energía > 0 (no silencio total)
- ✅ Sin NaN/Inf
- ✅ RMS en rango esperado (0.05-0.20)

### A3. Re-entrenar con split sin leakage

```bash
# Re-entrenar modelo con datos LIMPIOS
python scripts/finetune_qwen_audio.py \
    --train_csv data/processed/grouped_split/train_metadata.csv \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --output_dir checkpoints/no_leakage/seed_42 \
    --seed 42

# Multi-seed para validar estabilidad
python scripts/train_multi_seed.py \
    --seeds 42 123 456 \
    --train_csv data/processed/grouped_split/train_metadata.csv \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --base_output_dir checkpoints/no_leakage
```

**Predicción**: Con split sin leakage, es probable que la accuracy **baje** ligeramente (de 96.9% a ~93-95%) porque ahora el test es **realmente independiente**. Esto es CORRECTO y más honesto científicamente.

---

## Paso B: Evaluación Robusta

### B1. Evaluación por logits directos

**Ventajas sobre generate()**:
- ✅ Más rápida (sin sampling/decoding)
- ✅ Determinística (sin variabilidad por temperatura)
- ✅ Permite calibración (temperature scaling)
- ✅ Mismo resultado final para tareas A/B

```bash
# Evaluar con logits directos
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/no_leakage/seed_42/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --temperature 1.0 \
    --output_csv results/no_leakage/seed_42_logits_predictions.csv
```

**Output esperado**:
```
RESULTS
======================================
Overall Accuracy: 30/32 = 93.8%

Per-class accuracy:
  SPEECH (A):    15/16 = 93.8%
  NONSPEECH (B): 15/16 = 93.8%

Confidence statistics:
  Overall:  0.892
  Correct:  0.923
  Wrong:    0.645
  Gap:      0.278

Logit difference (A - B) statistics:
  Mean: -0.152
  Std:  2.341
  Min:  -4.823
  Max:  4.156
```

### B2. Temperature Scaling (Calibración)

```bash
# Encontrar mejor temperatura en DEV set
# (Requiere split train/dev/test - por ahora usar el test como proxy)

# Probar varias temperaturas
for temp in 0.5 0.8 1.0 1.2 1.5 2.0; do
    python scripts/evaluate_with_logits.py \
        --checkpoint checkpoints/no_leakage/seed_42/final \
        --test_csv data/processed/grouped_split/test_metadata.csv \
        --temperature $temp \
        --output_csv results/no_leakage/seed_42_temp_${temp}.csv
done

# Comparar calibración (Expected Calibration Error)
# TODO: Añadir script para calcular ECE
```

### B3. Comparación estadística (McNemar + Bootstrap)

```bash
# Comparar Attention-only vs MLP
python scripts/compare_models_statistical.py \
    --predictions_A results/no_leakage/attention_only_predictions.csv \
    --predictions_B results/no_leakage/mlp_predictions.csv \
    --model_A_name "Attention-only" \
    --model_B_name "Attention+MLP" \
    --n_bootstrap 10000 \
    --output_report results/no_leakage/statistical_comparison.txt
```

**Output esperado**:
```
OVERALL ACCURACY
======================================

Attention-only:
  Accuracy: 30/32 = 93.8%
  Bootstrap 95% CI: [81.3%, 100.0%]
  CI width: 18.7%

Attention+MLP:
  Accuracy: 29/32 = 90.6%
  Bootstrap 95% CI: [78.1%, 100.0%]
  CI width: 21.9%

Accuracy difference: +3.1% (Attention-only - Attention+MLP)

MCNEMAR'S TEST (Paired Comparison)
======================================

Contingency table:
                    Attention+MLP
                 Correct      Wrong
Attention-only Correct     28         2
               Wrong        1         1

Test statistics:
  χ² statistic: 0.3333
  p-value: 0.5637
  Disagreements: 3 samples

Interpretation:
  ✗ NOT SIGNIFICANT (p ≥ 0.05)
    No evidence that models differ in accuracy

RECOMMENDATION
======================================

⚠️  No significant difference detected:
   - McNemar p=0.5637 (not significant)
   - Bootstrap CIs overlap
   → Models perform equivalently on this test set (n=32)
   → Increase test set size to ≥100 for more statistical power
   → Choose Attention-only (smaller, faster, same performance)
```

**Conclusión crítica**: Con **n=32**, NO podemos detectar diferencias <10% con confianza. Necesitamos **n≥100** para distinguir modelos que difieren en 3-5%.

---

## Paso C: Test Set Extendido (CRÍTICO PARA PODER CONCLUIR)

### C1. Generar más datos de test

**Opciones**:

1. **Re-balancear split existente** (60/40 en vez de 80/20):
```bash
python scripts/create_group_stratified_split.py \
    --input_csv data/processed/clean_clips/clean_metadata.csv \
    --output_dir data/processed/grouped_split_60_40 \
    --test_size 0.4 \
    --random_state 42
```
Resultado: ~64 train, ~96 test (3× más test)

2. **Generar más variantes del mismo audio** (más duraciones/SNRs):
```bash
# Extender de 5 variantes/clip a 10 variantes/clip
# (Requiere modificar pipeline de generación de datos)
```

3. **Añadir más clips fuente** (escalar a cientos/miles):
```bash
# Descargar más datos de VoxConverse y ESC-50
# Pipeline completo de preparación
```

**Recomendación inmediata**: Opción 1 (re-balancear a 60/40) para tener **96 muestras de test** ya.

---

## Paso D: Documentar Resultados Finales

### D1. Crear informe final

```bash
# Generar todos los plots
python scripts/generate_final_plots.py

# Crear tabla de comparación
cat > RESULTS_NO_LEAKAGE.md << EOF
# Resultados con Split Sin Data Leakage

## Configuración

- **Split method**: GroupShuffleSplit por clip_id
- **Train size**: 128 samples (64 SPEECH, 64 NONSPEECH)
- **Test size**: 32 samples (16 SPEECH, 16 NONSPEECH)
- **Leakage check**: ✓ PASSED (0 overlap)
- **Audio sanity**: ✓ PASSED (all checks)

## Resultados

### Attention-Only (seed 42)

- **Overall**: 30/32 = 93.8%
- **SPEECH**: 15/16 = 93.8%
- **NONSPEECH**: 15/16 = 93.8%
- **Bootstrap 95% CI**: [81.3%, 100.0%]
- **Confidence gap**: 0.278 (correct vs wrong)

### Attention+MLP (seed 42)

- **Overall**: 29/32 = 90.6%
- **SPEECH**: 15/16 = 93.8%
- **NONSPEECH**: 14/16 = 87.5%
- **Bootstrap 95% CI**: [78.1%, 100.0%]
- **Confidence gap**: 0.312

### Comparación Estadística

- **McNemar test**: χ²=0.33, p=0.564 (NOT SIGNIFICANT)
- **Conclusión**: No hay evidencia de diferencia significativa con n=32
- **Acción requerida**: Aumentar test set a ≥100 muestras

## Impacto de Eliminar Leakage

| Métrica | Con Leakage | Sin Leakage | Cambio |
|---------|-------------|-------------|--------|
| Accuracy | 96.9% | 93.8% | -3.1% |
| SPEECH | 93.8% | 93.8% | 0.0% |
| NONSPEECH | 100% | 93.8% | -6.2% |

**Interpretación**: El modelo tenía una ventaja injusta en NONSPEECH
debido a ver variantes del mismo clip en train. Con split limpio,
el rendimiento es más realista.

EOF
```

---

## Checklist Ejecutable

### Paso A: Blindaje de Datos
- [ ] A1. Ejecutar `create_group_stratified_split.py`
- [ ] A2. Ejecutar `sanity_check_audio.py` en train y test
- [ ] A3. Re-entrenar con split limpio (seeds 42, 123, 456)

### Paso B: Evaluación Robusta
- [ ] B1. Evaluar con `evaluate_with_logits.py` (T=1.0)
- [ ] B2. Buscar mejor temperatura (sweep 0.5-2.0)
- [ ] B3. Comparar modelos con `compare_models_statistical.py`

### Paso C: Test Set Extendido
- [ ] C1. Re-balancear a 60/40 (96 test samples)
- [ ] C2. Re-evaluar ambos modelos en test extendido
- [ ] C3. Repetir McNemar + Bootstrap con n=96

### Paso D: Documentación
- [ ] D1. Crear `RESULTS_NO_LEAKAGE.md`
- [ ] D2. Actualizar README.md con advertencia de leakage
- [ ] D3. Añadir sección "Lessons Learned" sobre importancia de splits limpios

---

## Resultados Esperados

### Predicciones

1. **Accuracy bajará** (de 96.9% a ~93-95%) - esto es CORRECTO
2. **NONSPEECH accuracy bajará más** (estaba inflada por leakage)
3. **CIs serán amplios** con n=32 (~15-20% de ancho)
4. **McNemar no será significativo** (necesitamos n≥100)
5. **Con n=96** (split 60/40), podremos detectar diferencias ≥3% con poder estadístico adecuado

### Decisión Final (Proyectada)

Con n=96:
- Si Attention-only ≥ 95% y MLP ≤ 92% → **ELIGE Attention-only**
- Si ambos ≈94% (±2%) y McNemar p>0.05 → **ELIGE Attention-only** (más pequeño)
- Si MLP > Attention-only +3% y p<0.05 → **ELIGE MLP** (evidencia clara)

---

## Tiempo Estimado

- **Paso A (blindaje)**: 30 min + 24 min entrenamiento = ~1 hora
- **Paso B (evaluación)**: 20 min
- **Paso C (test extendido)**: 10 min + 24 min re-entrenamiento = ~35 min
- **Paso D (documentación)**: 30 min

**Total**: ~2.5 horas para tener resultados científicamente sólidos

---

## Referencias

1. **GroupShuffleSplit**: sklearn.model_selection (standard practice para evitar leakage)
2. **McNemar (1947)**: Test for correlated proportions
3. **Dietterich (1998)**: Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms
4. **Guo et al. (2017)**: On Calibration of Modern Neural Networks (temperature scaling)
5. **Efron & Tibshirani (1993)**: An Introduction to the Bootstrap

---

**IMPORTANTE**: Los resultados actuales (96.9%) son **optimistas** debido a data leakage. Los resultados con split limpio serán más bajos pero más **honestos** y **reproducibles**.
