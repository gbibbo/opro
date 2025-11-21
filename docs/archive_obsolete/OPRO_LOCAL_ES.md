# OPRO con LLM Local - Guía Rápida (Español)

**Optimización de prompts completamente LOCAL - sin API keys ni costos**

---

## ✅ Ventajas del LLM Local

- **Gratis**: Todo corre en tu GPU, sin APIs pagadas
- **Privacidad**: Tu datos nunca salen de tu máquina
- **Sin límites**: Sin rate limits ni cuotas mensuales
- **Offline**: No necesitas conexión a Internet (después de descargar el modelo)

---

## 🚀 Inicio Rápido

### Test de 5 iteraciones (~3-4 horas)

```bash
python scripts/run_opro_local.py \
    --n_iterations 5 \
    --output_dir results/sprint9_opro_local_test \
    --early_stopping 3
```

### Optimización completa (30-50 iteraciones, ~20-30 horas)

```bash
python scripts/run_opro_local.py \
    --optimizer_llm "Qwen/Qwen2.5-7B-Instruct" \
    --n_iterations 50 \
    --candidates_per_iter 3 \
    --early_stopping 5 \
    --output_dir results/sprint9_opro_local
```

---

## 📊 Qué Ocurre Durante la Ejecución

### Fase 1: Carga de modelos
```
LOADING EVALUATOR MODEL (Qwen2-Audio)
  Model: Qwen/Qwen2-Audio-7B-Instruct
  Device: cuda
  4-bit quantization: True

LOADING OPTIMIZER LLM
  Model: Qwen/Qwen2.5-7B-Instruct
  Device: cuda
  4-bit quantization: True
```

**Tiempo**: ~5-10 minutos (descarga modelos si es primera vez)

### Fase 2: Evaluación del baseline
```
Evaluating baseline prompt...
Evaluating on dev set (1400 variants)...
Baseline reward: 0.8234
```

**Tiempo**: ~30 minutos (evalúa todos los clips del dev set)

### Fase 3: Optimización OPRO (por iteración)
```
Iteration 1: Generating 3 candidates...
Generated 3 candidates:
  1. Is there human speech in this audio? Answer: SPEECH or NON-SPEECH.
  2. Detect speech presence. Respond with: SPEECH or NON-SPEECH.
  3. Does this contain a human voice? Reply: SPEECH or NON-SPEECH.

Evaluating candidate 1/3...
  Results: BA_clip=0.715, BA_cond=0.680, Reward=0.892

NEW BEST REWARD: 0.892 (+0.068)
```

**Tiempo por iteración**: ~30-40 minutos
- Generación de prompts: ~2-3 minutos
- Evaluación de 3 candidatos: ~30 minutos (10 min cada uno)

---

## 🔧 Configuración Avanzada

### Usar un LLM diferente para generar prompts

```bash
# Llama 3 (8B)
python scripts/run_opro_local.py \
    --optimizer_llm "meta-llama/Llama-3-8B-Instruct" \
    --n_iterations 5

# Qwen2.5-14B (mejor calidad, más lento)
python scripts/run_opro_local.py \
    --optimizer_llm "Qwen/Qwen2.5-14B-Instruct" \
    --n_iterations 5

# Qwen2.5-3B (más rápido, menor calidad)
python scripts/run_opro_local.py \
    --optimizer_llm "Qwen/Qwen2.5-3B-Instruct" \
    --n_iterations 5
```

### Usar CPU en lugar de GPU (MUY lento)

```bash
python scripts/run_opro_local.py \
    --optimizer_device cpu \
    --evaluator_device cpu \
    --n_iterations 2  # Solo 2 iteraciones, tardará horas
```

**Nota**: No recomendado - cada iteración puede tardar 2-3 horas en CPU.

### Más candidatos por iteración (más exploración)

```bash
python scripts/run_opro_local.py \
    --candidates_per_iter 5 \  # 5 en lugar de 3
    --n_iterations 30
```

**Nota**: Aumenta el tiempo por iteración proporcionalmente (~50 min por iteración con 5 candidatos).

---

## 💾 Uso de Memoria GPU

### Configuración por defecto (2 modelos cargados simultáneamente)

- **Qwen2-Audio-7B** (evaluador): ~4-5 GB VRAM
- **Qwen2.5-7B** (optimizador): ~4-5 GB VRAM
- **Total necesario**: ~10-12 GB VRAM

**GPU mínima recomendada**: RTX 3060 (12GB), RTX 4070 (12GB), o superior

### Si tienes menos VRAM (<12 GB)

Puedes descargar/recargar modelos entre fases (más lento pero funciona):

**Opción 1**: Modifica `run_opro_local.py` para descargar el evaluador después de cada evaluación
**Opción 2**: Usa modelos más pequeños:

```bash
python scripts/run_opro_local.py \
    --optimizer_llm "Qwen/Qwen2.5-3B-Instruct" \  # 3B en lugar de 7B
    --n_iterations 5
```

Con Qwen2.5-3B: ~8-9 GB VRAM total

---

## 📁 Estructura de Resultados

Después de la optimización, encontrarás:

```
results/sprint9_opro_local/
├── opro_prompts.jsonl              # Historial completo (todas las iteraciones)
├── opro_memory.json                 # Top-10 mejores prompts
├── opro_history.json                # Curva de recompensa
├── best_prompt.txt                  # MEJOR PROMPT (úsalo aquí)
├── best_metrics.json                # Métricas del mejor prompt
└── dev_predictions.parquet          # Predicciones con mejor prompt
```

**Archivo clave**: `best_prompt.txt` - Contiene el prompt optimizado

---

## 🎯 Resultados Esperados

Basado en el paper OPRO (mejoras de 8-50% en diversas tareas):

| Métrica | Baseline | Esperado OPRO | Mejora |
|---------|----------|---------------|---------|
| **BA_clip** | 0.690 | 0.720-0.750 | +0.03 a +0.06 |
| **DT75** | 34.8 ms | 25-30 ms | -5 a -10 ms |
| **SNR-75 (1000ms)** | -2.9 dB | -4 a -5 dB | -1 a -2 dB |

---

## ⏱️ Tiempo Total Estimado

### Test corto (5 iteraciones)
- Carga de modelos: ~10 min
- Baseline: ~30 min
- 5 iteraciones × 40 min: ~200 min
- **Total**: ~3-4 horas

### Optimización completa (50 iteraciones con early stopping ~30 iter)
- Carga de modelos: ~10 min
- Baseline: ~30 min
- 30 iteraciones × 40 min: ~1200 min
- **Total**: ~20-24 horas

**Recomendación**: Déjalo correr durante la noche.

---

## 🔍 Monitoreo Durante la Ejecución

### Ver el mejor prompt actual

```bash
cat results/sprint9_opro_local/best_prompt.txt
```

### Ver la curva de recompensa

```bash
cat results/sprint9_opro_local/opro_history.json | jq '.best_reward_per_iteration'
```

### Ver cuántas iteraciones lleva

```bash
cat results/sprint9_opro_local/opro_prompts.jsonl | wc -l
# Divide por 3 (si candidates_per_iter=3) para obtener número de iteraciones
```

---

## 🐛 Solución de Problemas

### Error: "CUDA out of memory"

**Solución 1**: Cierra otros programas que usen GPU
```bash
nvidia-smi  # Ver qué está usando la GPU
```

**Solución 2**: Usa un modelo optimizador más pequeño
```bash
python scripts/run_opro_local.py \
    --optimizer_llm "Qwen/Qwen2.5-3B-Instruct" \
    --n_iterations 5
```

**Solución 3**: Reduce batch size internamente (edita `qwen_audio.py` si es necesario)

### El LLM optimizador no genera prompts en el formato esperado

Esto es normal - el parser tiene fallbacks. Verás warnings como:
```
Warning: Failed to parse structured output, attempting fallback...
```

**Solución**: Continúa normalmente - el fallback parser extraerá los prompts.

### La optimización no mejora después de varias iteraciones

Esto activa el early stopping automáticamente. Es **esperado y correcto**.

```
Early stopping: No improvement for 5 iterations
OPTIMIZATION COMPLETE
```

---

## 🔄 Después de la Optimización

Una vez completo, continúa con:

### 1. Refitear curvas psicométricas

```bash
python scripts/refit_psychometric_opro.py \
    --opro_dir results/sprint9_opro_local
```

### 2. Evaluar en test set (UNA VEZ)

```bash
python scripts/evaluate_opro_test.py \
    --opro_dir results/sprint9_opro_local
```

### 3. Revisar reporte de comparación

```bash
cat results/sprint9_opro_local/comparison_report.md
```

---

## 📊 Comparación: Local vs API

| Aspecto | Local (Qwen2.5-7B) | API (Claude 3.5) |
|---------|-------------------|------------------|
| **Costo** | $0 (gratis) | ~$0.53 |
| **Tiempo/iteración** | ~40 min | ~30 min |
| **VRAM necesaria** | ~10-12 GB | 0 GB |
| **Privacidad** | 100% privado | Datos enviados a API |
| **Calidad prompts** | Buena | Excelente |
| **Requisitos** | GPU potente | API key |

**Recomendación**:
- Si tienes GPU ≥12GB: Usa local (gratis, privado)
- Si tu GPU es pequeña (<8GB): Usa API (más barato que upgrade de GPU)
- Si necesitas máxima calidad: Usa API con Claude 3.5 Sonnet

---

## 🎓 Modelos Locales Recomendados

### Para generar prompts (optimizer_llm):

1. **Qwen/Qwen2.5-7B-Instruct** (recomendado por defecto)
   - Tamaño: ~4.5 GB VRAM
   - Calidad: Excelente
   - Velocidad: Rápida

2. **meta-llama/Llama-3-8B-Instruct**
   - Tamaño: ~5 GB VRAM
   - Calidad: Excelente
   - Velocidad: Rápida

3. **Qwen/Qwen2.5-14B-Instruct** (mejor calidad, más lento)
   - Tamaño: ~8 GB VRAM
   - Calidad: Superior
   - Velocidad: Moderada

4. **Qwen/Qwen2.5-3B-Instruct** (para GPUs pequeñas)
   - Tamaño: ~2 GB VRAM
   - Calidad: Buena
   - Velocidad: Muy rápida

---

## ✅ Lista de Verificación

Antes de empezar:
- [ ] GPU con ≥10-12 GB VRAM (o usar modelo 3B)
- [ ] Espacio en disco: ~20 GB (para descargar modelos)
- [ ] Tiempo disponible: 3-4 horas (test) o 20-24 horas (completo)
- [ ] Paciencia: Es un proceso largo pero automático

---

**¿Listo para empezar?**

```bash
# Test corto (5 iteraciones, ~3-4 horas)
python scripts/run_opro_local.py \
    --n_iterations 5 \
    --output_dir results/sprint9_opro_local_test \
    --early_stopping 3
```

**¡Todo gratis, todo local, todo privado!** 🎉
