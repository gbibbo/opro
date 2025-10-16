# OPRO: Mejoras Implementadas y Pendientes

**Fecha**: 2025-10-13
**Versión actual**: `run_opro_local_8gb_fixed.py`
**Estado**: Corriendo iteración 1 (funcionando bien)

---

## ✅ Mejoras YA Implementadas

### 1. Sanitización de Candidatos
**Archivo**: `run_opro_local_8gb_fixed.py`

```python
def sanitize_prompt(prompt: str) -> Tuple[str, bool]:
    # Bloquea tokens especiales
    forbidden_tokens = ['<|audio_bos|>', '<|AUDIO|>', '<|audio_eos|>', ...]

    # Valida longitud (10-300 chars)
    # Requiere keywords "SPEECH" y "NON-SPEECH"
    # Limpia espacios múltiples
```

**Resultado**: ✓ No más crashes por tokens inválidos

### 2. Circuit Breaker
**Archivo**: `run_opro_local_8gb_fixed.py` (línea ~330)

```python
try:
    ba_clip, ba_cond, metrics = evaluate_prompt(...)
except Exception as e:
    print(f"ERROR evaluating candidate: {e}")
    continue  # Salta al siguiente candidato
```

**Resultado**: ✓ Una falla no para toda la optimización

### 3. Gestión de Memoria Mejorada
**Archivo**: `run_opro_local_8gb_fixed.py`

```python
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(0.5)  # Da tiempo a GPU para liberar
```

**Resultado**: ✓ Menos OOM errors en 8GB

### 4. Meta-Prompt Sin Tokens Especiales
**Archivo**: `run_opro_local_8gb_fixed.py` (línea ~240)

```python
# Limpia prompts en ejemplos
clean_prompt = candidate.prompt.replace('<|audio_bos|><|AUDIO|><|audio_eos|>', '').strip()
```

**Resultado**: ✓ Optimizador no aprende a generar tokens especiales

### 5. Fallback Inteligente
**Archivo**: `run_opro_local_8gb_fixed.py` (función `parse_and_sanitize_candidates`)

```python
if len(candidates_clean) == 0:
    # Fallback: variaciones simples del baseline
    candidates_clean = [
        "Does this audio contain human speech? Answer: SPEECH or NON-SPEECH.",
        "Is there speech in this audio? Reply: SPEECH or NON-SPEECH.",
        ...
    ]
```

**Resultado**: ✓ Nunca falla por falta de candidatos

---

## 🚧 Mejoras Pendientes (Prioridad Alta)

### 1. Constrained Decoding en Evaluador
**Status**: ⏳ Script creado (`evaluate_prompt_constrained.py`)
**Qué falta**: Integrar en el loop de OPRO

**Implementación**:
```python
# En run_opro_local_8gb_fixed.py
from evaluate_prompt_constrained import evaluate_prompt_constrained

# Reemplazar evaluate_prompt() con evaluate_prompt_constrained()
ba_clip, ba_cond, ba_hard, metrics = evaluate_prompt_constrained(
    prompt=prompt,
    use_constrained=True,  # Force "SPEECH" o "NONSPEECH"
)
```

**Beneficio**:
- ✅ Salida 100% parseable
- ✅ No más "UNKNOWN" labels
- ✅ Reduce varianza de formato

**Referencia**: [Hugging Face - Constrained Decoding](https://huggingface.co/docs/transformers/main_classes/text_generation)

### 2. Chat Templating Oficial de Qwen2-Audio
**Status**: ⏳ Parcialmente implementado
**Qué falta**: Verificar que `qwen_audio.py` usa el flujo canon

**Verificación necesaria** en `src/qsm/models/qwen_audio.py`:
```python
# ✓ CORRECTO (debe estar así):
conversation = [
    {"role": "system", "content": self.system_prompt},
    {
        "role": "user",
        "content": [
            {"type": "audio"},  # Audio se pasa separado
            {"type": "text", "text": self.user_prompt},  # Solo texto del usuario
        ],
    },
]

text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = self.processor(text=text, audios=audio, sampling_rate=sr, return_tensors="pt")
```

**Referencia**: [Qwen2-Audio Docs](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)

### 3. Reward Enfocado en Condiciones Duras
**Status**: ⏳ Script preparado (`evaluate_prompt_constrained.py` tiene `ba_hard`)
**Qué falta**: Usar `ba_hard` en reward

**Cambio en reward**:
```python
# ACTUAL (run_opro_local_8gb_fixed.py)
R = BA_clip + 0.25×BA_cond - 0.05×len/100

# PROPUESTO (task-aligned)
R = BA_clip + 0.5×BA_hard + 0.1×BA_rest - 0.05×len/100
```

**Dónde**: `run_opro_local_8gb_fixed.py`, función `compute_reward()`

**Beneficio**:
- Empuja mejoras donde el modelo es frágil
- Alineado con objetivos psicofísicos (DT75, SNR-75)

### 4. Successive Halving (Cribado Rápido)
**Status**: ⏳ No implementado
**Complejidad**: Media
**Impacto**: Alto (reduce tiempo 3-5×)

**Algoritmo**:
```python
# Genera 8-12 candidatos (en lugar de 3)
candidates = generate_candidates(n=12, temperature=0.8)

# Mini-dev: 20% de muestras (280 en lugar de 1400)
mini_dev_df = split_df.sample(frac=0.2, random_state=seed)

# Evalúa TODOS en mini-dev
for prompt in candidates:
    reward_mini = evaluate_prompt(prompt, mini_dev_df)

# Selecciona top-3
top_3 = sorted(candidates, key=lambda c: c.reward)[:3]

# Re-evalúa top-3 en dev completo
for prompt in top_3:
    reward_full = evaluate_prompt(prompt, split_df)
```

**Beneficio**:
- Tiempo por iteración: ~40 min → ~15 min
- Explora más candidatos con el mismo presupuesto

### 5. Deduplicación de Candidatos
**Status**: ⏳ No implementado
**Complejidad**: Baja
**Impacto**: Medio

**Implementación**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_candidates(candidates, memory, threshold=0.9):
    """Rechaza candidatos muy similares a memoria."""
    if len(memory) == 0:
        return candidates

    # TF-IDF de candidatos + memoria
    all_prompts = [c.prompt for c in memory] + candidates
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_prompts)

    # Similaridad
    mem_vectors = vectors[:len(memory)]
    cand_vectors = vectors[len(memory):]
    similarities = cosine_similarity(cand_vectors, mem_vectors)

    # Filtrar
    unique = []
    for i, prompt in enumerate(candidates):
        if similarities[i].max() < threshold:
            unique.append(prompt)

    return unique
```

**Dónde**: Después de `parse_and_sanitize_candidates()`

### 6. CPU Offloading del Optimizador
**Status**: ⏳ No implementado
**Complejidad**: Media
**Impacto**: Alto (en 8GB VRAM)

**Implementación**:
```python
# En LocalLLMGenerator.__init__()
from accelerate import cpu_offload

model_kwargs = {
    "device_map": "auto",
    "max_memory": {0: "4GB", "cpu": "16GB"},  # Limita GPU a 4GB
    "offload_folder": "offload_cache",
}
```

**Beneficio**:
- Evaluador residente en GPU (5GB)
- Optimizador offloadeado a CPU/disk
- Evita recargas completas

**Referencia**: [Accelerate - Big Models](https://huggingface.co/docs/accelerate/package_reference/big_modeling)

---

## 📊 Mejoras Estadísticas (Post-Optimización)

### 7. Test de McNemar
**Status**: ⏳ No implementado
**Cuándo**: Después de seleccionar best prompt
**Dónde**: Nuevo script `scripts/statistical_tests.py`

```python
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(baseline_preds, opro_preds, y_true):
    """Test de McNemar para diferencias pareadas."""
    # Tabla de contingencia
    baseline_correct = (baseline_preds == y_true)
    opro_correct = (opro_preds == y_true)

    # McNemar
    table = pd.crosstab(baseline_correct, opro_correct)
    result = mcnemar(table, exact=True)

    return result.pvalue, result.statistic
```

**Referencia**: [McNemar's Test](https://en.wikipedia.org/wiki/McNemar%27s_test)

### 8. Bootstrap Pareado
**Status**: ⏳ No implementado
**Cuándo**: Después de test set evaluation
**Dónde**: `scripts/statistical_tests.py`

```python
def bootstrap_paired_ba(baseline_df, opro_df, n_bootstrap=1000):
    """Bootstrap de Δ(BA_clip) con CI95."""
    deltas = []

    for _ in range(n_bootstrap):
        # Resample clips con reemplazo
        clip_ids = baseline_df["clip_id"].unique()
        sampled_ids = np.random.choice(clip_ids, size=len(clip_ids), replace=True)

        base_sample = baseline_df[baseline_df["clip_id"].isin(sampled_ids)]
        opro_sample = opro_df[opro_df["clip_id"].isin(sampled_ids)]

        ba_base = balanced_accuracy_score(base_sample["y_true"], base_sample["y_pred"])
        ba_opro = balanced_accuracy_score(opro_sample["y_true"], opro_sample["y_pred"])

        deltas.append(ba_opro - ba_base)

    # Percentiles
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)

    return np.mean(deltas), (ci_lower, ci_upper)
```

---

## 🎯 Prioridades para Implementar AHORA

**Mientras corre la optimización actual**:

1. ✅ **Verificar chat templating** en `qwen_audio.py` (5 min)
2. ⏳ **Integrar constrained decoding** (15 min)
3. ⏳ **Actualizar reward** para incluir `ba_hard` (5 min)

**Después de que termine (4 horas)**:

4. ⏳ **Revisar resultados** y decidir si vale la pena successive halving
5. ⏳ **Si resultados buenos**: Transferir al servidor y correr 50 iteraciones
6. ⏳ **Si resultados malos**: Implementar successive halving + dedupe y re-correr

**Para el paper**:

7. ⏳ **McNemar + bootstrap** en test set
8. ⏳ **Pseudo-R²** de curvas psicométricas

---

## 📁 Archivos Creados

| Archivo | Estado | Propósito |
|---------|--------|-----------|
| `run_opro_local_8gb.py` | ⚠️ Buggy | Versión inicial (con crashes) |
| `run_opro_local_8gb_fixed.py` | ✅ Running | Versión sanitizada (corriendo ahora) |
| `evaluate_prompt_constrained.py` | ✅ Ready | Evaluador con constrained decoding |
| `statistical_tests.py` | ⏳ TODO | McNemar + bootstrap |

---

## 🔍 Cómo Monitorear Tu Run Actual

```bash
# Ver mejor prompt actual
cat results/sprint9_opro_laptop_fixed/best_prompt.txt

# Ver progreso
tail -f results/sprint9_opro_laptop_fixed/opro_prompts.jsonl | wc -l
# Divide por 3 para obtener número de candidatos evaluados

# Ver GPU
nvidia-smi
```

---

## 📊 Resultado Esperado (Tu Run Actual)

**Baseline**: BA_clip = 0.891 (excelente)

**Con sanitización** (tu versión actual):
- Esperado: BA_clip = 0.893-0.897 (+0.002 a +0.006)
- Con 5 iteraciones: Mejora modesta pero validación del pipeline

**Con TODAS las mejoras** (servidor, 50 iteraciones):
- Esperado: BA_clip = 0.900-0.910 (+0.009 a +0.019)
- DT75: 35ms → 28-32ms
- SNR-75: -2.9dB → -4 a -5dB

---

## ✅ Checklist para Completar Sprint 9

- [x] Implementar sanitización
- [x] Circuit breaker
- [x] Gestión memoria mejorada
- [x] Meta-prompt limpio
- [x] Evaluador con constrained decoding (script listo)
- [ ] Integrar constrained en OPRO loop
- [ ] Reward con ba_hard
- [ ] Successive halving (opcional, si tiempo)
- [ ] Deduplicación (opcional)
- [ ] CPU offloading (opcional)
- [ ] McNemar test (post-optimización)
- [ ] Bootstrap CI (post-optimización)
- [ ] Comparison report
- [ ] Git tag v2.0-opro-baseline

---

**Estado actual**: Tu run está funcionando bien con las mejoras críticas. Las mejoras adicionales pueden esperar a ver los resultados.
