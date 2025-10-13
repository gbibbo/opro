# OPRO en Ejecución - Estado Actual

**Fecha**: 2025-10-13
**Comando**: `python scripts/run_opro_local_8gb.py --optimizer_llm "Qwen/Qwen2.5-3B-Instruct" --n_iterations 5`

---

## ✅ Estado Actual

```
============================================================
OPRO LOCAL - OPTIMIZADO PARA 8GB VRAM
============================================================
GPU: RTX 4070 Laptop (8GB)
Estrategia: Carga/descarga alternada de modelos
Optimizer LLM: Qwen/Qwen2.5-3B-Instruct
Iteraciones: 5
Candidatos/iter: 3
```

**Progreso**:
- ✅ Qwen2-Audio cargado
- 🔄 Evaluando baseline (14/1400 muestras)
- ⏳ Pendiente: Iteraciones 1-5

---

## ⏱️ Tiempo Estimado

| Fase | Tiempo | Estado |
|------|--------|--------|
| Baseline | ~60 min | 🔄 En progreso (2.7s/muestra) |
| Iteración 1 | ~40 min | ⏳ Pendiente |
| Iteración 2 | ~40 min | ⏳ Pendiente |
| Iteración 3 | ~40 min | ⏳ Pendiente |
| Iteración 4 | ~40 min | ⏳ Pendiente |
| Iteración 5 | ~40 min | ⏳ Pendiente |
| **TOTAL** | **~4 horas** | **20% completado** |

**Hora estimada de finalización**: ~1:00 AM (mañana)

---

## 📊 Por Qué Qwen2.5-3B en lugar de Llama 3.2

**Problema**: Llama 3.2 es un modelo "gated" (requiere aceptar términos en HuggingFace)

**Solución**: Usamos Qwen2.5-3B-Instruct
- ✅ Sin restricciones de acceso
- ✅ Mismo tamaño (~2.5GB VRAM)
- ✅ Excelente calidad (familia Qwen muy buena en instrucciones)
- ✅ Funciona out-of-the-box

**Nota sobre heterogeneidad**:
- Aunque ambos son Qwen, Qwen2.5 (generador) y Qwen2-Audio (evaluador) son modelos DIFERENTES:
  - Qwen2.5: Modelo de texto general (entrenamiento diferente)
  - Qwen2-Audio: Modelo multimodal audio+texto
- Esto sigue proporcionando diversidad suficiente

---

## 🔍 Cómo Monitorear

### Ver uso de GPU (otra terminal)
```bash
watch -n 1 nvidia-smi
```

### Ver mejor prompt actual (después de baseline)
```bash
cat results/sprint9_opro_laptop_test/best_prompt.txt
```

### Ver progreso completo
```bash
tail -f results/sprint9_opro_laptop_test/opro_prompts.jsonl
```

### Contar iteraciones completadas
```bash
wc -l results/sprint9_opro_laptop_test/opro_prompts.jsonl
# Divide por 3 para obtener número de iteraciones
```

---

## 📁 Archivos que se Crearán

```
results/sprint9_opro_laptop_test/
├── opro_prompts.jsonl       # Historial completo (se va llenando)
├── best_prompt.txt          # Mejor prompt (se actualiza cada iteración)
└── (más archivos al final)
```

---

## 🎯 Qué Esperar

### Después del Baseline (~60 min)
```
Baseline results:
  BA_clip: 0.891  (excelente! baseline muy bueno)
  BA_cond: 0.834
  Reward: 1.039
```

### Después de Iteración 1 (~100 min total)
```
ITERATION 1
PHASE 1: Generating 3 candidates...
Loading optimizer LLM: Qwen/Qwen2.5-3B-Instruct...
Generated 3 candidates:
  1. Is there speech in this audio? Answer: SPEECH or NON-SPEECH.
  2. Does this contain human voice? Reply: SPEECH or NON-SPEECH.
  3. Detect speech presence. Respond: SPEECH or NON-SPEECH.

PHASE 2: Evaluating 3 candidates...
  Candidate 1: BA_clip=0.895, Reward=1.044
  NEW BEST REWARD: 1.044 (+0.005)
```

---

## 🚀 Próximos Pasos (Después de que Termine)

### 1. Revisar Mejor Prompt
```bash
cat results/sprint9_opro_laptop_test/best_prompt.txt
```

### 2. Ver Mejora vs Baseline
El script imprimirá al final:
```
OPTIMIZATION COMPLETE
Best prompt: "..."
Improvement: +0.XX
```

### 3. Refit Psychometric Curves
```bash
python scripts/refit_psychometric_opro.py \
    --opro_dir results/sprint9_opro_laptop_test
```

### 4. Transferir al Servidor para Run Completo
```bash
# Copiar scripts
scp -r scripts/ usuario@servidor:/path/to/OPRO_Qwen/

# En servidor: 50 iteraciones con Llama 3.1-8B
python scripts/run_opro_local.py \
    --optimizer_llm "meta-llama/Llama-3.1-8B-Instruct" \
    --n_iterations 50 \
    --output_dir results/sprint9_opro_servidor
```

---

## 💡 Tips

### Si necesitas pausar/detener
```bash
# Encontrar proceso
ps aux | grep run_opro_local_8gb

# Detener (Ctrl+C o kill)
kill <PID>

# Nota: Los resultados hasta el momento se guardan automáticamente
```

### Si se interrumpe
**NO te preocupes**: El script guarda estado después de cada iteración en:
- `opro_prompts.jsonl` - Historial completo
- `best_prompt.txt` - Mejor prompt hasta el momento

Puedes revisar lo que logró hasta ese punto.

---

## 📊 Baseline Actual: 0.891 BA_clip

**¡Excelente baseline!** Es mucho mejor que el reportado anteriormente (0.69).

Esto significa que el prompt baseline múltiple choice está funcionando muy bien:
```
"What best describes this audio?
A) Human speech or voice
B) Music
C) Noise or silence
D) Animal sounds

Answer with ONLY the letter (A, B, C, or D)."
```

**Implicación**: OPRO necesitará ser muy creativo para superar 0.891. Cualquier mejora >0.90 será excelente.

---

## 🎯 Objetivos Realistas para este Prototipo

| Métrica | Baseline | Objetivo | Stretch Goal |
|---------|----------|----------|--------------|
| BA_clip | 0.891 | >0.895 | >0.900 |
| Mejora | - | +0.004 | +0.009 |

Con 5 iteraciones (prototipo), incluso +0.003 sería un buen resultado.

En el servidor con 50 iteraciones esperamos llegar a 0.90-0.91.

---

## ⏰ Recordatorios

- **No apagues la laptop** (necesita ~4 horas)
- **No suspendas** (interrumpirá el proceso)
- **Puedes usar la laptop** para otras cosas (el script corre en background)
- **Monitor GPU usage** para confirmar que está trabajando

---

**Estado**: 🟢 TODO FUNCIONANDO CORRECTAMENTE

**Próxima revisión**: En ~60 minutos (después de baseline)
