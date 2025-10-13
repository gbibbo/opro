# Setup para RTX 4070 Laptop (8GB VRAM)

**Tu Hardware**: RTX 4070 Laptop, 8GB VRAM
**Estrategia**: Carga/descarga alternada de modelos

---

## 🎯 Estrategia Optimizada

Tu GPU tiene **8GB VRAM**, que es suficiente para correr UN modelo a la vez pero no DOS simultáneamente.

**Solución**: Script optimizado que alterna entre modelos:

```
Iteración N:
  1. Carga Llama 3.2-3B (generador) ──> Genera 3 prompts ──> Descarga
  2. Carga Qwen2-Audio (evaluador) ──> Evalúa 3 prompts ──> Descarga
  3. Repite
```

**Ventajas**:
- ✅ Funciona perfectamente con 8GB
- ✅ No necesita API keys (100% local)
- ✅ Usa Llama 3.2 (heterogeneidad como pediste)

**Desventaja**:
- ⏱️ Más lento (~50-60 min por iteración vs 40 min con 2 modelos simultáneos)

---

## 🚀 Comando para Prototipar (5 iteraciones)

```bash
# Test en tu laptop (3-4 horas)
python scripts/run_opro_local_8gb.py \
    --n_iterations 5 \
    --early_stopping 3 \
    --output_dir results/sprint9_opro_laptop_test
```

**Tiempo estimado**: 3-4 horas (perfecto para prototipar)

---

## 📊 Uso de VRAM por Fase

### Fase 1: Generar prompts
```
┌─────────────────────────┐
│ Llama 3.2-3B Instruct   │  ~2.5 GB VRAM
│ (4-bit quantization)    │
└─────────────────────────┘
```

### Fase 2: Evaluar prompts
```
┌─────────────────────────┐
│ Qwen2-Audio-7B          │  ~5 GB VRAM
│ (4-bit quantization)    │
└─────────────────────────┘
```

**Pico máximo**: ~5 GB (nunca supera 8GB) ✅

---

## ⏱️ Tiempo por Iteración (8GB)

| Fase | Tiempo | Descripción |
|------|--------|-------------|
| **Carga Llama** | ~2 min | Primera vez descarga de HuggingFace |
| **Genera 3 prompts** | ~3 min | LLM genera variaciones |
| **Descarga Llama** | ~10 seg | Libera VRAM |
| **Carga Qwen2-Audio** | ~2 min | Ya descargado localmente |
| **Evalúa 3 prompts** | ~30 min | 10 min × 3 prompts |
| **Descarga Qwen2-Audio** | ~10 seg | Libera VRAM |
| **Total** | **~40 min** | Por iteración |

**Test completo (5 iteraciones)**: ~3.5 horas

---

## 🖥️ Para Servidor Remoto (Producción)

Cuando pases al servidor con más VRAM (>12GB), usa el script completo:

```bash
# En servidor potente (2 modelos simultáneos, más rápido)
python scripts/run_opro_local.py \
    --optimizer_llm "meta-llama/Llama-3.1-8B-Instruct" \
    --n_iterations 50 \
    --early_stopping 5 \
    --output_dir results/sprint9_opro_servidor
```

**Diferencias con servidor**:
- Carga ambos modelos a la vez (no alterna)
- Llama 3.1-8B en lugar de 3.2-3B (mejor calidad)
- Más iteraciones (50 vs 5)
- Más rápido (~30 min por iteración)

---

## 📁 Workflow Completo: Laptop → Servidor

### 1. Prototipar en Laptop (8GB) - HOY

```bash
# Test rápido (5 iteraciones, ~3.5 horas)
python scripts/run_opro_local_8gb.py \
    --n_iterations 5 \
    --output_dir results/sprint9_opro_laptop_test
```

**Objetivo**: Validar que todo funciona

### 2. Transferir al Servidor

```bash
# Copiar scripts y datos necesarios
scp -r scripts/ usuario@servidor:/path/to/OPRO_Qwen/
scp -r data/processed/conditions_final/ usuario@servidor:/path/to/OPRO_Qwen/data/processed/
```

### 3. Ejecutar en Servidor (>12GB)

```bash
# SSH al servidor
ssh usuario@servidor

# Optimización completa (30-50 iteraciones, ~20 horas)
cd /path/to/OPRO_Qwen
python scripts/run_opro_local.py \
    --optimizer_llm "meta-llama/Llama-3.1-8B-Instruct" \
    --n_iterations 50 \
    --early_stopping 5 \
    --output_dir results/sprint9_opro_servidor

# Déjalo correr con screen o tmux
screen -S opro
python scripts/run_opro_local.py ...
# Ctrl+A D para detach
```

### 4. Descargar Resultados

```bash
# En tu laptop
scp -r usuario@servidor:/path/to/OPRO_Qwen/results/sprint9_opro_servidor/ ./results/
```

---

## 🔍 Monitorear Durante Ejecución

### En laptop (otra terminal):

```bash
# Ver uso de GPU
watch -n 1 nvidia-smi

# Ver mejor prompt actual
tail -f results/sprint9_opro_laptop_test/best_prompt.txt

# Ver progreso (cuenta líneas en jsonl)
wc -l results/sprint9_opro_laptop_test/opro_prompts.jsonl
# Divide por 3 para obtener iteraciones completadas
```

---

## 🎓 Modelos Recomendados

### Para tu Laptop (8GB):
- **Llama 3.2-3B-Instruct** ✅ (por defecto, ~2.5GB)
- Qwen2.5-3B-Instruct (~2.5GB)
- Phi-3-mini-4k-instruct (~2GB)

### Para Servidor (>12GB):
- **Llama 3.1-8B-Instruct** ✅ (mejor calidad, ~5GB)
- Qwen2.5-7B-Instruct (~4.5GB)
- Mistral-7B-Instruct (~4.5GB)

---

## ✅ Checklist para Empezar

Antes de correr en laptop:

- [x] GPU: RTX 4070 (8GB) ✓
- [x] VRAM libre: ~7GB ✓
- [ ] Tiempo disponible: 3-4 horas
- [ ] Internet: Solo para primera descarga de Llama 3.2-3B (~1.5GB)

**Todo listo! Puedes empezar:**

```bash
python scripts/run_opro_local_8gb.py \
    --n_iterations 5 \
    --output_dir results/sprint9_opro_laptop_test
```

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory" durante generación

**Causa**: Llama 3.2-3B todavía muy grande

**Solución**: Usa Phi-3-mini (más pequeño)
```bash
python scripts/run_opro_local_8gb.py \
    --optimizer_llm "microsoft/Phi-3-mini-4k-instruct" \
    --n_iterations 5
```

### Error: "CUDA out of memory" durante evaluación

**Causa**: Qwen2-Audio no cabe (raro con 8GB)

**Solución**: No debería pasar con 4-bit. Revisa procesos en GPU:
```bash
nvidia-smi
# Cierra otros programas que usen GPU
```

### El script se detiene entre fases

**Es NORMAL**: Está descargando un modelo y cargando el otro. Verás:
```
Unloading optimizer LLM...
Loading Qwen2-Audio (evaluator)...
```

Esto toma ~2 minutos. **No lo interrumpas.**

---

## 📊 Resultados Esperados

Con 5 iteraciones (prototipo):

| Métrica | Baseline | Esperado | Mejora |
|---------|----------|----------|--------|
| BA_clip | 0.690 | 0.700-0.720 | +0.01 a +0.03 |

Con 50 iteraciones (servidor):

| Métrica | Baseline | Esperado | Mejora |
|---------|----------|----------|--------|
| BA_clip | 0.690 | 0.720-0.750 | +0.03 a +0.06 |

---

## 🎯 Próximos Pasos

1. **HOY - Laptop**: Corre prototipo (5 iter, 3.5h)
   ```bash
   python scripts/run_opro_local_8gb.py --n_iterations 5
   ```

2. **Revisar**: Verifica que funciona y genera prompts razonables
   ```bash
   cat results/sprint9_opro_laptop_test/best_prompt.txt
   ```

3. **Mañana - Servidor**: Transferir y correr optimización completa (50 iter, 20h)

4. **Después**: Refit psychometric curves + test eval

---

**¿Listo para empezar el prototipo en tu laptop?** 🚀
