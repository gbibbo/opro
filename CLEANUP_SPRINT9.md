# Sprint 9 Cleanup - Scripts Organizados

**Fecha**: 2025-10-13

---

## Scripts Activos (En Uso)

### Scripts Principales
- `scripts/run_opro_local_8gb_fixed.py` ✅ **USAR ESTE** - Versión corregida con sanitización
- `scripts/opro_optimizer_local.py` ✅ - Clases base para OPRO local
- `scripts/evaluate_prompt.py` ✅ - Evaluador básico (en uso)
- `scripts/evaluate_prompt_constrained.py` ✅ - Evaluador mejorado (preparado)
- `scripts/run_opro_local.py` ✅ - Para servidor con >12GB VRAM
- `scripts/refit_psychometric_opro.py` ✅ - Para reajustar curvas post-OPRO
- `scripts/evaluate_opro_test.py` ✅ - Para evaluación final en test set

### Documentación Activa
- `docs/OPRO_LOCAL_ES.md` ✅ - Guía en español para ejecución local
- `docs/OPRO_MEJORAS_IMPLEMENTADAS.md` ✅ - Tracking de mejoras
- `docs/SETUP_RTX4070.md` ✅ - Configuración específica para tu laptop
- `docs/sprints/SPRINT9_OPRO_SPECIFICATION.md` ✅ - Especificación técnica completa

---

## Scripts Archivados (Obsoletos)

Movidos a `scripts/archive_obsolete/`:

1. **`opro_optimizer.py`** (20 KB)
   - Razón: Versión basada en APIs (anthropic/openai)
   - Reemplazado por: `opro_optimizer_local.py`

2. **`run_opro.py`** (5.3 KB)
   - Razón: Runner para versión API
   - Reemplazado por: `run_opro_local_8gb_fixed.py`

3. **`run_opro_local_8gb.py`** (12 KB)
   - Razón: Versión con bug (no sanitizaba tokens especiales)
   - Reemplazado por: `run_opro_local_8gb_fixed.py`

### Documentación Archivada

Movida a `docs/archive_obsolete/`:

1. **`OPRO_SETUP.md`** (5.3 KB)
   - Razón: Setup para APIs (anthropic/openai)

2. **`OPRO_QUICKSTART.md`** (7.9 KB)
   - Razón: Quickstart para versión API

---

## Comando de Ejecución

```bash
cd "c:\VS code projects\OPRO Qwen"

# Ejecución corta (prototipado)
python scripts/run_opro_local_8gb_fixed.py \
    --optimizer_llm "Qwen/Qwen2.5-3B-Instruct" \
    --n_iterations 5 \
    --early_stopping 3 \
    --output_dir results/sprint9_opro_laptop_fixed

# Ejecución completa (cuando estés listo)
python scripts/run_opro_local_8gb_fixed.py \
    --optimizer_llm "Qwen/Qwen2.5-3B-Instruct" \
    --n_iterations 30 \
    --early_stopping 5 \
    --output_dir results/sprint9_opro_laptop_full
```

---

## Estado del Proyecto

### ✅ Implementado
- Ejecución 100% local con transformers
- Optimización para 8GB VRAM (alternancia de modelos)
- Sanitización de tokens especiales
- Validación de candidatos (longitud + keywords)
- Circuit breaker (manejo de errores por candidato)
- Memoria top-k con early stopping
- Reward function multi-objetivo

### 🔄 Preparado (No Integrado Aún)
- `evaluate_prompt_constrained.py` con constrained decoding
- Métrica `ba_hard` para condiciones difíciles
- Successive halving (comentado en código)

### 📋 Pendiente
- Deduplicación de candidatos
- Constrained decoding en OPRO loop
- McNemar test + bootstrap CI
- Ejecución completa en servidor remoto
- Evaluación ONE-TIME en test set
- Tag v2.0-opro-baseline

---

## Estructura de Archivos Final

```
OPRO Qwen/
├── scripts/
│   ├── run_opro_local_8gb_fixed.py      ← USAR ESTE
│   ├── opro_optimizer_local.py          ← Clases base
│   ├── evaluate_prompt.py               ← Evaluador actual
│   ├── evaluate_prompt_constrained.py   ← Evaluador mejorado
│   ├── run_opro_local.py                ← Para servidor
│   ├── refit_psychometric_opro.py       ← Post-optimización
│   ├── evaluate_opro_test.py            ← Test final
│   └── archive_obsolete/
│       ├── opro_optimizer.py            (API version)
│       ├── run_opro.py                  (API version)
│       └── run_opro_local_8gb.py        (buggy version)
│
├── docs/
│   ├── OPRO_LOCAL_ES.md                 ← Guía principal
│   ├── OPRO_MEJORAS_IMPLEMENTADAS.md    ← Tracking
│   ├── SETUP_RTX4070.md                 ← Hardware específico
│   ├── sprints/
│   │   └── SPRINT9_OPRO_SPECIFICATION.md
│   └── archive_obsolete/
│       ├── OPRO_SETUP.md                (API setup)
│       └── OPRO_QUICKSTART.md           (API quickstart)
│
└── requirements.txt                      ← Actualizado con OPRO deps
```

---

## Próximos Pasos

1. **Ejecutar desde terminal** (5 iteraciones, ~30-40 min)
2. **Revisar resultados** en `results/sprint9_opro_laptop_fixed/`
3. **Si BA_clip mejora >0.002**: Transferir a servidor para 30-50 iteraciones
4. **Si BA_clip mejora modesta**: Implementar successive halving + deduplicación
5. **Después de optimización**: McNemar test + bootstrap CI
6. **Final**: Refit psychometric + evaluación test + tag v2.0

---

**Notas**:
- Baseline actual: BA_clip = 0.891 (muy bueno)
- Mejora esperada (5 iter): +0.002 a +0.006
- RTX 4070 Laptop: 8GB VRAM → usar versión `_8gb_fixed`
- Tiempo estimado por iteración: ~6-8 minutos
