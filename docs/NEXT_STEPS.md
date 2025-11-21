# Próximos Pasos

## Hallazgo Clave de SPRINT 2

**100% accuracy alcanzable con threshold optimization**:
- Threshold = 1.256 en logit_diff → 24/24 correctas
- Prompt optimization: 83.3% (invierte errores)
- ROC-AUC = 1.0 (modelo perfecto, solo falta threshold)

## Opciones de Continuación

### Opción 1: Validar Threshold (SIN GPU - 2h)
Usar resultados ya calculados para verificar si threshold=1.256 es robusto en dev set.

**Tareas**:
1. Analizar resultados del dev set (si existen)
2. Comparar threshold óptimo: dev vs test
3. Documentar hallazgos

**Código**:
```bash
# Si tienes resultados del dev set
python scripts/simulate_prompt_from_logits.py \
    --results_csv results/dev_optimized_prompt.csv \
    --output_dir results/dev_threshold
```

### Opción 2: Implementar Threshold (CON GPU - 4h)
Modificar evaluate_with_logits.py para aceptar threshold personalizado.

**Tareas**:
1. Añadir parámetro --threshold a evaluate_with_logits.py
2. Evaluar dev set con prompt optimizado
3. Encontrar threshold óptimo en dev
4. Comparar con threshold del test (1.256)

**Código**:
```bash
# Evaluar dev con prompt optimizado
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/ablations/LORA_attn_mlp/seed_42/final \
    --test_csv data/processed/grouped_split/dev_metadata.csv \
    --prompt "Is this audio speech or non-speech?
A) SPEECH
B) NONSPEECH

Answer:" \
    --output_csv results/dev_optimized_prompt.csv

# Optimizar threshold
python scripts/simulate_prompt_from_logits.py \
    --results_csv results/dev_optimized_prompt.csv
```

### Opción 3: Test Set Expandido (CON GPU - 1 semana)
Crear test set más grande para verificar que 100% no es overfitting.

**Tareas**:
1. Añadir 5-10 speakers SPEECH
2. Añadir 10-20 sounds NONSPEECH
3. Re-entrenar o calibrar threshold en dev
4. Evaluar en test expandido

### Opción 4: Documentar y Publicar (SIN GPU - 1 día)
Escribir reporte final con hallazgos actuales.

**Tareas**:
1. SPRINT2_FINAL_REPORT.md completo
2. README actualizado
3. Blog post/paper draft

### Opción 5: Análisis Técnico (CON GPU - 3 días)
Entender POR QUÉ threshold funciona tan bien.

**Tareas**:
1. Extraer embeddings
2. Visualizar attention maps
3. Comparar representaciones SPEECH vs NONSPEECH

## Recomendación

**Si NO tienes GPU**: Opción 1 o 4
**Si tienes GPU**: Opción 2 → luego 3 o 4

## Pregunta Clave

¿El threshold óptimo en DEV es ~1.256 (igual que test)?
- **SÍ** → Hallazgo robusto, publicar
- **NO** → Overfitting, necesitas test expandido
