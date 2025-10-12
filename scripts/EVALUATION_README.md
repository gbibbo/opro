# Model Evaluation Script

Script completo para evaluar el rendimiento de Qwen2-Audio en condiciones psicoac├║sticas controladas.

## Características

- **Balance 50/50**: Garantiza igual número de muestras con y sin speech
- **Validación automática**: Verifica disponibilidad de muestras antes de ejecutar
- **Métricas completas**:
  - Precisión por duración (20, 50, 100, 200, 500, 1000 ms)
  - Precisión por nivel de SNR (-10, -5, 0, 5, 10, 20 dB)
  - Precisión por filtro de banda (none, telephony, lp3400, hp300)
  - Precisión por reverberación (bins de T60)
  - Promedio global de todas las condiciones
- **Prompts personalizables**: Prueba diferentes estrategias de prompting
- **Reproducibilidad**: Semilla aleatoria configurable

## Requisitos Previos

Antes de ejecutar la evaluación, debes generar las condiciones psicoac├║sticas:

```bash
python scripts/build_conditions.py \
    --input_manifest data/processed/qsm_dev_padded.jsonl \
    --output_dir data/processed/conditions/ \
    --snr_levels -10,-5,0,5,10,20 \
    --band_filters none,telephony,lp3400,hp300 \
    --rir_root data/external/RIRS_NOISES \
    --rir_t60_bins 0.0-0.4,0.4-0.8,0.8-1.5 \
    --n_workers 4
```

## Uso Básico

### Evaluación con prompt por defecto

```bash
python scripts/evaluate_model.py --n_samples 100
```

Esto evaluará 100 muestras (50 con speech, 50 sin speech) usando el prompt por defecto del modelo.

### Evaluación con user prompt personalizado

```bash
python scripts/evaluate_model.py \
    --n_samples 200 \
    --user_prompt "Does this audio contain human speech? Answer YES or NO." \
    --use_prompt
```

### Evaluación con ambos prompts personalizados

```bash
python scripts/evaluate_model.py \
    --n_samples 200 \
    --system_prompt "You are an expert audio classifier." \
    --user_prompt "Does this audio contain human speech? Answer YES or NO." \
    --use_prompt
```

### Evaluación completa con todas las opciones

```bash
python scripts/evaluate_model.py \
    --n_samples 500 \
    --system_prompt "You classify audio content." \
    --user_prompt "Classify this audio segment. Is there speech present?" \
    --use_prompt \
    --output_dir results/evaluation_$(date +%Y%m%d_%H%M%S) \
    --device cuda \
    --load_in_4bit \
    --seed 42
```

## Parámetros

### Obligatorios

- `--n_samples`: Número total de muestras a evaluar (debe ser par para mantener balance 50/50)

### Opcionales

- `--conditions_manifest`: Ruta al manifest de condiciones (default: `data/processed/conditions/conditions_manifest.parquet`)
- `--system_prompt`: System prompt personalizado para el modelo
- `--user_prompt`: User prompt personalizado para el modelo
- `--use_prompt`: Flag para activar los prompts personalizados (si no se especifica, usa los prompts por defecto)
- `--output_dir`: Directorio para guardar resultados (default: `results/`)
- `--device`: Dispositivo para ejecutar el modelo (default: `cuda`)
- `--load_in_4bit`: Usar cuantización de 4 bits (default: True)
- `--seed`: Semilla aleatoria para reproducibilidad (default: 42)

## Salida

El script genera tres archivos en el directorio de salida:

1. **evaluation_results.parquet**: Resultados detallados de cada predicción
   - Incluye: clip_id, ground_truth, predicted, correct, latency_ms, etc.

2. **evaluation_metrics.json**: M├®tricas agregadas en formato JSON
   - Precisión general
   - Precisión por duración
   - Precisión por SNR
   - Precisión por filtro de banda
   - Precisión por reverberación
   - Promedio global
   - Estadísticas de latencia

3. **evaluation_config.json**: Configuración usada para la evaluación
   - Parámetros del script
   - Timestamp
   - Tiempo de evaluación

## Ejemplo de Salida

```
================================================================================
EVALUATION RESULTS
================================================================================

Overall Performance
  Total samples: 100
  Correct: 85
  Accuracy: 85.00%

Global Average (across all conditions)
  Average accuracy: 83.45%

Accuracy by Duration
    20 ms:  65.00% (n=10)
    50 ms:  72.50% (n=10)
   100 ms:  80.00% (n=10)
   200 ms:  85.00% (n=10)
   500 ms:  90.00% (n=10)
  1000 ms:  95.00% (n=10)

Accuracy by SNR Level
   -10 dB:  60.00% (n=15)
    -5 dB:  70.00% (n=15)
     0 dB:  80.00% (n=15)
    +5 dB:  90.00% (n=15)
   +10 dB:  95.00% (n=15)
   +20 dB:  98.00% (n=15)

Accuracy by Band Filter
  none        :  90.00% (n=25)
  telephony   :  85.00% (n=25)
  lp3400      :  88.00% (n=25)
  hp300       :  87.00% (n=25)

Accuracy by Reverberation (T60)
  T60_0.0-0.4    :  92.00% (n=10)
  T60_0.4-0.8    :  88.00% (n=10)
  T60_0.8-1.5    :  84.00% (n=10)

Latency Statistics
  Mean: 1234.5 ms
  Median: 1220.0 ms
  Std: 45.2 ms
  Range: [1150.0, 1350.0] ms

================================================================================
```

## Validación de Errores

### Error: No hay suficientes muestras

```
ERROR: Not enough speech samples available. Requested: 500, Available: 300
```

**Solución**: Reduce el número de muestras solicitadas o genera más condiciones con `build_conditions.py`

### Error: Manifest no encontrado

```
ERROR: Conditions manifest not found at data/processed/conditions/conditions_manifest.parquet

Please run: python scripts/build_conditions.py
```

**Solución**: Ejecuta primero `build_conditions.py` para generar las condiciones psicoac├║sticas

### Error: n_samples impar

```
ERROR: n_samples must be even for 50/50 balance. Got: 101
```

**Solución**: Usa un número par de muestras (ej: 100, 200, 500)

## Estructura del Dataset de Condiciones

Cada clip original de 1000ms genera **20 variantes independientes**:

1. **8 duraciones** (segmentos extraídos del centro del 1000ms, cada uno paddeado a 2000ms):
   - 20, 40, 60, 80, 100, 200, 500, 1000 ms
   - ✅ Cada segmento se extrae del CENTRO del audio de 1000ms original
   - ✅ Luego se paddea a 2000ms con ruido de baja amplitud

2. **6 niveles SNR** (ruido blanco aplicado al 1000ms completo paddeado a 2000ms):
   - -10, -5, 0, +5, +10, +20 dB
   - ✅ SNR se aplica al 1000ms completo, NO a los segmentos de duración
   - ✅ El ruido se mezcla con el audio completo de 2000ms (1000ms + padding)

3. **3 filtros de banda** (aplicados al 1000ms completo paddeado a 2000ms):
   - `telephony`: Bandpass 300-3400 Hz
   - `hp300`: Highpass 300 Hz
   - `lp3400`: Lowpass 3400 Hz
   - ✅ Filtros se aplican al 1000ms completo, NO a los segmentos de duración

4. **3 bins de reverberación** (RIR aplicado al 1000ms completo paddeado a 2000ms):
   - T60 0.0-0.4 s (salas secas)
   - T60 0.4-0.8 s (salas medianas)
   - T60 0.8-1.5 s (salas reverberantes)
   - ✅ RIR se aplica al 1000ms completo, NO a los segmentos de duración

**Total**: 8 + 6 + 3 + 3 = **20 variantes por clip** (suma, NO multiplicación)

### Justificación del Diseño

Este diseño permite evaluar INDEPENDIENTEMENTE cada tipo de degradación:

- **Duraciones**: ¿Cuál es el umbral temporal mínimo para clasificar correctamente?
- **SNR**: ¿Cómo afecta el ruido a la clasificación del audio completo (1000ms)?
- **Filtros de banda**: ¿Cómo afecta la limitación de frecuencias al audio completo?
- **Reverberación**: ¿Cómo afecta el tiempo de reverberación al audio completo?

Si combináramos todas las condiciones (ej: 20ms + SNR=-10 + telephony + reverb), tendríamos:
- 8 × 6 × 3 × 3 = **432 variantes por clip** (inmanejable)
- No podríamos aislar el efecto de cada degradación

Con el diseño actual tenemos:
- **20 variantes por clip** (manejable)
- Efectos INDEPENDIENTES y medibles
- 87 clips × 20 variantes = **1,740 variantes totales** en el dataset de evaluación

## Casos de Uso

### 1. Validación rápida (2 clips = 40 variantes)

```bash
python scripts/evaluate_model.py --n_clips 2
```

### 2. Comparación de prompts (2 clips cada uno)

```bash
# Prompt A
python scripts/evaluate_model.py \
    --n_clips 2 \
    --user_prompt "Does this contain speech?" \
    --use_prompt \
    --output_dir results/prompt_a \
    --seed 42

# Prompt B
python scripts/evaluate_model.py \
    --n_clips 2 \
    --user_prompt "Is there human voice in this audio?" \
    --use_prompt \
    --output_dir results/prompt_b \
    --seed 42

# Prompt C - con system prompt diferente
python scripts/evaluate_model.py \
    --n_clips 2 \
    --system_prompt "You are a highly accurate audio classifier." \
    --user_prompt "Analyze this audio. Does it contain speech?" \
    --use_prompt \
    --output_dir results/prompt_c \
    --seed 42
```

### 3. Evaluación completa para benchmarking (10 clips = 200 variantes)

```bash
python scripts/evaluate_model.py \
    --n_clips 10 \
    --output_dir results/benchmark_baseline \
    --seed 42
```

### 4. Testing de fine-tuning

```bash
# Antes del fine-tuning
python scripts/evaluate_model.py \
    --n_samples 500 \
    --output_dir results/pre_finetuning

# Despu├®s del fine-tuning
python scripts/evaluate_model.py \
    --n_samples 500 \
    --output_dir results/post_finetuning
```

## Análisis de Resultados

Para analizar los resultados generados:

```python
import pandas as pd
import json

# Cargar resultados detallados
results = pd.read_parquet("results/evaluation_results.parquet")

# Cargar métricas
with open("results/evaluation_metrics.json") as f:
    metrics = json.load(f)

# Analizar errores
errors = results[results["correct"] == False]
print(f"Total errors: {len(errors)}")
print(errors["duration_ms"].value_counts())

# Comparar condiciones
print(results.groupby("variant_type")["correct"].mean())
```

## Notas

- El script utiliza cuantización de 4 bits por defecto para optimizar memoria
- Los resultados son reproducibles usando la misma semilla
- La evaluación puede tardar varios minutos dependiendo del número de muestras y hardware disponible
- El promedio global es el promedio de los promedios de cada tipo de condición (duración, SNR, filtros, reverb)
