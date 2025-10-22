# Trabajando con OPRO Qwen en Sistemas de Baja Memoria (8GB RAM)

## Tu Situación

- **RAM disponible**: 7.8GB
- **GPU**: No disponible en WSL (NVML bloqueado)
- **Modelo**: Qwen2-Audio-7B (7 mil millones de parámetros)

**Realidad**: El modelo Qwen2-Audio-7B necesita **16GB+ RAM** incluso con cuantización 4-bit.

---

## Opciones que SÍ Funcionan en Tu Sistema

### ✅ Opción 1: Analizar Resultados Existentes (RECOMENDADO)

**NO carga el modelo** - solo lee CSVs que ya fueron generados.

```bash
python scripts/analyze_existing_results.py
```

**Output que obtienes**:
- Top 5 prompts del dev set
- Evaluación detallada en test set
- Comparación con baseline
- Análisis de errores y confianza

**Ventajas**:
- ✅ Instantáneo (< 1 segundo)
- ✅ 0 MB de RAM
- ✅ Muestra todos los insights importantes

**Ya lo probaste y funcionó!**

---

### ✅ Opción 2: Usar Resultados Pre-calculados para Tu Propio Análisis

Los resultados ya están en:

```bash
# Resultados de prompt search en dev
results/prompt_opt_local/prompt_test_results_20251022_225050.csv

# Mejor prompt encontrado
results/prompt_opt_local/best_prompt.txt

# Evaluación del mejor prompt en test
results/prompt_opt_local/test_best_prompt_seed42.csv

# Baseline (prompt original)
checkpoints/ablations/LORA_attn_mlp/seed_42/test_predictions.csv
```

**Ejemplo de análisis personalizado**:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Leer resultados
df_dev = pd.read_csv('results/prompt_opt_local/prompt_test_results_20251022_225050.csv')
df_test = pd.read_csv('results/prompt_opt_local/test_best_prompt_seed42.csv')
df_baseline = pd.read_csv('checkpoints/ablations/LORA_attn_mlp/seed_42/test_predictions.csv')

# Crear gráfico de comparación
fig, ax = plt.subplots(figsize=(10, 6))

methods = ['Baseline', 'Optimized Prompt']
speech_acc = [
    df_baseline[df_baseline['ground_truth'] == 'SPEECH']['correct'].mean(),
    df_test[df_test['ground_truth'] == 'SPEECH']['correct'].mean()
]
nonspeech_acc = [
    df_baseline[df_baseline['ground_truth'] == 'NONSPEECH']['correct'].mean(),
    df_test[df_test['ground_truth'] == 'NONSPEECH']['correct'].mean()
]

x = range(len(methods))
width = 0.35

ax.bar([i - width/2 for i in x], speech_acc, width, label='SPEECH')
ax.bar([i + width/2 for i in x], nonspeech_acc, width, label='NONSPEECH')

ax.set_ylabel('Accuracy')
ax.set_title('Baseline vs Optimized Prompt')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()

plt.savefig('results/comparison.png')
print("Gráfico guardado en results/comparison.png")
```

---

### ✅ Opción 3: Crear Tu Propio Prompt y Evaluarlo (En Otro Sistema)

Si tienes acceso a una máquina con más RAM o GPU:

**En Google Colab (GRATIS, 12GB+ RAM):**

1. Sube tu checkpoint a Google Drive
2. Monta Drive en Colab
3. Ejecuta:

```python
!pip install -q transformers peft bitsandbytes soundfile

# Tu código de evaluación aquí
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
# ... (resto del código)
```

**En Lambda Labs / RunPod (Pago, ~$0.50/hora):**

```bash
# SSH a la instancia
ssh user@gpu-instance

# Clonar repo
git clone <tu-repo>
cd OPRO\ Qwen

# Ejecutar
python scripts/test_prompt_templates.py \
    --checkpoint checkpoints/ablations/LORA_attn_mlp/seed_42/final \
    --test_csv data/processed/grouped_split/dev_metadata.csv \
    --output_dir results/my_results \
    --num_samples 72
```

---

## Opciones que NO Funcionan en Tu Sistema

### ❌ Cargar el Modelo Localmente

```bash
# Esto SIEMPRE será "Killed" en tu sistema
python scripts/test_prompt_templates.py ...
python scripts/evaluate_with_logits.py ...
python scripts/finetune_qwen_audio.py ...
```

**Por qué**:
- Qwen2-Audio-7B base: ~14GB en FP16
- Con 4-bit quantization: ~3.5GB
- Con LoRA adapters: +500MB
- Overhead de PyTorch/Transformers: +2GB
- **Mínimo necesario: ~6-7GB solo para cargar**
- **Para inferencia: +2-3GB adicionales**
- **Total: ~10GB mínimo**

Tu sistema: 7.8GB total → OOM garantizado

---

## Qué Hacer para Ejecutar el Modelo

### Solución A: Upgrade de Hardware (Local)

**Comprar más RAM**:
- Mínimo: 16GB (funcional pero justo)
- Recomendado: 32GB (cómodo)
- Ideal: 64GB (sin problemas)

**Costo**: $30-150 USD dependiendo de tu sistema

### Solución B: Cloud GPU (Temporal)

**Google Colab**:
- ✅ GRATIS (con límites)
- ✅ 12-15GB RAM
- ✅ GPU T4 opcional
- ❌ Sesiones de 12h máximo
- ❌ Puede desconectarse

**Cómo usar**:
1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Nuevo notebook
3. Runtime → Change runtime type → GPU
4. Sube checkpoint o conecta Drive

**Lambda Labs** ($0.50-2.00/hora):
- ✅ GPU potentes (A100, H100)
- ✅ RAM abundante (80GB+)
- ✅ Setup completo
- ❌ Requiere tarjeta de crédito

**Cómo usar**:
```bash
# Crear instancia en lambda.cloud
# SSH
ssh ubuntu@<instance-ip>

# Clonar y ejecutar
git clone <repo>
cd OPRO\ Qwen
python scripts/test_prompt_templates.py ...
```

### Solución C: WSL con GPU (Si tienes GPU NVIDIA en Windows)

Si tu PC tiene GPU NVIDIA pero está bloqueada en WSL:

```bash
# En PowerShell como administrador (Windows)
wsl --update
wsl --shutdown

# Verificar drivers
nvidia-smi

# Instalar CUDA toolkit en WSL
# Seguir: https://docs.nvidia.com/cuda/wsl-user-guide/
```

Si logras habilitar GPU en WSL:
- RAM requerida baja a ~4-6GB (usa VRAM de GPU)
- Velocidad 10-50x más rápida

---

## Resumen: Qué Puedes Hacer AHORA

### Con tu sistema actual (8GB RAM, sin GPU):

✅ **Analizar resultados existentes**:
```bash
python scripts/analyze_existing_results.py
```

✅ **Explorar CSVs manualmente**:
```bash
head -20 results/prompt_opt_local/test_best_prompt_seed42.csv
```

✅ **Crear visualizaciones** (Matplotlib, Pandas):
```python
import pandas as pd
df = pd.read_csv('results/prompt_opt_local/test_best_prompt_seed42.csv')
print(df.groupby('ground_truth')['correct'].mean())
```

✅ **Diseñar nuevos prompts** (para probar en otro sistema):
```bash
# Editar lista de prompts
nano scripts/test_prompt_templates.py

# Guardar para ejecutar en Colab/cloud
```

### Para ejecutar el modelo:

🔄 **Opción rápida**: Google Colab (gratis, 1 hora setup)

💰 **Opción profesional**: Lambda Labs ($2-5 total para tus experimentos)

🛠️ **Opción permanente**: Upgrade RAM a 32GB ($50-100)

---

## Ejemplo: Workflow Híbrido Recomendado

```bash
# 1. Diseñar experimentos en tu sistema (8GB)
nano scripts/test_prompt_templates.py  # Añadir tus prompts

# 2. Subir a Colab o cloud
scp -r scripts/ user@cloud:/workspace/

# 3. Ejecutar en cloud (16GB+)
ssh user@cloud
cd /workspace
python scripts/test_prompt_templates.py ...

# 4. Descargar resultados
scp user@cloud:/workspace/results/*.csv results/

# 5. Analizar en tu sistema (8GB)
python scripts/analyze_existing_results.py
python custom_analysis.py
```

**Costo**: $0 (Colab) o $1-2 (Lambda, 1-2 horas)

---

## FAQ

**P: ¿Puedo usar CPU en vez de GPU?**
A: Sí, pero el modelo IGUAL necesita 10GB+ RAM para cargar, y será 50x más lento.

**P: ¿Y si cierro todos los programas?**
A: Ayuda, pero solo liberarás ~500MB. Necesitas 10GB, tienes 7.8GB → imposible.

**P: ¿Puedo usar swap?**
A: Sí, pero será EXTREMADAMENTE lento (100-1000x). Un prompt tomaría 30-60 minutos.

**P: ¿Qué tal 4-bit quantization?**
A: Ya está incluida en el script. Aun así necesitas 6-7GB solo para cargar.

**P: ¿Cuál es la solución MÁS BARATA?**
A: Google Colab gratuito (límite 12h/día).

**P: ¿Cuál es la solución MÁS RÁPIDA?**
A: Comprar RAM (si tu motherboard soporta 32GB).

---

## Conclusión

**Tu hardware actual (8GB RAM) es suficiente para**:
- ✅ Análisis de datos
- ✅ Visualizaciones
- ✅ Diseño de experimentos
- ✅ Código/debugging
- ✅ Git/documentación

**NO es suficiente para**:
- ❌ Cargar modelos de 7B parámetros
- ❌ Fine-tuning
- ❌ Inferencia del modelo

**Recomendación**: Usa Google Colab (gratis) para ejecutar el modelo, y tu sistema para todo lo demás.

---

## Siguiente Paso Recomendado

Ejecuta esto en tu sistema AHORA:

```bash
python scripts/analyze_existing_results.py
```

Te mostrará todos los resultados de prompt optimization sin necesitar más RAM.

Luego, si quieres probar tus propios prompts:

1. Edítalos en `scripts/test_prompt_templates.py`
2. Sube el repo a Google Colab
3. Ejecuta allí
4. Descarga resultados
5. Analiza en tu sistema

**Tiempo total**: ~30 minutos
**Costo**: $0
