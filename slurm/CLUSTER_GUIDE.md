# OPRO Classic - Guía de Ejecución en Cluster

Esta guía explica cómo ejecutar OPRO Classic en el cluster y monitorear el progreso en tiempo real.

---

## 📋 Pre-requisitos

1. **Acceso al cluster** vía SSH
2. **Conda environment** configurado con dependencias (`qwen_audio`)
3. **Datos preparados** en `data/processed/conditions_final/`
4. **Checkpoint LoRA** (opcional, solo si vas a usar modo `lora`)

---

## 🚀 Paso 1: Descargar el código

### Desde el cluster:

```bash
# SSH al cluster
ssh your_username@cluster.address

# Navegar a tu directorio de trabajo
cd /path/to/your/workspace

# Clonar el repo (si no lo tienes)
git clone https://github.com/gbibbo/opro.git
cd opro

# O actualizar si ya lo tienes
git pull origin main
```

---

## ⚙️ Paso 2: Verificar la configuración

### 2.1. Verificar datos

```bash
# Verificar que el manifest existe
ls -lh data/processed/conditions_final/conditions_manifest_split.parquet

# Ver cuántas muestras hay en dev/test
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/conditions_final/conditions_manifest_split.parquet')
print(f\"Dev samples: {len(df[df['split'] == 'dev'])}\")
print(f\"Test samples: {len(df[df['split'] == 'test'])}\")
"
```

### 2.2. Verificar conda environment

```bash
# Activar environment
conda activate qwen_audio

# Verificar paquetes críticos
python -c "
import torch
import transformers
import pandas as pd
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers: {transformers.__version__}')
"
```

### 2.3. Verificar checkpoint (solo para modo LoRA)

```bash
# Si vas a usar LoRA, verifica que el checkpoint existe
ls -lh checkpoints/qwen_lora_seed42/final/
```

---

## 🎬 Paso 3: Lanzar el job

### Opción A: Modelo base (sin fine-tuning)

```bash
sbatch slurm/opro_classic.job base 42
```

### Opción B: Modelo fine-tuned (con LoRA)

```bash
sbatch slurm/opro_classic.job lora 42
```

### Opción C: Múltiples seeds en paralelo

```bash
# Lanzar 3 runs con diferentes seeds
for seed in 42 123 456; do
    sbatch slurm/opro_classic.job base $seed
done
```

**Output esperado:**
```
Submitted batch job 1234567
```

Guarda el Job ID (e.g., `1234567`) para monitorear el job.

---

## 📊 Paso 4: Monitorear el job en tiempo real

### 4.1. Verificar estado del job

```bash
# Ver estado de tus jobs
squeue -u $USER

# Ver estado de un job específico
squeue -j 1234567
```

**Estados posibles:**
- `PD` (Pending): Esperando recursos
- `R` (Running): En ejecución
- `CG` (Completing): Finalizando
- `CD` (Completed): Completado

### 4.2. Ver output en tiempo real

```bash
# Ver output en tiempo real (auto-refresh cada 2 segundos)
watch -n 2 tail -50 logs/opro_classic_1234567.out

# O con tail -f (actualización continua)
tail -f logs/opro_classic_1234567.out
```

**Salir de `watch`:** Presiona `Ctrl+C`
**Salir de `tail -f`:** Presiona `Ctrl+C`

### 4.3. Ver errores en tiempo real

```bash
# Ver errores en tiempo real
tail -f logs/opro_classic_1234567.err

# O ambos en paralelo (en terminal dividida)
# Terminal 1:
tail -f logs/opro_classic_1234567.out

# Terminal 2:
tail -f logs/opro_classic_1234567.err
```

### 4.4. Ver progreso de OPRO

Durante la ejecución, verás:

```
============================================
Iteration 1: Generating 3 candidates...
============================================

Generating...
  Parsing 3 raw candidates...
    Candidate 1: Does this audio contain speech? ...
      ✓ Valid
    Candidate 2: Is there human voice in this audio? ...
      ✓ Valid
    Candidate 3: Audio classification task: SPEECH or NON-SPEECH...
      ✓ Valid

Evaluating candidate 1/3...
Prompt: Does this audio contain speech? ...
  Evaluating: 100%|██████████| 1200/1200 [02:15<00:00,  8.85it/s]
  ✓ Results: BA_clip=0.923, BA_cond=0.910, Reward=0.9152

Evaluating candidate 2/3...
...

NEW BEST REWARD: 0.9234 (+0.0082)
```

### 4.5. Ver métricas actuales

```bash
# Ver el mejor prompt hasta ahora
cat results/opro_classic_base_seed42/best_prompt.txt

# Ver las métricas del mejor prompt
cat results/opro_classic_base_seed42/best_metrics.json | python -m json.tool

# Ver el progreso (reward por iteración)
cat results/opro_classic_base_seed42/opro_history.json | python -m json.tool
```

### 4.6. Monitoreo avanzado con script helper

Crea un script de monitoreo:

```bash
cat > monitor_opro.sh <<'EOF'
#!/bin/bash
JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: ./monitor_opro.sh <job_id>"
    exit 1
fi

clear
echo "============================================"
echo "OPRO CLASSIC - Monitor (Job $JOB_ID)"
echo "============================================"
echo ""

# Job status
echo "📊 Job Status:"
squeue -j $JOB_ID -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
echo ""

# Last 10 lines of output
echo "📄 Last 10 lines (output):"
tail -10 logs/opro_classic_${JOB_ID}.out
echo ""

# Check for errors
if [ -s logs/opro_classic_${JOB_ID}.err ]; then
    echo "⚠️  Errors detected:"
    tail -5 logs/opro_classic_${JOB_ID}.err
else
    echo "✓ No errors"
fi
echo ""

# Current best metrics
OUTPUT_DIR=$(grep "Output:" logs/opro_classic_${JOB_ID}.out | tail -1 | awk '{print $2}')
if [ -n "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/best_metrics.json" ]; then
    echo "🏆 Current Best Metrics:"
    python3 -c "
import json
with open('$OUTPUT_DIR/best_metrics.json') as f:
    m = json.load(f)
    print(f\"  Reward: {m['reward']:.4f}\")
    print(f\"  BA_clip: {m['ba_clip']:.3f}\")
    print(f\"  BA_conditions: {m['ba_conditions']:.3f}\")
    if 'metrics' in m:
        print(f\"  BA_duration: {m['metrics'].get('ba_duration', 0):.3f}\")
        print(f\"  BA_SNR: {m['metrics'].get('ba_snr', 0):.3f}\")
        print(f\"  BA_filter: {m['metrics'].get('ba_filter', 0):.3f}\")
        print(f\"  BA_reverb: {m['metrics'].get('ba_reverb', 0):.3f}\")
"
fi

echo ""
echo "============================================"
echo "Press Ctrl+C to exit"
echo "Refreshing in 10 seconds..."
sleep 10
exec $0 $JOB_ID
EOF

chmod +x monitor_opro.sh
```

Luego úsalo:

```bash
./monitor_opro.sh 1234567
```

---

## 🛑 Paso 5: Cancelar un job (si es necesario)

```bash
# Cancelar un job específico
scancel 1234567

# Cancelar todos tus jobs
scancel -u $USER

# Cancelar solo jobs de OPRO
scancel -u $USER -n opro_classic
```

---

## 📁 Paso 6: Revisar resultados finales

### 6.1. Verificar que el job completó exitosamente

```bash
# Ver las últimas líneas del log
tail -20 logs/opro_classic_1234567.out
```

Buscar:
```
✓ SUCCESS
Results saved to: results/opro_classic_base_seed42
```

### 6.2. Inspeccionar archivos de salida

```bash
OUTPUT_DIR="results/opro_classic_base_seed42"

# Listar archivos generados
ls -lh $OUTPUT_DIR/

# Ver mejor prompt
cat $OUTPUT_DIR/best_prompt.txt

# Ver métricas completas
cat $OUTPUT_DIR/best_metrics.json | python -m json.tool

# Ver historial completo (todos los prompts evaluados)
head -20 $OUTPUT_DIR/opro_prompts.jsonl

# Ver progresión del reward
cat $OUTPUT_DIR/opro_history.json | python -m json.tool
```

### 6.3. Analizar métricas psicoacústicas

```bash
# Extraer métricas por dimensión
python -c "
import json
with open('$OUTPUT_DIR/best_metrics.json') as f:
    m = json.load(f)['metrics']

print('=== Psychoacoustic Metrics ===')
print(f\"BA_duration: {m['ba_duration']:.3f}\")
print(f\"BA_SNR: {m['ba_snr']:.3f}\")
print(f\"BA_filter: {m['ba_filter']:.3f}\")
print(f\"BA_reverb: {m['ba_reverb']:.3f}\")
print()

print('=== Duration Breakdown ===')
for dur, metrics in sorted(m['duration_metrics'].items(), key=lambda x: float(x[0])):
    print(f\"{dur}ms: BA={metrics['ba']:.3f}\")
print()

print('=== SNR Breakdown ===')
for snr, metrics in sorted(m['snr_metrics'].items(), key=lambda x: float(x[0])):
    print(f\"{snr}dB: BA={metrics['ba']:.3f}\")
"
```

### 6.4. Descargar resultados a tu máquina local

```bash
# Desde tu máquina local (no desde el cluster)
scp -r your_username@cluster.address:/path/to/opro/results/opro_classic_base_seed42 ./local_results/
```

---

## 🐛 Troubleshooting

### Problema: Job no arranca (estado PD por mucho tiempo)

**Causa:** No hay GPUs disponibles o cola saturada.

**Solución:**
```bash
# Ver información de la cola
sinfo -p gpu

# Ver jobs en la cola
squeue -p gpu

# Ajustar prioridad o cambiar partición en opro_classic.job
```

### Problema: Job falla inmediatamente

**Solución:**
```bash
# Ver errores
cat logs/opro_classic_1234567.err

# Causas comunes:
# - Manifest no encontrado → Verificar ruta en slurm/opro_classic.job
# - Conda env no activado → Verificar module load y conda activate
# - GPU no disponible → Verificar con nvidia-smi en el nodo
```

### Problema: Out of Memory (OOM)

**Solución:**
```bash
# Editar slurm/opro_classic.job y aumentar memoria:
#SBATCH --mem=64G

# O usar modelo más pequeño:
OPTIMIZER_LLM="Qwen/Qwen2.5-3B-Instruct"
```

### Problema: Timeout después de 48h

**Solución:**
```bash
# Editar slurm/opro_classic.job y aumentar tiempo:
#SBATCH --time=96:00:00

# O reducir iteraciones:
NUM_ITERATIONS=15
```

---

## 📊 Ejemplo completo: Workflow típico

```bash
# 1. SSH al cluster
ssh user@cluster

# 2. Actualizar código
cd /path/to/opro
git pull origin main

# 3. Activar environment
conda activate qwen_audio

# 4. Verificar datos
ls -lh data/processed/conditions_final/conditions_manifest_split.parquet

# 5. Lanzar job
sbatch slurm/opro_classic.job base 42
# Output: Submitted batch job 1234567

# 6. Monitorear en tiempo real
tail -f logs/opro_classic_1234567.out

# 7. Verificar progreso (en otra terminal)
cat results/opro_classic_base_seed42/best_prompt.txt

# 8. Cuando termine, revisar resultados
cat results/opro_classic_base_seed42/best_metrics.json | python -m json.tool

# 9. Descargar a local (desde tu máquina)
scp -r user@cluster:/path/to/opro/results/opro_classic_base_seed42 ./
```

---

## 📚 Archivos clave

```
opro/
├── slurm/
│   ├── opro_classic.job           # Job SLURM principal
│   └── CLUSTER_GUIDE.md           # Esta guía
├── scripts/
│   └── opro_classic_optimize.py   # Script Python de OPRO
├── logs/                          # Logs de SLURM (auto-generados)
│   ├── opro_classic_1234567.out
│   └── opro_classic_1234567.err
└── results/
    └── opro_classic_base_seed42/  # Resultados (auto-generados)
        ├── best_prompt.txt
        ├── best_metrics.json
        ├── opro_prompts.jsonl
        ├── opro_history.json
        └── job_info.txt
```

---

## 🎯 Comandos útiles (cheat sheet)

```bash
# Lanzar job
sbatch slurm/opro_classic.job base 42

# Ver mis jobs
squeue -u $USER

# Ver output en tiempo real
tail -f logs/opro_classic_JOBID.out

# Cancelar job
scancel JOBID

# Ver estado de GPUs
squeue -p gpu

# Ver mejor prompt actual
cat results/opro_classic_base_seed42/best_prompt.txt

# Ver métricas actuales
cat results/opro_classic_base_seed42/best_metrics.json | python -m json.tool

# Descargar resultados
scp -r user@cluster:/path/to/opro/results/opro_classic_base_seed42 ./
```

---

## 💡 Tips

1. **Usa `screen` o `tmux`** para mantener sesiones SSH activas:
   ```bash
   screen -S opro_monitor
   tail -f logs/opro_classic_1234567.out
   # Detach: Ctrl+A, D
   # Re-attach: screen -r opro_monitor
   ```

2. **Notificaciones por email** (edita `opro_classic.job`):
   ```bash
   #SBATCH --mail-type=END,FAIL
   #SBATCH --mail-user=your.email@domain.com
   ```

3. **Checkpoint automático**: OPRO Classic guarda estado después de cada iteración, por lo que puedes cancelar y reanudar si es necesario.

4. **Múltiples runs**: Usa un loop para lanzar múltiples seeds en paralelo:
   ```bash
   for seed in 42 123 456 789 2024; do
       sbatch slurm/opro_classic.job base $seed
   done
   ```

---

¿Preguntas? Revisa el README principal o abre un issue en GitHub.
