#!/bin/bash
set -e  # Exit on error

echo "=== OPRO Legacy: Usando datos existentes en cluster ==="

##### 0) VOLVER A LA RUTA CORRECTA + DATOS QUE YA EXISTEN
cd /mnt/fast/nobackup/users/gb0048/opro

# Sanidad mínima: CSVs + WAVs reales de experimental_variants
test -f data/processed/experimental_variants/dev_metadata.csv || { echo "[ERR] Falta dev_metadata.csv"; exit 1; }
echo "[OK] dev_metadata.csv encontrado"

find data/processed/experimental_variants/audio -type f -name '*.wav' | head -n 3
echo "[OK] Archivos WAV encontrados"

# Truco de compatibilidad por si algún CSV trae 'processed/...'
ln -sfn "$(pwd)/data/processed" processed
echo "[OK] Symlink processed creado"

##### 1) AJUSTES LIMPIOS DE ENTORNO (HF cache soportado)
export HF_HOME=/mnt/fast/nobackup/users/gb0048/.cache/huggingface
export TRANSFORMERS_VERBOSITY=info
echo "[OK] Variables de entorno configuradas"

##### 2) TEMPLATES: A/B, MC y Open (si no existen en prompts/)
mkdir -p prompts
echo "[OK] Directorio prompts/ creado"

cat > prompts/ab.json << 'JSON'
{
  "format": "ab",
  "templates": [
    {"id":"ab_01","text":"Binary decision. Output exactly one token: SPEECH or NONSPEECH.","mapping":{"A":"SPEECH","B":"NONSPEECH"},"verbalizers":["SPEECH","NONSPEECH"]},
    {"id":"ab_07","text":"Task: Speech detection. Select one:\nA) Speech\nB) Non-speech","mapping":{"A":"SPEECH","B":"NONSPEECH"},"verbalizers":["SPEECH","NONSPEECH"]}
  ]
}
JSON
echo "[OK] Template A/B creado"

cat > prompts/mc.json << 'JSON'
{
  "format": "mc",
  "templates": [
    {"id":"mc_05","text":"What do you hear?\nA) Person talking/speaking\nB) Song or melody\nC) Background noise\nD) Beeps/clicks/other","mapping":{"A":"SPEECH","B":"NONSPEECH","C":"NONSPEECH","D":"NONSPEECH"},"verbalizers":["SPEECH","NONSPEECH"]}
  ]
}
JSON
echo "[OK] Template MC creado"

cat > prompts/open.json << 'JSON'
{
  "format": "open",
  "templates": [
    {"id":"open_01","text":"Describe the main content of the audio in <=3 words, then decide: SPEECH or NONSPEECH. Output only SPEECH or NONSPEECH at the end.","mapping":{},"verbalizers":["SPEECH","NONSPEECH"]}
  ]
}
JSON
echo "[OK] Template Open creado"

##### 3) CONFIGURAR APPTAINER/SINGULARITY
REPO="/mnt/fast/nobackup/users/gb0048/opro"
CONTAINER="$REPO/qwen_pipeline_v2.sif"

if [ ! -f "$CONTAINER" ]; then
    echo "[ERR] Container not found: $CONTAINER"
    exit 1
fi
echo "[OK] Container found: $CONTAINER"

# Detectar si usar apptainer o singularity
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
    echo "[OK] Using apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
    echo "[OK] Using singularity"
else
    echo "[ERR] Neither apptainer nor singularity found in PATH"
    echo "[INFO] Try: module load singularity"
    exit 1
fi

# Función para ejecutar Python dentro del contenedor
run_python() {
    $CONTAINER_CMD exec --nv \
        --pwd "$REPO" \
        --env HF_HOME="$HF_HOME" \
        --env TRANSFORMERS_VERBOSITY=info \
        "$CONTAINER" python3 "$@"
}

##### 4) EJECUTAR OPRO EN DEV (datos existentes) PARA CADA FORMATO
echo ""
echo "=== Ejecutando OPRO A/B ==="
run_python scripts/opro_post_ft_v2.py \
  --no_lora \
  --train_csv data/processed/experimental_variants/dev_metadata.csv \
  --templates_file prompts/ab.json \
  --decoding ab \
  --output_dir results/opro_ab \
  --num_iterations 10 \
  --samples_per_iter 100 \
  --num_candidates 12

echo ""
echo "=== Ejecutando OPRO MC ==="
run_python scripts/opro_post_ft_v2.py \
  --no_lora \
  --train_csv data/processed/experimental_variants/dev_metadata.csv \
  --templates_file prompts/mc.json \
  --decoding mc \
  --output_dir results/opro_mc \
  --num_iterations 10 \
  --samples_per_iter 100 \
  --num_candidates 12

echo ""
echo "=== Ejecutando OPRO Open ==="
run_python scripts/opro_post_ft_v2.py \
  --no_lora \
  --train_csv data/processed/experimental_variants/dev_metadata.csv \
  --templates_file prompts/open.json \
  --decoding open \
  --output_dir results/opro_open \
  --num_iterations 10 \
  --samples_per_iter 100 \
  --num_candidates 12

##### 5) VER RESULTADOS RESUMIDOS
echo ""
echo "=== RESULTADOS ==="
echo ""
echo "=== MEJOR PROMPT A/B ==="
cat results/opro_ab/best_prompt.txt 2>/dev/null || echo "sin .txt; mira best_prompt.json"

echo ""
echo "=== MEJOR PROMPT MC ==="
cat results/opro_mc/best_prompt.txt 2>/dev/null || echo "sin .txt; mira best_prompt.json"

echo ""
echo "=== MEJOR PROMPT OPEN ==="
cat results/opro_open/best_prompt.txt 2>/dev/null || echo "sin .txt; mira best_prompt.json"

echo ""
echo "=== Archivos de predicciones ==="
ls -lh results/opro_*/iter*_all_predictions.csv 2>/dev/null || echo "No se encontraron CSVs de predicciones"

echo ""
echo "=== OPRO Legacy completado exitosamente ==="
