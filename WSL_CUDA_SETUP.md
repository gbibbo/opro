# WSL CUDA Setup for Qwen2-Audio

## El Problema

Estás viendo este error:
```
RuntimeError: Found no NVIDIA driver on your system
```

Esto significa que PyTorch no puede acceder a la GPU NVIDIA desde WSL.

## Soluciones

### Opción 1: Configurar CUDA en WSL (Recomendado)

#### 1. Verificar que tienes WSL 2
```bash
wsl --version
# Debe mostrar WSL versión 2.x
```

#### 2. Instalar NVIDIA CUDA Toolkit para WSL
En Windows (PowerShell como administrador):
```powershell
# Descargar e instalar NVIDIA CUDA Toolkit para WSL
# https://developer.nvidia.com/cuda-downloads
# Seleccionar: Linux > x86_64 > WSL-Ubuntu > 2.0 > deb (network)
```

O en WSL directamente:
```bash
# Añadir repositorio NVIDIA
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

#### 3. Reinstalar PyTorch con soporte CUDA
```bash
# En tu entorno conda
conda activate opro

# Desinstalar PyTorch actual
pip uninstall torch torchaudio

# Reinstalar con CUDA 12.1 (compatible con WSL)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Verificar instalación
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Deberías ver:
```
CUDA available: True
Device: NVIDIA GeForce RTX ...
```

### Opción 2: Ejecutar en Windows Nativo (Más Simple)

Si tienes problemas con WSL, ejecuta directamente en Windows:

#### 1. Abrir PowerShell o CMD nativo de Windows

#### 2. Activar conda
```powershell
conda activate opro
```

#### 3. Ejecutar inferencia
```powershell
cd "C:\VS projects\OPRO Qwen"

python scripts/run_qwen_inference.py `
    --segments-dir data/segments/ava_speech/train `
    --limit 2 `
    --device cuda `
    --dtype float16
```

### Opción 3: Usar CPU (Muy Lento - Solo para Testing)

**NO recomendado para dataset completo** (tomaría días), pero útil para verificar que el código funciona:

```bash
python scripts/run_qwen_inference.py \
    --segments-dir data/segments/ava_speech/train \
    --limit 1 \
    --device cpu
```

⚠️ **Advertencia**: CPU es ~100x más lento. Un segmento puede tomar 5-10 minutos.

## Diagnóstico

### 1. Verificar drivers NVIDIA
En Windows PowerShell:
```powershell
nvidia-smi
```

Debe mostrar tu GPU y drivers.

### 2. Verificar CUDA en WSL
```bash
nvidia-smi  # Debe funcionar en WSL también
```

### 3. Verificar PyTorch
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Debe mostrar versión CUDA (ej: "12.1"), no "None".

### 4. Verificar device mapping
```bash
python -c "import torch; print(torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

## Errores Comunes

### Error: `CUDA out of memory`
**Solución**: Usa `--dtype float16` para reducir uso de memoria a la mitad.

### Error: `module 'torch' has no attribute 'cuda'`
**Solución**: Reinstala PyTorch con soporte CUDA (ver Opción 1, paso 3).

### Error: `No CUDA GPUs are available`
**Solución**:
1. Verifica `nvidia-smi` funciona
2. Reinstala drivers NVIDIA
3. Reinicia WSL: `wsl --shutdown` y vuelve a abrir

## Recomendación Final

**Para máxima facilidad y rendimiento:**

1. **Ejecuta en Windows nativo** (Opción 2) - no requiere configuración adicional
2. Si prefieres WSL, sigue Opción 1 para configurar CUDA correctamente

Una vez que funcione, el tiempo de inferencia será:
- **Por segmento**: 3-10 segundos
- **Dataset completo (1,016 segmentos)**: 1.5-3 horas

## Próximos Pasos Después de Arreglar CUDA

```bash
# 1. Test rápido (2 segmentos)
python scripts/run_qwen_inference.py \
    --segments-dir data/segments/ava_speech/train \
    --limit 2 \
    --device cuda \
    --dtype float16

# 2. Si funciona, ejecutar en todos los datasets
python scripts/run_qwen_inference.py --segments-dir data/segments/ava_speech/train --device cuda --dtype float16
python scripts/run_qwen_inference.py --segments-dir data/segments/voxconverse/dev --device cuda --dtype float16
python scripts/run_qwen_inference.py --segments-dir data/segments/esc50/nonspeech --device cuda --dtype float16
```

## Referencias

- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [PyTorch Installation](https://pytorch.org/get-started/locally/)
- [WSL 2 CUDA Support](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
