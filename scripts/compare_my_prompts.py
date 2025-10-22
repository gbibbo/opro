#!/usr/bin/env python3
"""
Script simple para comparar tus propios prompts personalizados.

Uso:
    python scripts/compare_my_prompts.py
"""

import subprocess
from pathlib import Path

# Define tus prompts aquí
MY_PROMPTS = [
    "Is this audio SPEECH (A) or NON-SPEECH (B)? Answer with a single letter:",

    "Is this audio speech or non-speech?\nA) SPEECH\nB) NONSPEECH\n\nAnswer:",

    "Listen carefully. Does this contain human speech?\nA) Yes\nB) No\n\nAnswer:",

    # Añade tus propios prompts aquí
    "Tu prompt personalizado 1",
    "Tu prompt personalizado 2",
]

# Configuración
CHECKPOINT = "checkpoints/ablations/LORA_attn_mlp/seed_42/final"
TEST_CSV = "data/processed/grouped_split/test_metadata.csv"
OUTPUT_DIR = Path("results/my_prompt_comparison")

def test_prompt(prompt, idx):
    """Evalúa un prompt y retorna la accuracy."""
    output_csv = OUTPUT_DIR / f"prompt_{idx:02d}.csv"

    cmd = [
        "python", "scripts/evaluate_with_logits.py",
        "--checkpoint", CHECKPOINT,
        "--test_csv", TEST_CSV,
        "--prompt", prompt,
        "--output_csv", str(output_csv)
    ]

    print(f"\n{'='*80}")
    print(f"[{idx+1}/{len(MY_PROMPTS)}] Testing prompt:")
    print(f"  {prompt[:70]}..." if len(prompt) > 70 else f"  {prompt}")
    print(f"{'='*80}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Extraer accuracy del output
    for line in result.stdout.split('\n'):
        if 'Overall Accuracy:' in line:
            # Ejemplo: "Overall Accuracy: 20/24 = 83.3%"
            accuracy = line.split('=')[-1].strip().rstrip('%')
            return float(accuracy), line

    return 0.0, "Error"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPARACIÓN DE PROMPTS PERSONALIZADOS")
    print("="*80)
    print(f"\nCheckpoint: {CHECKPOINT}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Número de prompts: {len(MY_PROMPTS)}")

    results = []

    for i, prompt in enumerate(MY_PROMPTS):
        accuracy, details = test_prompt(prompt, i)
        results.append({
            'idx': i,
            'accuracy': accuracy,
            'prompt': prompt,
            'details': details
        })

    # Ordenar por accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)

    print("\n" + "="*80)
    print("RESULTADOS FINALES")
    print("="*80)

    for i, result in enumerate(results):
        print(f"\n[{i+1}] Accuracy: {result['accuracy']:.1f}%")
        print(f"    Prompt: {result['prompt'][:60]}...")
        print(f"    {result['details']}")

    print("\n" + "="*80)
    print(f"MEJOR PROMPT (Accuracy: {results[0]['accuracy']:.1f}%):")
    print("="*80)
    print(results[0]['prompt'])
    print()

    # Guardar mejor prompt
    best_file = OUTPUT_DIR / "best_prompt.txt"
    best_file.write_text(results[0]['prompt'])
    print(f"Mejor prompt guardado en: {best_file}")

if __name__ == '__main__':
    main()
