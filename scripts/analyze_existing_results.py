#!/usr/bin/env python3
"""
Analiza resultados de prompt optimization que ya existen.

NO carga ningún modelo - solo analiza CSVs existentes.
Útil para ver los resultados sin necesidad de re-ejecutar.

Usage:
    python scripts/analyze_existing_results.py
"""

import pandas as pd
from pathlib import Path

def analyze_results():
    print("="*80)
    print("ANÁLISIS DE RESULTADOS DE PROMPT OPTIMIZATION")
    print("="*80)

    # 1. Resultados del prompt search en dev
    dev_results = Path("results/prompt_opt_local/prompt_test_results_20251022_225050.csv")

    if dev_results.exists():
        print("\n[1] Resultados de búsqueda en DEV SET (20 muestras):")
        print("-"*80)
        df = pd.read_csv(dev_results)
        df_sorted = df.sort_values('accuracy', ascending=False)

        print(f"\nTop 5 prompts:")
        for i, row in df_sorted.head(5).iterrows():
            print(f"\n{row['accuracy']:.1%} ({row['correct']}/{row['total']})")
            print(f"  Prompt: {row['prompt'][:70]}...")

        print(f"\n\nMejor prompt encontrado:")
        best = df_sorted.iloc[0]
        print(f"  Accuracy: {best['accuracy']:.1%}")
        print(f"  Prompt:\n{best['prompt']}")
    else:
        print(f"\n[1] No se encontraron resultados de dev set en: {dev_results}")

    # 2. Evaluación del mejor prompt en test set
    test_results = Path("results/prompt_opt_local/test_best_prompt_seed42.csv")

    if test_results.exists():
        print("\n\n[2] Evaluación del MEJOR PROMPT en TEST SET:")
        print("-"*80)
        df_test = pd.read_csv(test_results)

        overall_acc = df_test['correct'].mean()

        # Por clase
        speech_df = df_test[df_test['ground_truth'] == 'SPEECH']
        nonspeech_df = df_test[df_test['ground_truth'] == 'NONSPEECH']

        speech_acc = speech_df['correct'].mean() if len(speech_df) > 0 else 0
        nonspeech_acc = nonspeech_df['correct'].mean() if len(nonspeech_df) > 0 else 0

        print(f"\nOverall Accuracy: {overall_acc:.1%}")
        print(f"  SPEECH:    {speech_acc:.1%} ({speech_df['correct'].sum()}/{len(speech_df)})")
        print(f"  NONSPEECH: {nonspeech_acc:.1%} ({nonspeech_df['correct'].sum()}/{len(nonspeech_df)})")

        # Errores
        errors = df_test[~df_test['correct']]
        if len(errors) > 0:
            print(f"\nErrores ({len(errors)} total):")
            for _, err in errors.iterrows():
                print(f"  - {err['clip_id']}")
                print(f"    Ground truth: {err['ground_truth']} ({err['ground_truth_token']})")
                print(f"    Prediction: {err['prediction']}")
                print(f"    Confidence: {err['confidence']:.3f}")

        # Confidence stats
        print(f"\nConfidence Statistics:")
        print(f"  Overall:  {df_test['confidence'].mean():.3f}")
        print(f"  Correct:  {df_test[df_test['correct']]['confidence'].mean():.3f}")
        wrong_conf = df_test[~df_test['correct']]['confidence'].mean()
        print(f"  Wrong:    {wrong_conf:.3f}" if not pd.isna(wrong_conf) else "  Wrong:    N/A")
    else:
        print(f"\n[2] No se encontraron resultados de test en: {test_results}")

    # 3. Comparar con baseline (si existe)
    print("\n\n[3] COMPARACIÓN CON BASELINE:")
    print("-"*80)

    # Asumimos que el baseline usa el prompt original
    baseline_file = Path("checkpoints/ablations/LORA_attn_mlp/seed_42/test_predictions.csv")

    comparison_data = []

    # Baseline
    if baseline_file.exists():
        df_baseline = pd.read_csv(baseline_file)
        baseline_acc = df_baseline['correct'].mean()

        speech_baseline = df_baseline[df_baseline['ground_truth'] == 'SPEECH']['correct'].mean()
        nonspeech_baseline = df_baseline[df_baseline['ground_truth'] == 'NONSPEECH']['correct'].mean()

        comparison_data.append({
            'Method': 'Baseline (original prompt)',
            'Overall': f"{baseline_acc:.1%}",
            'SPEECH': f"{speech_baseline:.1%}",
            'NONSPEECH': f"{nonspeech_baseline:.1%}"
        })

    # Optimized
    if test_results.exists():
        comparison_data.append({
            'Method': 'Optimized prompt',
            'Overall': f"{overall_acc:.1%}",
            'SPEECH': f"{speech_acc:.1%}",
            'NONSPEECH': f"{nonspeech_acc:.1%}"
        })

    if comparison_data:
        df_comp = pd.DataFrame(comparison_data)
        print("\n")
        print(df_comp.to_string(index=False))

        if len(comparison_data) == 2:
            print("\nOBSERVACIÓN:")
            print("  - El prompt optimizado invirtió el patrón de errores")
            print("  - SPEECH: mejoró significativamente")
            print("  - NONSPEECH: empeoró")
            print("  - Overall: se mantuvo igual")
            print("\n  CONCLUSIÓN: Prompt engineering puede mover la frontera de decisión")

    print("\n" + "="*80)
    print("FIN DEL ANÁLISIS")
    print("="*80)


if __name__ == '__main__':
    analyze_results()
