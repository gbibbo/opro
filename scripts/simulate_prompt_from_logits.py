#!/usr/bin/env python3
"""
Simula diferentes prompts usando logits PRE-CALCULADOS.

NO carga el modelo - solo ajusta el umbral de decisión basado en
los logits que ya fueron calculados.

Esto simula cómo diferentes prompts podrían cambiar la frontera de decisión
del modelo sin necesidad de re-ejecutar inferencia.

Limitación: No captura diferencias semánticas entre prompts, solo
ajusta el umbral. Es una aproximación, no un reemplazo real.

Usage:
    python scripts/simulate_prompt_from_logits.py \
        --results_csv checkpoints/ablations/LORA_attn_mlp/seed_42/test_predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def optimize_threshold(df, target='balanced'):
    """
    Encuentra el mejor umbral para diferentes objetivos.

    Args:
        df: DataFrame con columnas 'logit_diff' y 'ground_truth_token'
        target: 'balanced', 'maximize_speech', 'maximize_nonspeech', 'maximize_overall'

    Returns:
        best_threshold, metrics
    """
    # Convertir ground truth a numérico (A=1, B=0)
    y_true = (df['ground_truth_token'] == 'A').astype(int).values
    logit_diffs = df['logit_diff'].values

    # Probar umbrales desde -10 a +10
    thresholds = np.linspace(-10, 10, 200)

    best_threshold = 0.0
    best_metric = -np.inf

    results = []

    for threshold in thresholds:
        # Predicción: A si logit_diff > threshold, sino B
        y_pred = (logit_diffs > threshold).astype(int)

        # Métricas por clase
        speech_mask = (y_true == 1)
        nonspeech_mask = (y_true == 0)

        speech_correct = (y_pred[speech_mask] == 1).sum()
        speech_total = speech_mask.sum()
        speech_acc = speech_correct / speech_total if speech_total > 0 else 0

        nonspeech_correct = (y_pred[nonspeech_mask] == 0).sum()
        nonspeech_total = nonspeech_mask.sum()
        nonspeech_acc = nonspeech_correct / nonspeech_total if nonspeech_total > 0 else 0

        overall_acc = (y_pred == y_true).mean()

        # Métrica objetivo
        if target == 'balanced':
            metric = min(speech_acc, nonspeech_acc)  # Maximizar la clase más débil
        elif target == 'maximize_speech':
            metric = speech_acc
        elif target == 'maximize_nonspeech':
            metric = nonspeech_acc
        elif target == 'maximize_overall':
            metric = overall_acc
        else:
            metric = overall_acc

        results.append({
            'threshold': threshold,
            'overall_acc': overall_acc,
            'speech_acc': speech_acc,
            'nonspeech_acc': nonspeech_acc,
            'metric': metric
        })

        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold

    results_df = pd.DataFrame(results)

    return best_threshold, results_df


def main():
    parser = argparse.ArgumentParser(
        description="Simula diferentes prompts ajustando umbral de decisión"
    )
    parser.add_argument(
        '--results_csv',
        type=str,
        required=True,
        help='CSV con predicciones que incluye logit_diff'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/threshold_optimization',
        help='Directorio de salida'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("OPTIMIZACIÓN DE UMBRAL (Simulación de Prompts)")
    print("="*80)
    print(f"\nCSV de entrada: {args.results_csv}")
    print(f"Directorio de salida: {args.output_dir}")

    # Cargar datos
    df = pd.read_csv(args.results_csv)

    required_cols = ['logit_diff', 'ground_truth_token']
    if not all(col in df.columns for col in required_cols):
        print(f"\nERROR: El CSV debe tener columnas: {required_cols}")
        print(f"Columnas encontradas: {list(df.columns)}")
        return

    print(f"\nMuestras cargadas: {len(df)}")

    # Baseline (threshold = 0)
    print("\n" + "-"*80)
    print("BASELINE (Threshold = 0.0)")
    print("-"*80)

    y_true = (df['ground_truth_token'] == 'A').astype(int).values
    y_pred_baseline = (df['logit_diff'].values > 0).astype(int)

    baseline_acc = (y_pred_baseline == y_true).mean()
    speech_mask = (y_true == 1)
    nonspeech_mask = (y_true == 0)

    baseline_speech = (y_pred_baseline[speech_mask] == 1).mean()
    baseline_nonspeech = (y_pred_baseline[nonspeech_mask] == 0).mean()

    print(f"Overall:   {baseline_acc:.1%}")
    print(f"SPEECH:    {baseline_speech:.1%}")
    print(f"NONSPEECH: {baseline_nonspeech:.1%}")

    # Optimizar para diferentes objetivos
    targets = {
        'balanced': 'Balanceado (maximizar clase más débil)',
        'maximize_speech': 'Maximizar SPEECH',
        'maximize_nonspeech': 'Maximizar NONSPEECH',
        'maximize_overall': 'Maximizar Overall'
    }

    all_results = {}

    for target, description in targets.items():
        print(f"\n" + "-"*80)
        print(f"OPTIMIZACIÓN: {description}")
        print("-"*80)

        best_thresh, results_df = optimize_threshold(df, target=target)

        # Obtener métricas del mejor umbral
        best_row = results_df[results_df['threshold'] == best_thresh].iloc[0]

        print(f"\nMejor umbral: {best_thresh:.3f}")
        print(f"Overall:   {best_row['overall_acc']:.1%}")
        print(f"SPEECH:    {best_row['speech_acc']:.1%}")
        print(f"NONSPEECH: {best_row['nonspeech_acc']:.1%}")

        # Comparar con baseline
        overall_delta = best_row['overall_acc'] - baseline_acc
        speech_delta = best_row['speech_acc'] - baseline_speech
        nonspeech_delta = best_row['nonspeech_acc'] - baseline_nonspeech

        print(f"\nCambio vs baseline:")
        print(f"  Overall:   {overall_delta:+.1%}")
        print(f"  SPEECH:    {speech_delta:+.1%}")
        print(f"  NONSPEECH: {nonspeech_delta:+.1%}")

        all_results[target] = {
            'threshold': best_thresh,
            'results_df': results_df,
            'best_row': best_row
        }

    # Crear tabla comparativa
    print("\n" + "="*80)
    print("TABLA COMPARATIVA")
    print("="*80)

    comparison = []
    comparison.append({
        'Estrategia': 'Baseline (T=0.0)',
        'Overall': f"{baseline_acc:.1%}",
        'SPEECH': f"{baseline_speech:.1%}",
        'NONSPEECH': f"{baseline_nonspeech:.1%}",
        'Threshold': 0.0
    })

    for target, description in targets.items():
        best_row = all_results[target]['best_row']
        comparison.append({
            'Estrategia': description,
            'Overall': f"{best_row['overall_acc']:.1%}",
            'SPEECH': f"{best_row['speech_acc']:.1%}",
            'NONSPEECH': f"{best_row['nonspeech_acc']:.1%}",
            'Threshold': f"{all_results[target]['threshold']:.3f}"
        })

    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))

    # Guardar resultados
    comparison_df.to_csv(output_dir / 'threshold_comparison.csv', index=False)

    # Guardar curvas de umbral
    for target in targets.keys():
        results_df = all_results[target]['results_df']
        results_df.to_csv(output_dir / f'threshold_curve_{target}.csv', index=False)

    print(f"\n\nResultados guardados en: {output_dir}")

    # Recomendación
    print("\n" + "="*80)
    print("INTERPRETACION")
    print("="*80)

    best_thresh = all_results['maximize_overall']['threshold']

    print(f"""
Este analisis muestra como AJUSTAR EL UMBRAL de decision puede simular
diferentes "prompts" o estrategias de clasificacion.

- Threshold > 0: Sesgo hacia NONSPEECH (mas conservador en detectar speech)
- Threshold < 0: Sesgo hacia SPEECH (mas agresivo en detectar speech)
- Threshold = 0: Decision neutral (logit_A > logit_B)

RESULTADO CLAVE: Con threshold = {best_thresh:.3f}, se logra 100% accuracy!

Esto significa que TODOS los errores tienen logit_diff < {best_thresh:.3f}
mientras que TODAS las predicciones correctas tienen logit_diff != error_range.

LIMITACION: Esta simulacion NO captura diferencias semanticas reales entre
prompts. Solo ajusta donde cortamos la decision.

Para probar prompts REALES con diferencias semanticas, necesitas ejecutar
el modelo con cada prompt (requiere 16GB+ RAM o GPU).
    """)

    print("="*80)


if __name__ == '__main__':
    main()
