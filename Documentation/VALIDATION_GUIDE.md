# Validation Guide: Zero-Leakage Split and Ablation Studies

**Purpose**: Complete guide for validating models with zero data leakage

**Last Updated**: 2025-10-21

---

## Quick Start: Validate Current Results

### 1. Quick Comparison
```bash
python scripts/quick_compare.py \
    results/no_leakage_v2_predictions.csv \
    results/ablations/LORA_attn_mlp_seed42.csv \
    "Attention-only" \
    "Attention+MLP"
```

### 2. Statistical Comparison
```bash
python scripts/compare_models_statistical.py \
    --predictions_A results/no_leakage_v2_predictions.csv \
    --predictions_B results/ablations/LORA_attn_mlp_seed42.csv \
    --model_A_name "Attention-only" \
    --model_B_name "Attention+MLP" \
    --n_bootstrap 10000
```

---

## Complete Workflow

### Step 1: Create Zero-Leakage Split

```bash
python scripts/create_group_stratified_split.py \
    --input_csv data/processed/clean_clips/clean_metadata.csv \
    --output_dir data/processed/grouped_split \
    --test_size 0.2 \
    --random_state 42
```

### Step 2: Audit for Leakage

```bash
python scripts/audit_split_leakage.py \
    --train_csv data/processed/grouped_split/train_metadata.csv \
    --test_csv data/processed/grouped_split/test_metadata.csv
```

**Expected**: "Overlap: 0" (zero leakage)

### Step 3: Train Model

```bash
# Attention-only (baseline)
python scripts/finetune_qwen_audio.py \
    --seed 42 \
    --output_dir checkpoints/no_leakage_v2/seed_42

# Attention+MLP (comparison)
python scripts/finetune_qwen_audio.py \
    --seed 42 \
    --output_dir checkpoints/ablations/LORA_attn_mlp/seed_42 \
    --add_mlp_targets
```

### Step 4: Evaluate

```bash
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/no_leakage_v2/seed_42/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --temperature 1.0 \
    --output_csv results/no_leakage_v2_predictions.csv
```

### Step 5: Compare Models

Use commands from Quick Start above.

---

## Multi-Seed Validation

### Train Additional Seeds

```bash
for seed in 123 456; do
    python scripts/finetune_qwen_audio.py \
        --seed $seed \
        --output_dir checkpoints/ablations/LORA_attn_mlp/seed_$seed \
        --add_mlp_targets

    python scripts/evaluate_with_logits.py \
        --checkpoint checkpoints/ablations/LORA_attn_mlp/seed_$seed/final \
        --test_csv data/processed/grouped_split/test_metadata.csv \
        --temperature 1.0 \
        --output_csv results/ablations/LORA_attn_mlp_seed$seed.csv
done
```

### Aggregate Results

```bash
python -c "
import pandas as pd
import glob

config = 'LORA_attn_mlp'
files = sorted(glob.glob(f'results/ablations/{config}_seed*.csv'))
results = []

for f in files:
    df = pd.read_csv(f)
    seed = f.split('seed')[-1].split('.')[0]
    total = len(df)
    correct = df['correct'].sum()

    speech_df = df[df['ground_truth'] == 'SPEECH']
    speech_correct = speech_df['correct'].sum()

    nonspeech_df = df[df['ground_truth'] == 'NONSPEECH']
    nonspeech_correct = nonspeech_df['correct'].sum()

    results.append({
        'seed': seed,
        'overall': 100 * correct / total,
        'speech': 100 * speech_correct / len(speech_df),
        'nonspeech': 100 * nonspeech_correct / len(nonspeech_df)
    })

results_df = pd.DataFrame(results)
print('Multi-Seed Results:')
print(results_df.to_string(index=False))
print()
print(f\"Overall: {results_df['overall'].mean():.1f}% ± {results_df['overall'].std():.1f}%\")
print(f\"SPEECH: {results_df['speech'].mean():.1f}% ± {results_df['speech'].std():.1f}%\")
print(f\"NONSPEECH: {results_df['nonspeech'].mean():.1f}% ± {results_df['nonspeech'].std():.1f}%\")
"
```

---

## Temperature Calibration (Optional)

### Create Dev Split
```bash
python scripts/create_group_stratified_split.py \
    --input_csv data/processed/grouped_split/train_metadata.csv \
    --output_dir data/processed/dev_split \
    --test_size 0.2 \
    --random_state 42
```

### Find Optimal Temperature
```bash
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/no_leakage_v2/seed_42/final \
    --test_csv data/processed/dev_split/test_metadata.csv \
    --temperature 1.0 \
    --output_csv results/dev_predictions.csv

python scripts/calibrate_temperature.py \
    --predictions_csv results/dev_predictions.csv \
    --output_temp results/optimal_temperature.txt \
    --n_bins 5
```

### Evaluate with Calibrated Temperature
```bash
T=$(cat results/optimal_temperature.txt)
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/no_leakage_v2/seed_42/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --temperature $T \
    --output_csv results/test_calibrated.csv
```

---

## Troubleshooting

### GPU Memory Issues
If evaluation fails with memory error:
```bash
# The script now includes llm_int8_enable_fp32_cpu_offload=True
# If still failing, reduce batch size or use CPU offloading
```

### Leakage Detected
If audit finds overlap:
```bash
# Re-run split creation with different random_state
python scripts/create_group_stratified_split.py \
    --input_csv data/processed/clean_clips/clean_metadata.csv \
    --output_dir data/processed/grouped_split_v2 \
    --test_size 0.2 \
    --random_state 123  # Try different seed
```

### Predictions Mismatch
If compare fails with "predictions don't match":
```bash
# Verify both CSVs have same samples
head -1 results/no_leakage_v2_predictions.csv
head -1 results/ablations/LORA_attn_mlp_seed42.csv
```

---

## Current Results (As of 2025-10-21)

**Attention-only (seed 42)**:
- Overall: 79.2% (19/24)
- SPEECH: 37.5% (3/8)
- NONSPEECH: 100% (16/16)

**Attention+MLP (seed 42)**:
- Overall: 83.3% (20/24)
- SPEECH: 50.0% (4/8) - **+12.5% improvement**
- NONSPEECH: 100% (16/16)

**Statistical Test**: McNemar p=1.0000 (not significant with n=24)

**Interpretation**: MLP shows promising improvement but needs multi-seed validation to confirm.

---

## References

- [LEAKAGE_FIX_REPORT.md](LEAKAGE_FIX_REPORT.md) - Complete timeline of discovery
- [FINAL_RESULTS_ZERO_LEAKAGE.md](FINAL_RESULTS_ZERO_LEAKAGE.md) - Detailed analysis
- [EXPERIMENT_MATRIX_SMALL_DATA.md](EXPERIMENT_MATRIX_SMALL_DATA.md) - Ablation study design
- [ABLATION_EXECUTION_PLAN.md](ABLATION_EXECUTION_PLAN.md) - Step-by-step execution guide
