# SPRINT 1: Calibration + Stats - Execution Plan

**Goal**: Train with proper train/dev/test split + Temperature calibration + Publication-ready stats

**Expected outcome**:
- Model trained on 64 samples (6 groups)
- Temperature calibrated on 72 dev samples (4 groups)
- Evaluated on 24 test samples (3 groups, held-out)
- ECE < 0.05 (well-calibrated)
- Possibly 90-100% accuracy (vs current 83.3%)

---

## Prerequisites

✅ Dev split created: `data/processed/grouped_split/dev_metadata.csv` (72 samples)
✅ Train split updated: `data/processed/grouped_split/train_metadata.csv` (64 samples)
✅ Test split unchanged: `data/processed/grouped_split/test_metadata.csv` (24 samples)

---

## Step 1: Train Model with New Split (3 seeds)

**Duration**: ~20-40 minutes per seed (~1-2 hours total)

```bash
# Seed 42
python scripts/finetune_qwen_audio.py \
    --seed 42 \
    --output_dir checkpoints/calibrated_v1/seed_42 \
    --add_mlp_targets

# Seed 123
python scripts/finetune_qwen_audio.py \
    --seed 123 \
    --output_dir checkpoints/calibrated_v1/seed_123 \
    --add_mlp_targets

# Seed 456
python scripts/finetune_qwen_audio.py \
    --seed 456 \
    --output_dir checkpoints/calibrated_v1/seed_456 \
    --add_mlp_targets
```

**Note**: You can run these in parallel in different terminals. Each uses ~12GB VRAM.

---

## Step 2: Get Uncalibrated Predictions on Dev Set

**Duration**: ~5-7 minutes per seed

After each training completes, get predictions on dev with T=1.0 (uncalibrated):

```bash
# Seed 42
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/calibrated_v1/seed_42/final \
    --test_csv data/processed/grouped_split/dev_metadata.csv \
    --temperature 1.0 \
    --output_csv results/calibration/seed_42_dev_uncalibrated.csv

# Seed 123
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/calibrated_v1/seed_123/final \
    --test_csv data/processed/grouped_split/dev_metadata.csv \
    --temperature 1.0 \
    --output_csv results/calibration/seed_123_dev_uncalibrated.csv

# Seed 456
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/calibrated_v1/seed_456/final \
    --test_csv data/processed/grouped_split/dev_metadata.csv \
    --temperature 1.0 \
    --output_csv results/calibration/seed_456_dev_uncalibrated.csv
```

---

## Step 3: Calibrate Temperature (Find Optimal T)

**Duration**: ~30 seconds per seed

```bash
# Create results directory
mkdir -p results/calibration

# Seed 42
python scripts/calibrate_temperature.py \
    --predictions_csv results/calibration/seed_42_dev_uncalibrated.csv \
    --output_temp results/calibration/seed_42_optimal_temp.txt \
    --plot results/calibration/seed_42_calibration_analysis.png

# Seed 123
python scripts/calibrate_temperature.py \
    --predictions_csv results/calibration/seed_123_dev_uncalibrated.csv \
    --output_temp results/calibration/seed_123_optimal_temp.txt \
    --plot results/calibration/seed_123_calibration_analysis.png

# Seed 456
python scripts/calibrate_temperature.py \
    --predictions_csv results/calibration/seed_456_dev_uncalibrated.csv \
    --output_temp results/calibration/seed_456_optimal_temp.txt \
    --plot results/calibration/seed_456_calibration_analysis.png
```

**Output**: Each will print the optimal temperature (e.g., T=0.523) and save it to a text file.

---

## Step 4: Extract Optimal Temperatures

```bash
# View optimal temperatures
cat results/calibration/seed_42_optimal_temp.txt
cat results/calibration/seed_123_optimal_temp.txt
cat results/calibration/seed_456_optimal_temp.txt
```

**Expected**: Something like:
```
0.523000
0.541000
0.518000
```

---

## Step 5: Evaluate on Test Set with Calibrated Temperature

**Duration**: ~5-7 minutes per seed

Replace `<T_optimal>` with the value from Step 4:

```bash
# Seed 42 (example: T=0.523)
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/calibrated_v1/seed_42/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --temperature 0.523 \
    --output_csv results/calibration/seed_42_test_calibrated.csv

# Seed 123 (example: T=0.541)
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/calibrated_v1/seed_123/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --temperature 0.541 \
    --output_csv results/calibration/seed_123_test_calibrated.csv

# Seed 456 (example: T=0.518)
python scripts/evaluate_with_logits.py \
    --checkpoint checkpoints/calibrated_v1/seed_456/final \
    --test_csv data/processed/grouped_split/test_metadata.csv \
    --temperature 0.518 \
    --output_csv results/calibration/seed_456_test_calibrated.csv
```

---

## Step 6: Aggregate Multi-Seed Results (Calibrated)

**Duration**: ~30 seconds

```bash
python scripts/aggregate_multi_seed.py \
    --model_name "calibrated_v1" \
    --seeds 42 123 456 \
    --predictions_dir results/calibration \
    --test_pattern "seed_{seed}_test_calibrated.csv" \
    --output_file results/calibration/multi_seed_calibrated_summary.txt
```

**Note**: This may require updating the aggregate script to accept custom patterns. Let me know if it fails.

---

## Expected Results

### Uncalibrated (T=1.0, current)
```
Overall:   83.3% ± 0.0%
SPEECH:    50.0% ± 0.0%
NONSPEECH: 100.0% ± 0.0%
ECE:       ~0.15-0.20 (poorly calibrated)
```

### Calibrated (T~0.5)
```
Overall:   90-100%  (potentially perfect given ROC-AUC=1.0)
SPEECH:    75-100%  (huge improvement expected)
NONSPEECH: 100%     (already perfect)
ECE:       <0.05    (well-calibrated)
```

---

## Files Generated

After completion, you should have:

### Checkpoints
- `checkpoints/calibrated_v1/seed_42/final/`
- `checkpoints/calibrated_v1/seed_123/final/`
- `checkpoints/calibrated_v1/seed_456/final/`

### Optimal Temperatures
- `results/calibration/seed_42_optimal_temp.txt`
- `results/calibration/seed_123_optimal_temp.txt`
- `results/calibration/seed_456_optimal_temp.txt`

### Calibration Plots
- `results/calibration/seed_42_calibration_analysis.png`
- `results/calibration/seed_123_calibration_analysis.png`
- `results/calibration/seed_456_calibration_analysis.png`

### Predictions
- `results/calibration/seed_*_dev_uncalibrated.csv` (3 files)
- `results/calibration/seed_*_test_calibrated.csv` (3 files)

### Summary
- `results/calibration/multi_seed_calibrated_summary.txt`

---

## Troubleshooting

### Issue 1: Out of memory during training
**Solution**: Train seeds sequentially instead of parallel, or reduce batch size in finetune script

### Issue 2: calibrate_temperature.py missing columns
**Solution**: Check that evaluate_with_logits.py outputs `logit_A`, `logit_B`, `ground_truth_token` columns

### Issue 3: aggregate_multi_seed.py doesn't find files
**Solution**: Update the script to accept custom filename patterns (I can do this if needed)

---

## Next Steps (After Sprint 1)

Once you have calibrated results:

1. Compare calibrated vs uncalibrated (should see huge ECE improvement)
2. If accuracy doesn't improve much, investigate:
   - Are errors still on same speaker (voxconverse_abjxc)?
   - Is threshold still the issue?
   - Need more training data?
3. Move to SPRINT 2 (OPRO post-FT, Qwen3 baseline)

---

## Estimated Total Time

- **Step 1** (Training 3 seeds): 1-2 hours
- **Step 2** (Dev predictions): 15-20 minutes
- **Step 3** (Calibration): 2 minutes
- **Step 4-6** (Test evaluation + aggregation): 20-25 minutes

**Total**: ~2-3 hours (mostly training)

---

**Ready to start?** Begin with Step 1 (training). You can run the 3 seeds in parallel if you have enough VRAM, or sequentially if not.

Let me know when training completes and I'll help with any issues in the subsequent steps.
