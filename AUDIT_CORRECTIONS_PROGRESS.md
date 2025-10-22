# Audit Corrections Progress Report

**Date**: 2025-10-22
**Status**: Corrections in Progress - Following User Audit Feedback

---

## User Audit Feedback Summary

The user provided a comprehensive audit with three main sections:
1. **Consistency audit**: Data integrity, evaluation methods, statistical rigor
2. **README/reporting improvements**: Structure, provenance, confidence intervals
3. **Pre-scaling action plan**: Tasks to complete before scaling data

---

## ✅ COMPLETED CORRECTIONS

### 1. Evaluation Method - Robust A/B Token Handling

**Issue**: Token probability calculation only used single token ID for "A" and "B", but Qwen tokenizer can produce variants with leading space (" A", " B") depending on context.

**Fix Applied**: Updated `scripts/evaluate_with_logits.py`

**Changes**:
- Modified `get_ab_token_ids()` to collect BOTH variants:
  - "A" → ID 32, " A" → ID 362
  - "B" → ID 33, " B" → ID 425
- Changed aggregation method from `max()` to `logsumexp()` for summing probabilities
- This properly sums probability mass across tokenization variants

**Result**: Re-evaluated LORA_attn_mlp_seed42 → **same accuracy** (83.3%), confirming model was already mostly using non-space tokens, but now method is more robust.

**Code snippet**:
```python
# OLD: Only single token
ids_A = tokenizer.encode('A', add_special_tokens=False)  # [32]

# NEW: Both variants
ids_A = []
ids_A.extend(tokenizer.encode('A', add_special_tokens=False))  # [32]
ids_A_space = tokenizer.encode(' A', add_special_tokens=False)  # [362]
for id_val in ids_A_space:
    if id_val not in ids_A:
        ids_A.append(id_val)
# Result: ids_A = [32, 362]

# Aggregation: logsumexp instead of max
logit_A = torch.logsumexp(logits_A, dim=0).item()
```

---

### 2. Calibration - Proper Dev Split Created

**Issue**: Previous calibration was done on test set (data leakage for hyperparameter tuning).

**Fix Applied**: Created `scripts/create_dev_split.py`

**Implementation**:
- Splits original train (136 samples) into:
  - **New Train**: 80 samples, 8 groups (56 NONSPEECH, 24 SPEECH)
  - **Dev**: 56 samples, 2 groups (48 SPEECH, 8 NONSPEECH)
- Uses **GroupShuffleSplit** to prevent leakage
- **Zero overlap** verified
- **Test set** (24 samples) remains completely held-out

**Workflow**:
1. Train on new train split (80 samples)
2. Calibrate temperature on dev (56 samples)
3. Evaluate final model on test (24 samples) with calibrated temperature

**Location**: [data/processed/grouped_split_with_dev/](data/processed/grouped_split_with_dev/)

**Note**: This requires re-training models. Current models (trained on 136 samples) can still be used for other analyses, but proper calibrated results will need new training run.

---

## 🚧 IN PROGRESS

### 3. Comprehensive Progress Summary

Creating document to summarize all corrections and next steps.

---

## ⏳ PENDING CORRECTIONS (Per User Audit)

### Category A: README/Documentation Improvements

1. **README structure update**:
   - ✅ Move zero-leakage results to header
   - ⏳ Demote 99% to "Legacy Results (With Leakage)" box
   - ⏳ Make corrected results (83.3%) the prominent ones

2. **Dataset provenance table**:
   - ⏳ Add table with: CSV hash, speakers per split, clips per class, normalization method
   - ⏳ Document exact versions of VoxConverse, ESC-50 used

3. **Wilson confidence intervals**:
   - ⏳ Add Wilson score intervals to ALL metrics tables (not just reports)
   - ⏳ Current per-condition analysis uses Wilson CI (done ✅), need to propagate to README tables

4. **Split reproducibility**:
   - ⏳ Add code snippet in README showing GroupShuffleSplit usage
   - ⏳ Link to `create_group_stratified_split.py` and `create_dev_split.py`

5. **Calibration documentation**:
   - ⏳ Clarify that T is optimized on dev, reported on test
   - ⏳ Reference Guo et al. (2017) for temperature scaling

---

### Category B: Additional Experiments (Pre-Scaling)

6. **Symmetric multi-seed**:
   - ⏳ Train **attn+MLP seed 456** to balance seeds (currently: attn-only has 123,456; attn+MLP has 42,123)
   - ⏳ Report mean±SD across all 3 seeds per configuration

7. **ROC-AUC / PR-AUC curves**:
   - ⏳ Add script to compute AUC with bootstrap CI
   - ⏳ Generate ROC and PR curves per class
   - ⏳ Save logit_diff in CSV for threshold analysis

8. **Baseline VAD comparisons**:
   - ⏳ Run **WebRTC VAD** on same test set
   - ⏳ Run **Silero VAD** on same test set
   - ⏳ Add to results table as lower bounds

9. **Hyperparameter mini-grid**:
   - ⏳ Test r ∈ {8, 16, 32} × dropout ∈ {0, 0.05} on attn+MLP
   - ⏳ Use NEW dev split (56 samples) for selection
   - ⏳ Report best config on test

10. **LOSO cross-validation**:
    - ⏳ Leave-One-Speaker-Out for SPEECH (3 folds)
    - ⏳ GroupKFold for NONSPEECH
    - ⏳ More stable estimate than single 1-speaker test split

11. **OPRO post-FT**:
    - ⏳ Optimize prompt on fine-tuned model using dev split
    - ⏳ Expected +1-3% improvement
    - ⏳ 50-100 iterations

---

## 📋 EXECUTION PLAN

### Phase 1: Documentation (Fastest, ~1h)

Priority order:
1. Update README structure (move results, add provenance)
2. Add Wilson CIs to all tables
3. Add GroupShuffleSplit code snippet
4. Clarify calibration methodology

**Why first**: No computation needed, just documentation. Makes paper-ready.

---

### Phase 2: Quick Analyses (Current Models, ~2h)

Can be done with existing checkpoints:

5. ✅ Re-evaluate with fixed A/B tokens (DONE - no change)
6. Train attn+MLP seed 456 (~10 min train + 1 min eval)
7. ROC/PR curves script (~30 min to write + test)
8. WebRTC + Silero baselines (~1h to run + document)

**Why second**: Uses existing infrastructure, no model re-training needed.

---

### Phase 3: Dev Split Experiments (New Training, ~4-6h)

Requires re-training with new 80/56 split:

9. Train best config on new split
10. Proper temperature calibration on dev
11. HP mini-grid search (6 configs × 10 min = 1h)
12. LOSO cross-validation

**Why third**: More expensive (training time), but critical for proper methodology.

---

### Phase 4: Advanced Optimizations (Optional, ~2h)

13. OPRO post-FT on dev

**Why last**: Marginal improvement, can be done after Phase 3 results.

---

## 📊 CURRENT STATUS

**Completed**:
- ✅ Robust A/B token handling (logsumexp aggregation)
- ✅ Dev split created (80 train / 56 dev / 24 test, zero leakage)
- ✅ Per-condition analysis (SNR breakdown)
- ✅ Calibration infrastructure (ECE, Brier, reliability diagrams)
- ✅ Complete citations in README (13 references)
- ✅ Reproducibility section (versions, hardware, hyperparams)

**In Progress**:
- 🚧 Documentation improvements (README restructure)
- 🚧 Progress summary document

**Pending** (prioritized):
1. README updates (Phase 1)
2. Seed 456 + ROC/PR + VAD baselines (Phase 2)
3. Dev split experiments (Phase 3)
4. OPRO post-FT (Phase 4)

---

## 🎯 NEXT IMMEDIATE STEPS

1. **Finish this progress document**
2. **Update README** per audit feedback (Phase 1)
3. **Train attn+MLP seed 456** to balance multi-seed (10 min)
4. **Implement ROC/PR script** with bootstrap CI
5. **Run VAD baselines** (WebRTC + Silero)

After these, we'll have:
- ✅ Documentation complete and paper-ready
- ✅ Symmetric multi-seed validation
- ✅ ROC/PR analysis
- ✅ Classical baselines for comparison

Then user can decide:
- Continue to Phase 3 (dev split experiments)?
- Or scale dataset first?

---

## 📝 FILES CREATED/MODIFIED

### New Files
- `scripts/create_dev_split.py` - Dev split creation with GroupShuffleSplit
- `data/processed/grouped_split_with_dev/train_metadata.csv` - New train (80 samples)
- `data/processed/grouped_split_with_dev/dev_metadata.csv` - Dev set (56 samples)
- `AUDIT_CORRECTIONS_PROGRESS.md` - This document

### Modified Files
- `scripts/evaluate_with_logits.py` - Robust A/B token handling with logsumexp

### To Be Modified
- `README.md` - Structure update, provenance table, Wilson CIs
- (New scripts to be created for ROC/PR, VAD baselines)

---

**Last Updated**: 2025-10-22
**Status**: Ready for Phase 1 (Documentation) and Phase 2 (Quick Analyses)
