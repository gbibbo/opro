# Next Steps - Qwen2-Audio Speech Detection

**Current Status**: Phase 2 Complete (v0.3.0)
**Current Performance**: 62.5% accuracy on challenging short/noisy clips
**Immediate Goal**: ≥75% accuracy with loss masking
**Final Goal**: ≥85% accuracy with scaled dataset + prompt optimization

---

## Quick Action Items (Priority Order)

### 🔴 IMMEDIATE (Ready to Execute)

#### 1. Re-train with Loss Masking (HIGH IMPACT - Est. +5-10%)

**Command**:
```bash
python scripts/finetune_qwen_audio.py
```

**What Changed**:
- Loss now computed only on assistant's A/B token (not entire prompt)
- Better gradient signal → faster convergence
- Implementation already in `finetune_qwen_audio.py`

**Expected Results**:
- Training loss: <8.0 (from 8.69)
- Accuracy: **70-75%** (from 62.5%)
- SPEECH: ≥75%
- NONSPEECH: ≥70%
- Training time: ~8 minutes

**Success Criteria**:
- ✅ No `sampling_rate` warnings during training
- ✅ Training loss <8.0
- ✅ Test accuracy ≥70%

**If Success** → Proceed to Phase 3 (Dataset Scaling)
**If Failure** → Implement NONSPEECH hygiene (Step 2)

---

#### 2. Evaluate New Model

**Command**:
```bash
python scripts/test_normalized_model.py
```

**What to Check**:
- Overall accuracy ≥70%
- Balanced performance (SPEECH ≈ NONSPEECH)
- Confidence gap ≥0.10 (correct vs wrong)
- All outputs are "A" or "B" (constrained decoding working)

**Metrics to Record**:
```
Overall: XX/32 = XX.X%
SPEECH: XX/16 = XX.X%
NONSPEECH: XX/16 = XX.X%
Confidence:
  Overall avg: X.XXX
  Correct avg: X.XXX
  Wrong avg: X.XXX
  Gap: X.XXX
```

---

### 🟡 CONDITIONAL (If accuracy <75%)

#### 3. NONSPEECH Hygiene Validation

**Goal**: Ensure NONSPEECH clips truly contain no speech

**Implementation**:

1. **Install WebRTC VAD**:
   ```bash
   pip install webrtcvad
   ```

2. **Create validation script** (`scripts/validate_nonspeech.py`):
   ```python
   import webrtcvad
   import soundfile as sf

   vad = webrtcvad.Vad(3)  # Aggression mode 3 (most aggressive)

   def compute_speech_activity(audio_path, frame_duration_ms=30):
       audio, sr = sf.read(audio_path)
       # Resample to 16kHz if needed
       # Split into frames
       # Run VAD on each frame
       # Return % of frames with speech
       ...

   # For SPEECH clips
   activity = compute_speech_activity(clip_path)
   assert activity >= 0.70, f"SPEECH clip has only {activity:.1%} speech"

   # For NONSPEECH clips
   activity = compute_speech_activity(clip_path)
   assert activity <= 0.05, f"NONSPEECH clip has {activity:.1%} speech leakage"
   ```

3. **Filter dataset**:
   - Remove NONSPEECH clips with >5% speech activity
   - Remove SPEECH clips with <70% speech activity
   - Re-balance and re-split

4. **Re-train** with cleaned dataset

**Expected Impact**: +5-10% on NONSPEECH class

---

### 🟢 PHASE 3 (If accuracy ≥75%)

#### 4. Dataset Scaling (1-3k Clips)

**Goal**: Expand from 160 → 1000-3000 clips with factorial balance

##### 4.1 Design Factorial Matrix

**Dimensions**:
- **Duration**: [200, 300, 500, 1000] ms (4 levels)
- **SNR**: [-5, 0, +5, +10, +20] dB (5 levels)
- **Class**: [SPEECH, NONSPEECH] (2 levels)

**Total cells**: 4 × 5 × 2 = **40 cells**

**Samples per cell**: 25-75 clips

**Total dataset**: 1000-3000 clips

**Split Strategy**:
- Train: 60% (600-1800 clips)
- Dev: 20% (200-600 clips)
- Test: 20% (200-600 clips)
- **By video/file** to prevent leakage

##### 4.2 Expand NONSPEECH Sources

**Current**: ESC-50 (environmental sounds only)

**Add**:
1. **Music**: GTZAN, FMA, MusicCaps
   - 25% of NONSPEECH samples
   - Various genres (classical, rock, electronic, etc.)
2. **Animals**: ESC-50 subset (dogs, cats, birds, insects)
   - 25% of NONSPEECH samples
3. **Pure Noise**: MUSAN, custom synthetic
   - 25% of NONSPEECH samples
   - White noise, pink noise, ambient noise
4. **Keep ESC-50** environmental sounds
   - 25% of NONSPEECH samples

**Balance**: Equal representation from each category

##### 4.3 Create Expanded Dataset

**Script**: `scripts/create_large_dataset.py` (to be created)

**Process**:
1. Load all source datasets
2. For each cell in factorial matrix:
   - Sample 25-75 clips matching duration/SNR/class
   - Apply SNR mixing if needed
   - Extract center portion (remove padding)
   - Apply peak normalization
3. Split by video/file (not random)
4. Save train/dev/test splits

**Validation**:
- Each cell has ≥25 samples
- Each split has balanced representation
- No leakage between splits

**Timeline**: ~30-60 minutes to create

##### 4.4 Add SpecAugment (Light)

**Implementation** in training script:

```python
from torchaudio.transforms import FrequencyMasking, TimeMasking

class SpecAugment:
    def __init__(self, freq_mask_param=10, time_mask_param_ratio=0.05):
        self.freq_mask = FrequencyMasking(freq_mask_param)
        self.time_mask_param_ratio = time_mask_param_ratio

    def __call__(self, spec):
        # Apply frequency masking
        spec = self.freq_mask(spec)

        # Apply time masking (5% of time dimension)
        time_mask_param = int(spec.shape[-1] * self.time_mask_param_ratio)
        time_mask = TimeMasking(time_mask_param)
        spec = time_mask(spec)

        return spec

# In dataset __getitem__:
if self.augment and random.random() < 0.5:
    # Apply to 50% of training samples
    features = spec_augment(features)
```

**Parameters** (conservative for short clips):
- Frequency masking: F=10 (mask up to 10 mel bins)
- Time masking: T=5% of clip length
- Probability: 50% during training

**Expected Impact**: +2-5% robustness improvement

##### 4.5 Re-train on Large Dataset

**Configuration**:
```python
@dataclass
class TrainingConfig:
    # Same LoRA config
    lora_r: int = 16
    lora_alpha: int = 32

    # More epochs for larger dataset
    num_epochs: int = 5  # increased from 3

    # Larger batch for stability
    batch_size: int = 8  # increased from 4
    gradient_accumulation_steps: int = 2  # reduced from 4
    # Effective batch size = 16 (same)

    # Learning rate schedule
    learning_rate: float = 2e-4
    warmup_steps: int = 100  # NEW
    lr_scheduler_type: str = "cosine"  # NEW

    # Data
    train_csv: Path = project_root / "data" / "processed" / "large_dataset" / "train_metadata.csv"
    dev_csv: Path = project_root / "data" / "processed" / "large_dataset" / "dev_metadata.csv"  # NEW
    test_csv: Path = project_root / "data" / "processed" / "large_dataset" / "test_metadata.csv"

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 50  # Evaluate every 50 steps
```

**Timeline**: ~30-60 minutes (5 epochs × larger dataset)

**Expected Results**:
- Accuracy: **80-85%** on test set
- Better generalization across SNR/duration
- Lower variance in predictions

---

### 🔵 PHASE 4 (After Dataset Scaling)

#### 5. Prompt Optimization with OPRO

##### 5.1 Baseline Prompt Testing

**Create** `scripts/test_prompts.py`:

Test 5-10 hand-crafted prompts on dev set:

**Prompt 1 (Current)**:
```
Choose one:
A) SPEECH (human voice)
B) NONSPEECH (music/noise/silence/animals)

Answer with A or B ONLY.
```

**Prompt 2 (Explicit)**:
```
Listen carefully to this audio clip.

Does it contain human speech?

A) Yes - I hear human voice speaking
B) No - I hear music, noise, silence, or non-human sounds

Your answer:
```

**Prompt 3 (Task-Focused)**:
```
Task: Detect if human speech is present in this audio.

A) SPEECH detected
B) NO SPEECH detected

Answer:
```

**Prompt 4 (Question Format)**:
```
Is there human speech in this audio?

A) Yes
B) No

Your response:
```

**Prompt 5 (Binary Choice)**:
```
Classification task: SPEECH vs NONSPEECH

Listen to the audio.

A = Contains human voice
B = Does not contain human voice

Select:
```

**Evaluation**:
- Run each prompt on full dev set (200-600 samples)
- Record accuracy, confidence, per-class metrics
- Select top 2-3 prompts for OPRO

##### 5.2 Implement OPRO Optimizer

**Create** `scripts/opro_optimize_prompt.py`:

```python
"""OPRO-based prompt optimization for Qwen2-Audio."""

class OPROOptimizer:
    def __init__(self, model, dev_set, metric="accuracy"):
        self.model = model
        self.dev_set = dev_set
        self.metric = metric
        self.history = []

    def generate_variants(self, base_prompts, num_variants=10):
        """Generate prompt variants using meta-prompt."""
        meta_prompt = f"""
        Given these prompts for speech detection:

        {base_prompts}

        Generate {num_variants} improved variants that:
        1. Are clear and unambiguous
        2. Guide the model to focus on human voice
        3. Are concise (2-3 sentences max)
        4. Use A/B format

        Variants:
        """
        # Use LLM to generate variants
        ...

    def evaluate_prompt(self, prompt):
        """Evaluate prompt on dev set."""
        correct = 0
        for sample in self.dev_set:
            pred = self.model.predict(sample.audio, prompt=prompt)
            correct += (pred == sample.label)
        return correct / len(self.dev_set)

    def optimize(self, initial_prompts, num_iterations=5, variants_per_iter=10):
        """Run OPRO optimization."""
        current_prompts = initial_prompts

        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # Generate variants
            variants = self.generate_variants(current_prompts, variants_per_iter)

            # Evaluate all variants
            results = []
            for variant in variants:
                score = self.evaluate_prompt(variant)
                results.append((variant, score))
                print(f"Variant score: {score:.3f}")

            # Keep top K
            results.sort(key=lambda x: x[1], reverse=True)
            current_prompts = [r[0] for r in results[:3]]

            # Store history
            self.history.append(results)

            print(f"\nBest prompt (score={results[0][1]:.3f}):")
            print(results[0][0])

        return results[0][0], results[0][1]
```

**Usage**:
```bash
python scripts/opro_optimize_prompt.py \
  --model checkpoints/qwen2_audio_speech_detection_normalized/final \
  --dev-set data/processed/large_dataset/dev_metadata.csv \
  --initial-prompts prompts/baseline.txt \
  --num-iterations 5 \
  --variants-per-iteration 10 \
  --output prompts/optimized.txt
```

**Timeline**: ~2-4 hours (depends on dev set size)

**Expected Impact**: +3-5% accuracy improvement

##### 5.3 Final Model Comparison Matrix

**Evaluate all model variants**:

| ID | Model | Fine-Tuning | Prompt | Dataset | Accuracy | Notes |
|----|-------|-------------|--------|---------|----------|-------|
| **A1** | Qwen2-Audio Base | No | Baseline | - | ~85% | Normal clips |
| **A2** | Qwen2-Audio Base | No | OPRO | - | ~90% | Normal clips |
| **B1** | Qwen2-Audio | LoRA | Baseline | 128 | 62.5% | Current |
| **B2** | Qwen2-Audio | LoRA + Mask | Baseline | 128 | ~75% | Next |
| **B3** | Qwen2-Audio | LoRA + Mask | Baseline | 1-3k | ~80% | Phase 3 |
| **B4** | Qwen2-Audio | LoRA + Mask | OPRO | 1-3k | ~85% | Phase 4 |
| **C1** | Qwen3-Omni | No | OPRO | - | TBD | Future |
| **D1** | WebRTC VAD | - | - | - | ~60-70% | Baseline |
| **D2** | Silero VAD | - | - | - | ~70-80% | Baseline |

**Evaluation Protocol**:
- Same test set for all models
- Same constrained A/B decoding
- Same confidence threshold (learned on dev)
- Report: accuracy, per-class, per-SNR, per-duration
- DT50/DT75 (psychometric curves)

---

### 🟣 PHASE 5 (Final Evaluation & Paper)

#### 6. Comprehensive Evaluation

##### 6.1 Main Results Table

**Metrics**:
- Overall accuracy
- SPEECH accuracy
- NONSPEECH accuracy
- Precision, Recall, F1 per class
- Confidence calibration (ECE, MCE)

**Breakdowns**:
- By SNR: [-5, 0, +5, +10, +20] dB
- By duration: [200, 300, 500, 1000] ms
- By NONSPEECH type: [music, animals, noise, environment]

##### 6.2 Psychometric Curves

**DT50 (Detection Threshold 50%)**:
- SNR at which accuracy = 50%
- Lower is better (detects at lower SNR)

**DT75 (Detection Threshold 75%)**:
- SNR at which accuracy = 75%

**Plot**:
- X-axis: SNR (dB)
- Y-axis: Accuracy (%)
- Curves for each model variant
- Show DT50/DT75 markers

##### 6.3 Latency Analysis

**Components**:
1. Audio loading: Measure with `time.time()`
2. Feature extraction: Measure separately
3. Model forward pass: Measure with `torch.cuda.Event`
4. Post-processing: Minimal

**Comparison**:
- Qwen2-Audio (7B): ~300ms
- Qwen3-Omni: TBD
- WebRTC VAD: ~10-30ms
- Silero VAD: ~30-100ms

**Trade-off analysis**:
- Accuracy vs Latency plot
- Pareto frontier identification

##### 6.4 Error Analysis

**Failure Modes**:
1. False Positives (NONSPEECH → SPEECH)
   - Which NONSPEECH types are confused?
   - Music with singing-like patterns?
2. False Negatives (SPEECH → NONSPEECH)
   - Very short utterances (200ms)?
   - Low SNR (+0dB or below)?
   - Specific phonemes/sounds?

**Visualizations**:
- Confusion matrix
- Error rate by SNR/duration
- Audio examples of failures

---

## Timeline Summary

| Phase | Duration | Cumulative | Target Accuracy |
|-------|----------|------------|-----------------|
| **Immediate** (Steps 1-2) | ~15 mins | 0.25 hours | 70-75% |
| **Conditional** (Step 3) | ~2 hours | 2.25 hours | 75-80% |
| **Phase 3** (Step 4) | ~2-3 hours | 5 hours | 80-85% |
| **Phase 4** (Step 5) | ~3-5 hours | 10 hours | 85%+ |
| **Phase 5** (Step 6) | ~5-10 hours | 20 hours | Paper ready |

**Total**: ~20 hours from current state to paper-ready results

---

## Decision Tree

```
Current: 62.5% accuracy
    ↓
[1] Re-train with loss masking (~10 mins)
    ↓
Test accuracy ≥70%?
    ├─ YES → [Phase 3] Dataset scaling
    │         ↓
    │    Accuracy ≥80%?
    │         ├─ YES → [Phase 4] Prompt optimization
    │         │         ↓
    │         │    Accuracy ≥85%?
    │         │         ├─ YES → [Phase 5] Final evaluation
    │         │         └─ NO → Investigate failures
    │         └─ NO → Add SpecAugment, more data
    │
    └─ NO → [3] NONSPEECH hygiene
              ↓
         Re-train
              ↓
         Accuracy ≥75%?
              ├─ YES → [Phase 3]
              └─ NO → Review dataset quality
```

---

## Risk Mitigation

### Risk 1: Loss masking doesn't improve accuracy

**Probability**: Low (this fix is well-established)

**Mitigation**:
- Verify loss masking is working (check token positions)
- Try different masking strategies (mask from assistant token, not before)
- Increase epochs to 5

### Risk 2: Dataset scaling introduces noise

**Probability**: Medium

**Mitigation**:
- Validate all samples with VAD before inclusion
- Manual inspection of random 100 samples per class
- Gradual scaling (500 → 1000 → 2000 → 3000)

### Risk 3: OPRO doesn't find better prompts

**Probability**: Medium

**Mitigation**:
- Start with strong baselines (hand-crafted prompts)
- Use larger pool of initial prompts (10-20)
- Combine with manual prompt engineering

### Risk 4: Model plateaus below target

**Probability**: Low

**Mitigation**:
- Increase LoRA rank to 32 or 64
- Try full fine-tuning (not just LoRA)
- Ensemble multiple models
- Switch to larger base model (13B or Qwen3-Omni)

---

## Success Criteria

### Minimum Viable (for publication):
- ✅ Overall accuracy ≥75%
- ✅ SPEECH ≥80%, NONSPEECH ≥70%
- ✅ Improvement over VAD baselines
- ✅ Thorough error analysis

### Target (for strong publication):
- ✅ Overall accuracy ≥85%
- ✅ Balanced performance across SNR levels
- ✅ Competitive with state-of-the-art VADs
- ✅ Demonstrate prompt optimization benefit
- ✅ Psychometric curves + latency analysis

### Stretch (for top-tier venue):
- ✅ Overall accuracy ≥90%
- ✅ Better than Silero VAD at comparable latency
- ✅ Novel insights from error analysis
- ✅ Released model + dataset
- ✅ Reproducible pipeline

---

## Resources Needed

### Computational:
- GPU: 1× RTX 4090 or similar (24GB VRAM)
- Training time: ~20-30 hours total
- Storage: ~50GB (datasets + checkpoints)

### Data:
- VoxConverse: Already have
- ESC-50: Already have
- Music datasets: GTZAN (free), FMA (free)
- MUSAN: Free download

### Software:
- All libraries already installed
- WebRTC VAD: `pip install webrtcvad`
- (Optional) Silero VAD: `pip install silero-vad`

---

## Open Questions

1. **Optimal LoRA rank** for this task?
   - Current: r=16
   - Try: r=8, 32, 64?

2. **Best prompt format** for A/B constrained decoding?
   - More explicit options?
   - Question vs statement format?

3. **Threshold optimization** for confidence scores?
   - Use dev set to find optimal p(A) threshold
   - Temperature scaling for calibration?

4. **SpecAugment parameters** for ultra-short clips?
   - Current: F=10, T=5%
   - Too aggressive for 200ms clips?

5. **Ensemble strategy**?
   - Multiple LoRA adapters?
   - Multiple base models?
   - Voting or averaging?

---

**Last Updated**: 2025-10-19
**Next Action**: Execute Step 1 (Re-train with loss masking)
**Expected Completion**: 15 minutes
