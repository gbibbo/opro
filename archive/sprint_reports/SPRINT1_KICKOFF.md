# Sprint 1: KICKOFF - Dataset Ingestion

**Start Date:** 2025-10-08
**Status:** 🚀 **READY TO START**
**Prerequisites:** ✅ Sprint 0 Complete

---

## 🎯 Sprint Goal

> Implement robust loaders for all 5 datasets with high-precision ground truth, creating a unified FrameTable for downstream processing.

---

## 📋 Sprint Objectives

### Primary Deliverables

1. **Full RTTM Loaders** (DIHARD, VoxConverse)
   - Parse RTTM format with onset/offset timestamps
   - Handle speaker labels and overlaps
   - Validate against official counts

2. **AVA-Speech Loader**
   - Parse CSV annotations (frame-level)
   - Map labels to SPEECH/NONSPEECH
   - Extract conditions (clean/music/noise)
   - Convert frame timestamps to seconds

3. **AMI Loader**
   - Parse forced alignment files
   - Convert word-level to frame-level annotations
   - Handle overlapping speakers
   - Validate alignment precision (~10ms)

4. **Unified FrameTable**
   - Combine all datasets
   - Validate consistency
   - Export to parquet
   - Add dataset-specific metadata

5. **Validation & Testing**
   - Unit tests for each loader
   - Integration tests for FrameTable
   - Validate against official dataset statistics
   - Update documentation

---

## 📊 Task Breakdown

### T-103: Implement Full RTTM Loaders

**Subtasks:**

#### T-103a: DIHARD Loader
- [ ] Parse DIHARD II/III RTTM format
- [ ] Extract speaker segments with timestamps
- [ ] Handle UEM (Un-partitioned Evaluation Map) files
- [ ] Validate against official counts
- [ ] Write unit tests
- [ ] Update documentation

**Acceptance Criteria:**
- Loads all DIHARD dev/test splits correctly
- Matches official segment counts (±1%)
- Handles edge cases (overlaps, gaps)
- Unit tests pass

#### T-103b: VoxConverse Loader
- [ ] Parse VoxConverse v0.3 RTTM format
- [ ] Handle v0.3 fixes and corrections
- [ ] Extract speaker segments
- [ ] Validate against official counts
- [ ] Write unit tests
- [ ] Update documentation

**Acceptance Criteria:**
- Loads all VoxConverse splits correctly
- Matches official segment counts (±1%)
- Uses correct v0.3 annotations
- Unit tests pass

---

### T-103c: AVA-Speech Loader

**Subtasks:**
- [ ] Parse AVA-Speech CSV format
- [ ] Map frame labels to binary SPEECH/NONSPEECH
- [ ] Extract acoustic conditions (clean/music/noise)
- [ ] Convert frame timestamps (25 fps → seconds)
- [ ] Handle dense frame-level annotations
- [ ] Validate frame counts
- [ ] Write unit tests
- [ ] Update documentation

**Acceptance Criteria:**
- Loads AVA-Speech annotations correctly
- Correctly maps all label types
- Extracts conditions accurately
- Frame timing is precise (40ms resolution)
- Unit tests pass

---

### T-103d: AMI Loader

**Subtasks:**
- [ ] Parse AMI forced alignment files
- [ ] Convert word-level to frame-level (10ms steps)
- [ ] Handle overlapping speakers
- [ ] Merge adjacent words into speech segments
- [ ] Validate alignment precision
- [ ] Write unit tests
- [ ] Update documentation

**Acceptance Criteria:**
- Loads AMI meetings correctly
- Word-to-frame conversion is accurate (10ms precision)
- Handles overlaps correctly
- Timing is precise
- Unit tests pass

---

### T-103e: AVA-ActiveSpeaker Loader (Optional)

**Subtasks:**
- [ ] Parse AVA-ActiveSpeaker annotations
- [ ] Map speaking & audible to SPEECH
- [ ] Handle video frame timing (25 fps)
- [ ] Validate against frame counts
- [ ] Write unit tests
- [ ] Update documentation

**Acceptance Criteria:**
- Loads annotations correctly
- Binary mapping is correct
- Timing is accurate
- Unit tests pass

---

### T-104: Build Unified FrameTable

**Subtasks:**
- [ ] Combine all dataset loaders
- [ ] Validate schema consistency
- [ ] Add dataset-specific metadata columns
- [ ] Handle different precision levels
- [ ] Export to parquet format
- [ ] Add statistics function (duration, % speech, etc.)
- [ ] Write integration tests
- [ ] Update documentation

**Acceptance Criteria:**
- All datasets load into single FrameTable
- Schema is consistent across datasets
- Metadata is complete
- Can filter by dataset, split, condition
- Save/load works correctly
- Integration tests pass

---

## 🔧 Technical Details

### RTTM Format (DIHARD, VoxConverse)

```
SPEAKER <file-id> 1 <start-time> <duration> <NA> <NA> <speaker-id> <NA> <NA>
```

**Key points:**
- Times in seconds (float)
- Duration NOT end time
- Speaker labels may overlap
- Need to handle gaps

**Reference:** NIST RTTM specification

### AVA-Speech Format

```csv
video_id, frame_timestamp, label
abc123, 0.5, SPEECH_CLEAN
abc123, 0.6, NO_SPEECH
abc123, 0.7, SPEECH_WITH_MUSIC
```

**Key points:**
- Frame rate: 25 fps (40ms per frame)
- Labels: SPEECH_CLEAN, SPEECH_WITH_MUSIC, SPEECH_WITH_NOISE, NO_SPEECH
- Dense annotations (every frame)

**Reference:** AVA-Speech paper

### AMI Format

Forced alignment files (CTM or TextGrid):
- Word-level timestamps
- Multiple speakers per meeting
- 10ms step size
- Overlapping speakers common

**Reference:** AMI corpus documentation

---

## 📚 Datasets Overview

### 1. DIHARD (Diarization Hard)

**Stats:**
- DIHARD II: ~5 hours (dev), ~10 hours (test)
- DIHARD III: ~6 hours (dev), ~11 hours (test)
- Multiple domains (clinical, court, restaurant, etc.)
- High overlap rate

**Ground truth:** RTTM with precise onset/offset

**Use case:** Challenging diarization scenarios

### 2. VoxConverse

**Stats:**
- ~50 hours total
- YouTube videos (interviews, debates)
- v0.3 has corrected annotations

**Ground truth:** RTTM with manual corrections

**Use case:** Conversational speech in media

### 3. AVA-Speech

**Stats:**
- ~160 hours from movies
- Dense frame-level labels (25 fps)
- 3 acoustic conditions (clean/music/noise)

**Ground truth:** Frame-level labels (40ms precision)

**Use case:** Speech in noisy/musical backgrounds

### 4. AMI Corpus

**Stats:**
- ~100 hours of meetings
- Word-level forced alignment
- Multiple microphones per meeting

**Ground truth:** Word-level alignment (~10ms precision)

**Use case:** Overlapping multi-speaker conversations

### 5. AVA-ActiveSpeaker (Optional)

**Stats:**
- ~65 hours from movies
- Frame-level speaking/audible labels

**Ground truth:** Frame-level binary labels

**Use case:** Visual+audio speech detection

---

## 🧪 Testing Strategy

### Unit Tests

For each loader:
- [ ] Test with mock/sample data
- [ ] Test edge cases (empty files, overlaps, gaps)
- [ ] Test error handling (malformed data)
- [ ] Test PROTOTYPE_MODE limiting
- [ ] Test save/load functionality

### Integration Tests

- [ ] Load all datasets into single FrameTable
- [ ] Verify schema consistency
- [ ] Check statistics (total duration, speech %)
- [ ] Test filtering by dataset/split/condition
- [ ] Test export to parquet

### Validation Tests

- [ ] Compare counts with official statistics
- [ ] Verify timing precision
- [ ] Check for data loss/corruption
- [ ] Validate metadata completeness

---

## 📊 Success Criteria

### Quantitative

- [ ] All 5 dataset loaders implemented
- [ ] 100% unit test pass rate
- [ ] Load times < 5 seconds for prototype data
- [ ] Official count validation within ±1%
- [ ] No data loss in FrameTable conversion

### Qualitative

- [ ] Code is clean and well-documented
- [ ] Error messages are helpful
- [ ] Easy to add new datasets
- [ ] Logging is comprehensive
- [ ] README updated with examples

---

## 🗂️ File Structure

New files to create:

```
src/qsm/data/
├── loaders.py                 # Update with full implementations
├── rttm.py                    # RTTM parsing utilities (new)
├── ava_speech.py              # AVA-Speech loader (new)
├── ami.py                     # AMI loader (new)
└── validators.py              # Data validation utils (new)

tests/
├── test_loaders.py            # Update with full tests
├── test_rttm.py               # RTTM loader tests (new)
├── test_ava_speech.py         # AVA-Speech tests (new)
├── test_ami.py                # AMI tests (new)
└── test_integration.py        # Integration tests (new)

configs/datasets/
├── dihard.yaml                # DIHARD config (update)
├── voxconverse.yaml           # VoxConverse config (update)
├── ava_speech.yaml            # AVA-Speech config (update)
├── ami.yaml                   # AMI config (update)
└── ava_activespeaker.yaml     # AVA-AS config (update)

scripts/
├── validate_datasets.py       # Dataset validation script (new)
└── build_frame_table.py       # Build unified FrameTable (new)
```

---

## 🔄 Development Workflow

1. **Setup:**
   ```bash
   # Ensure Sprint 0 dependencies installed
   pip install -e .
   ```

2. **For each dataset:**
   - Write loader function
   - Write unit tests
   - Test with mock data
   - Validate against real data (if available)
   - Document usage

3. **Integration:**
   - Combine all loaders
   - Test unified FrameTable
   - Validate consistency
   - Write integration tests

4. **Finalize:**
   - Update documentation
   - Run all tests
   - Update README
   - Commit and push

---

## 📈 Estimated Timeline

### Day 1: RTTM Loaders
- Morning: DIHARD loader + tests
- Afternoon: VoxConverse loader + tests
- Evening: Integration and validation

### Day 2: CSV/Alignment Loaders
- Morning: AVA-Speech loader + tests
- Afternoon: AMI loader + tests
- Evening: AVA-ActiveSpeaker (optional)

### Day 3: Integration & Documentation
- Morning: Unified FrameTable
- Afternoon: Integration tests + validation
- Evening: Documentation + README update

**Total: 2-3 days** (adjustable based on complexity)

---

## 🎯 Definition of Done

Sprint 1 is complete when:

- [ ] All 5 dataset loaders implemented and tested
- [ ] Unified FrameTable working
- [ ] All unit tests passing (100%)
- [ ] Integration tests passing
- [ ] Validation against official counts complete
- [ ] Documentation updated
- [ ] README updated with examples
- [ ] All changes committed to GitHub
- [ ] Smoke test still passes
- [ ] Logs properly saved

---

## 📚 Resources

### Documentation

- [pyannote.database RTTM docs](https://github.com/pyannote/pyannote-database)
- [DIHARD challenge](https://dihardchallenge.github.io/dihard3/)
- [VoxConverse repo](https://github.com/joonson/voxconverse)
- [AVA-Speech paper](https://arxiv.org/abs/1808.00606)
- [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/)
- [NIST RTTM spec](https://www.nist.gov/system/files/documents/itl/iad/mig/KWS14-evalplan-v11.pdf)

### Code References

- Sprint 0 loaders (skeleton in `src/qsm/data/loaders.py`)
- pyannote RTTMLoader examples
- pandas CSV parsing examples

---

## 🚦 Getting Started

### Step 1: Review Sprint 0 Closure

Read [SPRINT0_CLOSURE.md](SPRINT0_CLOSURE.md) for context.

### Step 2: Set Up Environment

```bash
# Activate environment
conda activate opro

# Verify Sprint 0 tests pass
python scripts/run_all_tests.py
```

### Step 3: Start with T-103a (DIHARD Loader)

Create a new branch (optional):
```bash
git checkout -b sprint1-dataset-ingestion
```

Begin implementing DIHARD loader:
```bash
# Edit src/qsm/data/loaders.py
# Add DIHARD-specific parsing logic
```

### Step 4: Follow TDD

1. Write test first
2. Implement functionality
3. Run tests
4. Refactor
5. Commit

---

## 📞 Questions?

Refer to:
- [SPRINT0_SUMMARY.md](SPRINT0_SUMMARY.md) for infrastructure details
- [TESTING_GUIDE.md](TESTING_GUIDE.md) for testing instructions
- [QUICKSTART.md](QUICKSTART.md) for quick commands

---

**Sprint 1 is ready to begin!** 🚀

Let's build robust dataset loaders with high-precision ground truth.

---

**Document version:** 1.0
**Sprint:** 1 (Dataset Ingestion)
**Status:** 🚀 READY TO START
**Previous:** Sprint 0 ✅ COMPLETE
