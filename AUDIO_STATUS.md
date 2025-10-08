# Audio Download Status

**Last Updated:** 2025-10-08

## Overview

This document tracks the availability of real audio files for each dataset in the project.

## Current Status (PROTOTYPE_MODE)

| Dataset | Status | Files | Size | Format | Notes |
|---------|--------|-------|------|--------|-------|
| **AMI** | ✅ Downloaded | 5 meetings | ~298 MB | WAV (Mix-Headset) | Ready for inference |
| **VoxConverse** | ✅ Downloaded | 5 conversations | ~45 MB | WAV (preprocessed) | Ready for inference |
| **AVA-Speech** | ✅ Downloaded | 5 videos | ~1.3 GB | MP4 (with audio) | Ready for extraction |
| **AVA-ActiveSpeaker** | ⏳ Manual Required | 0 | 0 MB | MP4 → WAV | Requires Google license |
| **DIHARD** | ❌ Removed | - | - | - | Expensive LDC license |

**Total Prototype Audio:** ~1.6 GB (15 files)

## Download Commands

### Prototype Mode (5 samples per dataset)

**AMI Corpus:**
```bash
python scripts/download_datasets.py --datasets ami
```
Downloads 5 AMI meetings with Mix-Headset audio.

**VoxConverse:**
```bash
python scripts/download_voxconverse_audio.py --splits dev
```
Downloads and keeps only 5 VoxConverse dev samples.

**AVA-Speech:**
```bash
python scripts/download_ava_speech.py --datasets ava-speech
```
Downloads 5 AVA-Speech videos with annotations.

**AVA-ActiveSpeaker:**
```bash
python scripts/download_ava_activespeaker.py
```
Creates placeholder structure. Requires manual download from Google.

### Full Dataset Downloads (NOT EXECUTED - Space Constraints)

**AMI Corpus (Full):**
```bash
python scripts/download_datasets.py --datasets ami --force-full
```
Downloads ~160 meetings from AMI corpus mirror. Estimated size: ~15-20 GB.

**VoxConverse (Full):**
```bash
python scripts/download_voxconverse_audio.py --splits dev test --force-full
```
Downloads complete VoxConverse dev + test sets. Estimated size: ~5 GB.

**AVA-Speech (Full):**
```bash
python scripts/download_ava_speech.py --datasets ava-speech --force-full
```
Downloads complete AVA-Speech dataset (160 videos). Estimated size: ~40 GB.

**AVA-ActiveSpeaker (Full):**
```bash
python scripts/download_ava_activespeaker.py --force-full
```
Creates full placeholder structure. Requires manual download of videos and extraction of audio. Estimated size: 10+ GB.

## Audio Details

### AMI Corpus
- **Downloaded:** 5 meetings (ES2002a through ES2004b)
- **Format:** 16kHz mono WAV (Mix-Headset channel)
- **Total Duration:** ~38 minutes
- **Source:** http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/
- **License:** CC BY 4.0
- **Annotations:** Word-level forced alignment (XML)

### VoxConverse
- **Downloaded:** 5 conversations from dev split
- **Format:** 16kHz mono WAV (preprocessed)
- **Total Duration:** ~10 minutes
- **Source:** https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
- **License:** Research use
- **Annotations:** RTTM v0.3 with speaker diarization

### AVA-Speech
- **Downloaded:** 5 videos from trainval split
- **Format:** MP4 video with embedded audio
- **Total Size:** ~1.3 GB
- **Source:** https://research.google.com/ava/download.html (S3 mirror)
- **License:** Research use
- **Annotations:** CSV with frame-level speech labels (Speech/Noise/Music)
- **Frame Rate:** 25 fps
- **Conditions:** Clean, music, noise backgrounds
- **Next Steps:**
  1. Extract audio using ffmpeg: `ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 audio.wav`
  2. Align frame-level labels with extracted audio

### AVA-ActiveSpeaker
- **Downloaded:** 0 (placeholder annotations only)
- **Format:** MP4 video → extract WAV audio
- **Source:** https://github.com/cvdfoundation/ava-dataset
- **License:** Requires Google approval
- **Annotations:** CSV with frame-level speaking labels
- **Next Steps:**
  1. Visit GitHub page and accept license
  2. Download video clips from Google
  3. Extract audio using ffmpeg: `ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 audio.wav`

### DIHARD (REMOVED)
- **Status:** Permanently removed from project
- **Reason:** Requires expensive LDC license ($100-300 USD)
- **Alternative:** Using VoxConverse for RTTM-based evaluation

## Storage Requirements

### Current (Prototype Mode)
- **Audio:** ~1.6 GB
- **Annotations:** ~2 MB
- **Total:** ~1.6 GB

### Full Datasets (Not Downloaded)
- **AMI:** ~15-20 GB
- **VoxConverse:** ~5 GB
- **AVA-Speech:** ~40 GB
- **AVA-ActiveSpeaker:** ~10+ GB
- **Total:** ~70-80 GB

## Verification

To verify downloaded audio:

```bash
# Check AMI audio
ls -lh data/raw/ami/audio/*.wav

# Check VoxConverse audio
ls -lh data/raw/voxconverse/audio/dev/*.wav

# Check AVA-Speech videos
ls -lh data/raw/ava-speech/videos/trainval/*.mp4

# Count total files
find data/raw -name "*.wav" -o -name "*.mp4" | wc -l  # Should be 15

# Check total size
du -sh data/raw  # Should be ~1.6 GB
```

## Next Steps

1. ✅ AMI prototype audio downloaded
2. ✅ VoxConverse prototype audio downloaded
3. ✅ AVA-Speech prototype videos downloaded
4. ⏳ AVA-ActiveSpeaker manual download pending
5. ⏳ Build unified annotation pipeline (Sprint 1)
6. ⏳ Extract segments at target durations (Sprint 2)

## Troubleshooting

### AMI Download Fails
- Check internet connection to http://groups.inf.ed.ac.uk
- Mirror may be temporarily down, try again later
- Some meetings may be unavailable, script will skip them

### VoxConverse Download Fails
- Check internet connection to https://www.robots.ox.ac.uk
- ZIP files are large (~1-2 GB), ensure stable connection
- If extraction fails, delete ZIP and re-run

### AVA-ActiveSpeaker
- Cannot auto-download (Google license required)
- Follow manual instructions in script output
- Use placeholder annotations for development

## Configuration

Audio downloads respect the `PROTOTYPE_MODE` setting in `config.yaml`:

```yaml
PROTOTYPE_MODE: true   # true = 5 samples, false = full datasets
PROTOTYPE_SAMPLES: 5   # Number of samples in prototype mode
```

To switch to full downloads:
1. Set `PROTOTYPE_MODE: false` in `config.yaml`
2. Run download scripts with `--force-full` flag
3. Ensure sufficient storage (~35 GB)
