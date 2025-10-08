# Audio Status - 2.5 GB Real Audio Available! 🎉

**Updated:** 2025-10-08 after VoxConverse download
**Status:** ✅ 2.5 GB of real audio downloaded

---

## ✅ Audio Datasets Downloaded

### 1. AMI Corpus - 5 Files (298 MB)
**Location:** `data/raw/ami/audio/`
**Total:** ~38 minutes of meeting audio

### 2. VoxConverse - 216 Files (2.2 GB)
**Location:** `data/raw/voxconverse/audio/dev/`
**Total:** 216 WAV files from YouTube with speaker diarization

---

## 📊 Total: 221 files, 2.5 GB

**Ready for inference!** 🚀

---

## 🎧 Listen Now

```bash
# AMI meeting
start data/raw/ami/audio/ES2002a.wav

# VoxConverse
start data/raw/voxconverse/audio/dev/abjxc.wav
```

---

## 📥 Download More

```bash
# VoxConverse test set
python scripts/download_voxconverse_audio.py --splits test

# More AMI meetings
python scripts/download_datasets.py --datasets ami --force-full
```
