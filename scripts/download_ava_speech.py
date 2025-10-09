#!/usr/bin/env python3
"""
Download AVA-Speech dataset with automatic cleanup for PROTOTYPE_MODE.

Downloads:
  1) Annotations: ava_speech_labels_v1.csv
  2) Video list: ava_speech_file_names_v1.txt
  3) Videos (trainval) from S3 using the video list

Sources:
- Google Research AVA (AVA-Speech section) -> ava_speech_labels_v1.csv
  https://research.google.com/ava/download.html#ava_speech_download
- CVDF AVA repository (official S3 links)
  https://github.com/cvdfoundation/ava-dataset

Usage:
  python download_ava_speech.py --datasets ava-speech
  python download_ava_speech.py --datasets ava-speech --force-full

Requirements:
  pip install requests tqdm
"""

import argparse
import concurrent.futures as futures
import logging
import subprocess
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from qsm import CONFIG, PROTOTYPE_MODE, PROTOTYPE_SAMPLES

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Official S3 endpoints (see CVDF + Google Research)
S3_ANN_BASE = "https://s3.amazonaws.com/ava-dataset/annotations/"
S3_VID_BASE = "https://s3.amazonaws.com/ava-dataset/trainval/"

AVA_SPEECH_LABELS = "ava_speech_labels_v1.csv"  # annotations (Speech/Noise/Music)
AVA_SPEECH_FILENAMES = "ava_speech_file_names_v1.txt"  # list of video files


def make_session() -> requests.Session:
    """Create requests session with retry strategy."""
    sess = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    sess.headers.update({"User-Agent": "ava-speech-downloader/1.0"})
    return sess


def human_size(num: int) -> str:
    """Convert bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def ensure_dir(p: Path):
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def need_download(dest: Path, expected_size: int | None, session: requests.Session) -> bool:
    """Check if file needs to be downloaded."""
    if not dest.exists():
        return True
    if expected_size is None:
        return False
    # Size matches → don't re-download
    if dest.stat().st_size == expected_size:
        return False
    # Size differs → try to re-download
    return True


def head_size(url: str, session: requests.Session) -> int | None:
    """Get file size from HEAD request."""
    try:
        r = session.head(url, timeout=15)
        if r.ok and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except requests.RequestException:
        return None
    return None


def download_file(url: str, dest: Path, session: requests.Session, desc: str | None = None) -> bool:
    """Download a file with progress bar."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    expected = head_size(url, session)
    if not need_download(dest, expected, session):
        logger.info(f"✅ {dest.name} already exists, skipping")
        return True

    try:
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with (
                open(tmp, "wb") as f,
                tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc or dest.name,
                    leave=False,
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=1024 * 512):
                    if chunk:
                        f.write(chunk)
                        if total > 0:
                            pbar.update(len(chunk))
        tmp.replace(dest)
        # Simple size verification
        if expected and dest.stat().st_size != expected:
            logger.warning(f"Unexpected size: {dest} ({dest.stat().st_size} vs {expected})")
        logger.info(f"✅ Downloaded {dest.name} ({human_size(dest.stat().st_size)})")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


def load_filename_list(path: Path) -> list[str]:
    """Load video filename list from text file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # File is separated by spaces/newlines
    # Return a normalized unique list
    names = []
    for tok in text.split():
        tok = tok.strip()
        if not tok:
            continue
        # Sanity: expected extensions
        if any(tok.endswith(ext) for ext in (".mp4", ".mkv", ".webm")):
            names.append(tok)
    # Remove duplicates preserving order
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def download_annotations(out_dir: Path, session: requests.Session) -> Path:
    """Download AVA-Speech annotations."""
    ann_dir = out_dir / "annotations"
    ensure_dir(ann_dir)

    csv_url = S3_ANN_BASE + AVA_SPEECH_LABELS
    csv_path = ann_dir / AVA_SPEECH_LABELS
    logger.info(f"Downloading annotations → {csv_path.name}")
    ok = download_file(csv_url, csv_path, session, desc="annotations.csv")
    if not ok:
        logger.error("Failed to download annotation CSV")
        sys.exit(2)

    # Also save video filename list
    list_url = S3_ANN_BASE + AVA_SPEECH_FILENAMES
    list_path = ann_dir / AVA_SPEECH_FILENAMES
    logger.info(f"Downloading video list → {list_path.name}")
    ok = download_file(list_url, list_path, session, desc="file_names.txt")
    if not ok:
        logger.warning(
            "Failed to download filename list. (You can continue if you already have it)"
        )
    return list_path


def _dl_video_one(args):
    """Download a single video (worker function)."""
    idx, name, out_dir_str = args
    session = make_session()
    url = S3_VID_BASE + name
    dest = Path(out_dir_str) / name
    ensure_dir(dest.parent)

    # Expected size (HEAD)
    expected = head_size(url, session)
    if not need_download(dest, expected, session):
        return (name, True, expected, dest.stat().st_size, "skip")

    ok = download_file(url, dest, session, desc=f"{idx:04d}-{name}")
    size = dest.stat().st_size if dest.exists() else 0
    return (name, ok, expected, size, "ok" if ok else "fail")


def download_videos(
    out_dir: Path,
    file_list_path: Path,
    max_videos: int | None,
    workers: int,
    prototype: bool = True,
):
    """Download AVA-Speech videos."""
    vids_dir = out_dir / "videos" / "trainval"
    ensure_dir(vids_dir)

    names = load_filename_list(file_list_path)

    # PROTOTYPE MODE: limit to N samples
    if prototype:
        names = names[: min(PROTOTYPE_SAMPLES, len(names))]
        logger.info(
            f"PROTOTYPE_MODE: Downloading only {len(names)} videos "
            f"(out of {len(load_filename_list(file_list_path))} total)"
        )
    elif max_videos is not None:
        names = names[:max_videos]

    logger.info(f"Downloading {len(names)} videos to {vids_dir} (workers={workers})")

    tasks = [(i, n, str(vids_dir)) for i, n in enumerate(names, start=1)]
    ok_count = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for name, ok, expected, size, status in tqdm(
            ex.map(_dl_video_one, tasks),
            total=len(tasks),
            desc="videos",
            unit="file",
            leave=True,
        ):
            if ok:
                ok_count += 1
            else:
                logger.error(f"{name} failed (size={size}, expected={expected})")

    logger.info(f"Videos OK: {ok_count}/{len(names)}")

    # PROTOTYPE MODE: Clean up extra files if any exist
    if prototype:
        all_videos = sorted(vids_dir.glob("*.mp4"))
        if len(all_videos) > PROTOTYPE_SAMPLES:
            logger.info(
                f"PROTOTYPE_MODE: Keeping only {PROTOTYPE_SAMPLES} files, "
                f"removing {len(all_videos) - PROTOTYPE_SAMPLES}"
            )
            for video_file in all_videos[PROTOTYPE_SAMPLES:]:
                video_file.unlink()
                logger.info(f"Removed {video_file.name}")


def extract_audio_from_videos(videos_dir: Path, audio_dir: Path):
    """Extract audio from video files using ffmpeg."""
    ensure_dir(audio_dir)

    # Check if ffmpeg is available
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            logger.error("ffmpeg not found. Please install ffmpeg to extract audio.")
            logger.info("Install: https://ffmpeg.org/download.html")
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.error("ffmpeg not found. Please install ffmpeg to extract audio.")
        logger.info("Install: https://ffmpeg.org/download.html")
        return

    # Get all video files
    video_files = (
        list(videos_dir.glob("*.mp4"))
        + list(videos_dir.glob("*.mkv"))
        + list(videos_dir.glob("*.webm"))
    )

    if not video_files:
        logger.warning("No video files found to extract audio from")
        return

    logger.info(f"Extracting audio from {len(video_files)} videos...")

    for video_file in tqdm(video_files, desc="Extracting audio", unit="file"):
        audio_file = audio_dir / f"{video_file.stem}.wav"

        # Skip if audio already exists
        if audio_file.exists():
            logger.debug(f"Audio already exists: {audio_file.name}")
            continue

        try:
            # Extract audio to WAV format (16kHz, mono, 16-bit)
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(video_file),
                    "-vn",  # No video
                    "-acodec",
                    "pcm_s16le",  # 16-bit PCM
                    "-ar",
                    "16000",  # 16kHz sample rate
                    "-ac",
                    "1",  # Mono
                    "-y",  # Overwrite
                    str(audio_file),
                ],
                check=True,
                capture_output=True,
                timeout=300,
            )
            logger.info(f"✅ Extracted audio: {audio_file.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to extract audio from {video_file.name}: {e}")
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout extracting audio from {video_file.name}")

    # Summary
    audio_count = len(list(audio_dir.glob("*.wav")))
    total_audio_size = sum(f.stat().st_size for f in audio_dir.glob("*.wav"))
    logger.info(f"Audio files extracted: {audio_count}")
    logger.info(f"Total audio size: {human_size(total_audio_size)}")


def download_ava_speech(
    data_root: Path,
    prototype: bool = True,
    max_videos: int | None = None,
    workers: int = 8,
):
    """
    Download AVA-Speech dataset with REAL AUDIO.

    Args:
        data_root: Root directory for data storage
        prototype: If True, download only PROTOTYPE_SAMPLES videos
        max_videos: Maximum videos to download (overrides prototype if set)
        workers: Number of concurrent download workers
    """
    out_dir = data_root / "raw" / "ava-speech"
    ensure_dir(out_dir)

    session = make_session()
    list_path = out_dir / "annotations" / AVA_SPEECH_FILENAMES

    # Download annotations
    logger.info("=" * 80)
    logger.info("Step 1: Downloading AVA-Speech annotations")
    logger.info("=" * 80)
    list_path = download_annotations(out_dir, session)

    # Download videos
    logger.info("=" * 80)
    logger.info("Step 2: Downloading AVA-Speech videos")
    logger.info("=" * 80)
    if not list_path.exists():
        logger.info(f"Video list not found at {list_path}. Attempting to download...")
        ensure_dir(list_path.parent)
        ok = download_file(
            S3_ANN_BASE + AVA_SPEECH_FILENAMES,
            list_path,
            session,
            desc="file_names.txt",
        )
        if not ok:
            logger.error("Failed to get video list (ava_speech_file_names_v1.txt)")
            sys.exit(3)

    download_videos(out_dir, list_path, max_videos, workers, prototype)

    # Extract audio from videos
    logger.info("=" * 80)
    logger.info("Step 3: Extracting audio from videos")
    logger.info("=" * 80)
    videos_dir = out_dir / "videos" / "trainval"
    audio_dir = out_dir / "audio"
    extract_audio_from_videos(videos_dir, audio_dir)

    logger.info("=" * 80)
    logger.info("✅ AVA-Speech download complete!")
    logger.info("=" * 80)

    # Summary
    video_count = len(list(videos_dir.glob("*.mp4"))) + len(list(videos_dir.glob("*.mkv")))
    total_size = sum(f.stat().st_size for f in videos_dir.glob("*.mp4")) + sum(
        f.stat().st_size for f in videos_dir.glob("*.mkv")
    )
    audio_count = len(list(audio_dir.glob("*.wav")))

    logger.info(f"Videos downloaded: {video_count}")
    logger.info(f"Total video size: {human_size(total_size)}")
    logger.info(f"Video location: {videos_dir}")
    logger.info(f"Audio files: {audio_count}")
    logger.info(f"Audio location: {audio_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download AVA-Speech dataset (annotations + videos)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ava-speech"],
        help="Datasets to download (default: ava-speech)",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Download full dataset even if PROTOTYPE_MODE=true",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum videos to download (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download workers (default: 8)",
    )

    args = parser.parse_args()

    if "ava-speech" not in args.datasets:
        logger.error("This script only supports 'ava-speech' dataset")
        sys.exit(1)

    # Determine prototype mode
    prototype = PROTOTYPE_MODE and not args.force_full

    if prototype:
        logger.info(f"PROTOTYPE_MODE: Will download only {PROTOTYPE_SAMPLES} videos")
    else:
        logger.info("FULL MODE: Will download complete dataset")

    # Get data root from config
    data_root = Path(CONFIG["data"]["root"])

    download_ava_speech(
        data_root=data_root,
        prototype=prototype,
        max_videos=args.max,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
