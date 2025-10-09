#!/usr/bin/env python3
"""
Extract audio from AVA-Speech videos without ffmpeg.
Uses moviepy which can handle video decoding.
"""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import CONFIG, PROTOTYPE_MODE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_audio_with_ffmpeg(videos_dir: Path, audio_dir: Path):
    """Extract audio from video files using imageio_ffmpeg."""
    try:
        import imageio_ffmpeg as ffmpeg
    except ImportError:
        logger.error("imageio_ffmpeg not found. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=True)
        import imageio_ffmpeg as ffmpeg

    audio_dir.mkdir(parents=True, exist_ok=True)

    # Get all video files
    video_files = (
        list(videos_dir.glob("*.mp4"))
        + list(videos_dir.glob("*.mkv"))
        + list(videos_dir.glob("*.webm"))
    )

    if not video_files:
        logger.warning(f"No video files found in {videos_dir}")
        return

    logger.info(f"Found {len(video_files)} videos to process")

    # Get ffmpeg executable
    ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
    logger.info(f"Using ffmpeg: {ffmpeg_exe}")

    import subprocess

    for video_file in tqdm(video_files, desc="Extracting audio", unit="file"):
        audio_file = audio_dir / f"{video_file.stem}.wav"

        # Skip if audio already exists
        if audio_file.exists():
            logger.info(f"✅ Audio already exists: {audio_file.name}")
            continue

        try:
            logger.info(f"Processing {video_file.name}...")

            # Use ffmpeg to extract audio
            cmd = [
                ffmpeg_exe,
                "-i", str(video_file),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # 16-bit PCM
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                str(audio_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"❌ ffmpeg error: {result.stderr}")
                continue

            logger.info(f"✅ Extracted: {audio_file.name}")

        except subprocess.TimeoutExpired:
            logger.error(f"❌ Timeout extracting audio from {video_file.name}")
            if audio_file.exists():
                audio_file.unlink()
        except Exception as e:
            logger.error(f"❌ Failed to extract audio from {video_file.name}: {e}")
            if audio_file.exists():
                audio_file.unlink()

    # Summary
    audio_files = list(audio_dir.glob("*.wav"))
    total_size = sum(f.stat().st_size for f in audio_files)

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ AUDIO EXTRACTION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Audio files extracted: {len(audio_files)}")
    logger.info(f"Total size: {total_size / (1024**2):.1f} MB")
    logger.info(f"Location: {audio_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract audio from AVA-Speech videos")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=None,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None, help="Output directory for audio files"
    )

    args = parser.parse_args()

    # Use default paths if not specified
    data_root = Path(CONFIG["data"]["root"])
    videos_dir = args.videos_dir or data_root / "raw" / "ava-speech" / "videos" / "trainval"
    audio_dir = args.audio_dir or data_root / "raw" / "ava-speech" / "audio"

    if not videos_dir.exists():
        logger.error(f"Videos directory not found: {videos_dir}")
        sys.exit(1)

    logger.info(f"Videos directory: {videos_dir}")
    logger.info(f"Audio output directory: {audio_dir}")

    extract_audio_with_ffmpeg(videos_dir, audio_dir)


if __name__ == "__main__":
    main()
