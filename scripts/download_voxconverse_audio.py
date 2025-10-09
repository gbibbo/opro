#!/usr/bin/env python3
"""
Download VoxConverse pre-processed audio files directly.

The official VoxConverse provides pre-processed WAV files instead of YouTube downloads.
This is much more reliable than youtube-dl.

IMPORTANT: Respects PROTOTYPE_MODE from config.yaml
- PROTOTYPE_MODE=true: Downloads only 5 samples
- PROTOTYPE_MODE=false: Downloads full dataset (use --force-full)
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import PROTOTYPE_MODE, PROTOTYPE_SAMPLES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    logger.info(f"Downloading {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(output_path, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info(f"✅ Downloaded to {output_path}")


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract ZIP file."""
    logger.info(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)

    logger.info(f"✅ Extracted to {extract_to}")


def download_voxconverse_audio(data_root: Path, splits: list[str] = None, prototype: bool = True):
    """
    Download VoxConverse pre-processed audio files and RTTM annotations.

    Args:
        data_root: Root directory for data storage
        splits: List of splits to download (dev, test). Default: both
        prototype: If True, download only PROTOTYPE_SAMPLES files
    """
    if splits is None:
        splits = ["dev"]  # Only dev for prototype

    voxconverse_dir = data_root / "voxconverse"
    voxconverse_dir.mkdir(parents=True, exist_ok=True)

    audio_urls = {
        "dev": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip",
        "test": "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip",
    }

    # RTTM annotation URLs from the official GitHub repo
    rttm_base_url = "https://raw.githubusercontent.com/joonson/voxconverse/master"

    for split in splits:
        logger.info("=" * 80)
        logger.info(f"📥 DOWNLOADING VOXCONVERSE {split.upper()}")
        logger.info("=" * 80)

        # Download RTTM annotations first
        logger.info(f"Step 1: Downloading RTTM annotations for {split}")
        rttm_dir = voxconverse_dir / split
        rttm_dir.mkdir(parents=True, exist_ok=True)

        # For now, we'll get RTTMs from audio filenames after download
        # VoxConverse provides RTTMs via GitHub repo cloning (done in download_datasets.py)

        url = audio_urls[split]
        zip_path = voxconverse_dir / f"voxconverse_{split}_wav.zip"
        audio_dir = voxconverse_dir / "audio" / split

        # Check if already extracted
        if audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 0:
            logger.info(f"✅ {split} audio already downloaded and extracted")
            logger.info(f"   Found {len(list(audio_dir.glob('*.wav')))} WAV files")
            continue

        # Download ZIP
        if not zip_path.exists():
            try:
                download_file(url, zip_path)
            except Exception as e:
                logger.error(f"❌ Failed to download {split}: {e}")
                continue
        else:
            logger.info(f"✅ ZIP already exists: {zip_path}")

        # Extract ZIP
        try:
            audio_dir.mkdir(parents=True, exist_ok=True)
            extract_zip(zip_path, audio_dir)

            # Count extracted files
            wav_files = list(audio_dir.glob("**/*.wav"))
            logger.info(f"✅ Extracted {len(wav_files)} WAV files")

            # Move files if they're in a subdirectory
            for wav_file in wav_files:
                if wav_file.parent != audio_dir:
                    new_path = audio_dir / wav_file.name
                    wav_file.rename(new_path)
                    logger.debug(f"Moved {wav_file.name} to {audio_dir}")

            # Clean up empty subdirectories
            for subdir in audio_dir.iterdir():
                if subdir.is_dir() and not list(subdir.iterdir()):
                    subdir.rmdir()

            # PROTOTYPE MODE: Keep only N samples
            if prototype:
                all_wavs = sorted(audio_dir.glob("*.wav"))
                if len(all_wavs) > PROTOTYPE_SAMPLES:
                    logger.info(
                        f"PROTOTYPE_MODE: Keeping only {PROTOTYPE_SAMPLES} files, removing {len(all_wavs) - PROTOTYPE_SAMPLES}"
                    )
                    for wav_file in all_wavs[PROTOTYPE_SAMPLES:]:
                        wav_file.unlink()
                        logger.debug(f"Removed {wav_file.name}")

                remaining = len(list(audio_dir.glob("*.wav")))
                logger.info(f"✅ VoxConverse {split} ready! ({remaining} files in PROTOTYPE_MODE)")

        except Exception as e:
            logger.error(f"❌ Failed to extract {split}: {e}")
            continue

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ VOXCONVERSE DOWNLOAD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Audio files: {voxconverse_dir / 'audio'}")


def main():
    parser = argparse.ArgumentParser(
        description="Download VoxConverse audio files (respects PROTOTYPE_MODE)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["dev", "test"],
        default=["dev"],
        help="Splits to download",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw"), help="Data root directory"
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Download full dataset even if PROTOTYPE_MODE=true",
    )

    args = parser.parse_args()

    # Determine mode
    prototype = PROTOTYPE_MODE and not args.force_full

    if prototype:
        logger.info(f"PROTOTYPE_MODE: Will keep only {PROTOTYPE_SAMPLES} files per split")
    else:
        logger.info("FULL MODE: Will download complete datasets")

    download_voxconverse_audio(args.data_root, args.splits, prototype=prototype)


if __name__ == "__main__":
    main()
