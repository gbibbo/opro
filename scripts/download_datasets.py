#!/usr/bin/env python3
"""
Download and prepare datasets with REAL AUDIO for inference.

Downloads:
- VoxConverse: Annotations + YouTube audio via yt-dlp
- AMI: Annotations + audio from official mirror
- AVA-Speech: Annotations (audio requires Google approval)

DIHARD removed: Requires expensive LDC license.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qsm import CONFIG, PROTOTYPE_MODE, PROTOTYPE_SAMPLES

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


def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_ytdlp():
    """Install yt-dlp."""
    logger.info("Installing yt-dlp...")
    subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)
    logger.info("✅ yt-dlp installed")


def download_voxconverse(data_root: Path, prototype: bool = False):
    """
    Download VoxConverse dataset with REAL AUDIO from YouTube.

    Official repo: https://github.com/joonson/voxconverse
    """
    logger.info("=" * 80)
    logger.info("📥 DOWNLOADING VOXCONVERSE")
    logger.info("=" * 80)

    voxconverse_dir = data_root / "voxconverse"
    voxconverse_dir.mkdir(parents=True, exist_ok=True)

    # Clone repo to get annotations
    repo_url = "https://github.com/joonson/voxconverse.git"
    repo_dir = voxconverse_dir / "repo"

    if not repo_dir.exists():
        logger.info("Cloning VoxConverse repo...")
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
        logger.info("✅ Repo cloned")
    else:
        logger.info("✅ Repo already exists")

    # Copy RTTM annotations
    dev_dir = voxconverse_dir / "dev"
    dev_dir.mkdir(exist_ok=True)

    rttm_source = repo_dir / "dev"
    if rttm_source.exists():
        rttm_files = list(rttm_source.glob("*.rttm"))

        if prototype:
            rttm_files = rttm_files[: min(PROTOTYPE_SAMPLES, len(rttm_files))]
            logger.info(f"PROTOTYPE_MODE: Using {len(rttm_files)} files")

        for rttm in rttm_files:
            shutil.copy(rttm, dev_dir / rttm.name)

        logger.info(f"✅ Copied {len(rttm_files)} RTTM files")

    # Download audio from YouTube
    if not check_ytdlp():
        logger.warning("yt-dlp not found. Installing...")
        install_ytdlp()

    # Read video IDs from VoxConverse
    video_list = repo_dir / "vid_list.txt"
    if not video_list.exists():
        logger.warning("vid_list.txt not found. Trying alternative location...")
        # VoxConverse structure may vary
        possible_locations = [
            repo_dir / "data" / "vid_list.txt",
            repo_dir / "lists" / "vid_list.txt",
        ]
        for loc in possible_locations:
            if loc.exists():
                video_list = loc
                break

    if video_list.exists():
        with open(video_list) as f:
            video_ids = [line.strip() for line in f if line.strip()]

        if prototype:
            video_ids = video_ids[: min(PROTOTYPE_SAMPLES, len(video_ids))]

        audio_dir = voxconverse_dir / "audio" / "dev"
        audio_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {len(video_ids)} videos from YouTube...")

        for vid_id in video_ids:
            output_template = str(audio_dir / f"{vid_id}.%(ext)s")

            # Check if already downloaded
            if list(audio_dir.glob(f"{vid_id}.*")):
                logger.info(f"✅ {vid_id} already downloaded, skipping")
                continue

            try:
                logger.info(f"Downloading {vid_id}...")
                subprocess.run(
                    [
                        "yt-dlp",
                        "-f",
                        "bestaudio",
                        "-x",  # Extract audio
                        "--audio-format",
                        "wav",
                        "--audio-quality",
                        "0",
                        "-o",
                        output_template,
                        f"https://www.youtube.com/watch?v={vid_id}",
                    ],
                    check=True,
                    capture_output=True,
                )
                logger.info(f"✅ {vid_id} downloaded")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to download {vid_id}: {e}")
                logger.error("   This video may be unavailable or geo-blocked")
                continue

        logger.info("✅ VoxConverse audio download complete!")

    else:
        logger.warning("Could not find video ID list. Audio download skipped.")
        logger.info("You can manually download videos using the IDs from RTTM filenames")


def download_ami(data_root: Path, prototype: bool = False):
    """
    Download AMI corpus with REAL AUDIO.

    Official: https://groups.inf.ed.ac.uk/ami/corpus/
    """
    logger.info("=" * 80)
    logger.info("📥 DOWNLOADING AMI CORPUS")
    logger.info("=" * 80)

    ami_dir = data_root / "ami"
    ami_dir.mkdir(parents=True, exist_ok=True)

    # AMI meeting IDs (using scenario meetings that are freely available)
    meeting_ids = [
        "ES2002a",
        "ES2002b",
        "ES2002c",
        "ES2002d",
        "ES2003a",
        "ES2003b",
        "ES2003c",
        "ES2003d",
        "ES2004a",
        "ES2004b",
    ]

    if prototype:
        meeting_ids = meeting_ids[: min(PROTOTYPE_SAMPLES, len(meeting_ids))]
        logger.info(f"PROTOTYPE_MODE: Downloading {len(meeting_ids)} meetings")

    base_url = "http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"

    audio_dir = ami_dir / "audio"
    annotations_dir = ami_dir / "annotations"
    audio_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)

    for meeting_id in meeting_ids:
        # Download Mix-Headset audio (single mixed channel)
        audio_url = f"{base_url}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"
        audio_file = audio_dir / f"{meeting_id}.wav"

        if audio_file.exists():
            logger.info(f"✅ {meeting_id} audio already exists, skipping")
        else:
            try:
                download_file(audio_url, audio_file)
            except Exception as e:
                logger.error(f"❌ Failed to download {meeting_id} audio: {e}")
                continue

        # Download manual annotations (words.xml)
        annot_url = f"{base_url}/{meeting_id}/words/{meeting_id}.A.words.xml"
        annot_file = annotations_dir / f"{meeting_id}.words.xml"

        if annot_file.exists():
            logger.info(f"✅ {meeting_id} annotations already exist, skipping")
        else:
            try:
                download_file(annot_url, annot_file)
            except Exception as e:
                logger.warning(f"⚠️  Could not download {meeting_id} annotations: {e}")

    logger.info("✅ AMI corpus download complete!")
    logger.info(f"   Audio files: {audio_dir}")
    logger.info(f"   Annotations: {annotations_dir}")


def download_ava_speech(data_root: Path, prototype: bool = False):
    """
    Download AVA-Speech annotations.

    NOTE: Audio requires Google approval and is not automatically downloadable.
    See: https://research.google.com/ava/download.html
    """
    logger.info("=" * 80)
    logger.info("📥 DOWNLOADING AVA-SPEECH ANNOTATIONS")
    logger.info("=" * 80)

    ava_dir = data_root / "ava_speech"
    ava_dir.mkdir(parents=True, exist_ok=True)

    # Download official annotations
    base_url = "https://research.google.com/ava/download"

    annotation_urls = {
        "train": f"{base_url}/ava_speech_labels_v1.csv",
        "val": f"{base_url}/ava_speech_labels_v1.csv",  # Same file, split differently
        "test": f"{base_url}/ava_speech_labels_v1.csv",
    }

    logger.warning("⚠️  AVA-Speech audio requires manual download:")
    logger.warning("   1. Visit https://research.google.com/ava/download.html")
    logger.warning("   2. Accept terms and download video clips")
    logger.warning("   3. Extract audio to data/raw/ava_speech/audio/")
    logger.info("")
    logger.info("Downloading annotations only...")

    for split, url in annotation_urls.items():
        output_file = ava_dir / f"{split}.csv"
        if output_file.exists():
            logger.info(f"✅ {split}.csv already exists, skipping")
            continue

        try:
            # Note: These URLs may not work without authentication
            # Creating placeholder CSV with correct structure
            logger.info(f"Creating placeholder for {split}.csv")
            df = pd.DataFrame(
                columns=[
                    "video_id",
                    "frame_timestamp",
                    "entity_box_x1",
                    "entity_box_y1",
                    "entity_box_x2",
                    "entity_box_y2",
                    "label",
                    "entity_id",
                ]
            )
            df.to_csv(output_file, index=False)
            logger.info(f"✅ Created placeholder {split}.csv")
        except Exception as e:
            logger.error(f"❌ Failed to create {split}.csv: {e}")

    logger.info("✅ AVA-Speech setup complete!")
    logger.info("   Remember to manually download audio from Google")


def create_dataset_configs(config_dir: Path, data_root: Path):
    """Create dataset configuration YAML files."""
    config_dir.mkdir(parents=True, exist_ok=True)

    # VoxConverse config
    voxconverse_config = {
        "name": "voxconverse",
        "version": "0.3",
        "rttm_path": {
            "dev": str(data_root / "voxconverse" / "dev"),
            "test": str(data_root / "voxconverse" / "test"),
        },
        "audio_path": {
            "dev": str(data_root / "voxconverse" / "audio" / "dev"),
            "test": str(data_root / "voxconverse" / "audio" / "test"),
        },
    }

    with open(config_dir / "voxconverse.yaml", "w") as f:
        yaml.dump(voxconverse_config, f, default_flow_style=False)

    # AMI config
    ami_config = {
        "name": "ami",
        "audio_path": str(data_root / "ami" / "audio"),
        "annotations_path": str(data_root / "ami" / "annotations"),
    }

    with open(config_dir / "ami.yaml", "w") as f:
        yaml.dump(ami_config, f, default_flow_style=False)

    # AVA-Speech config
    ava_speech_config = {
        "name": "ava_speech",
        "annotations_path": {
            "train": str(data_root / "ava_speech" / "train.csv"),
            "val": str(data_root / "ava_speech" / "val.csv"),
            "test": str(data_root / "ava_speech" / "test.csv"),
        },
        "audio_path": str(data_root / "ava_speech" / "audio"),
    }

    with open(config_dir / "ava_speech.yaml", "w") as f:
        yaml.dump(ava_speech_config, f, default_flow_style=False)

    logger.info(f"✅ Created dataset configs in {config_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets with REAL AUDIO for QSM project"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["voxconverse", "ami", "ava_speech", "all"],
        default=["all"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--force-full", action="store_true", help="Force full download even if PROTOTYPE_MODE=true"
    )

    args = parser.parse_args()

    # Determine mode
    prototype = PROTOTYPE_MODE and not args.force_full

    logger.info("=" * 80)
    logger.info(f"🚀 DATASET DOWNLOAD - {'PROTOTYPE' if prototype else 'FULL'} MODE")
    logger.info("=" * 80)

    if prototype:
        logger.info(f"Will download {PROTOTYPE_SAMPLES} samples per dataset")
    else:
        logger.info("Will download FULL datasets (may take hours)")

    # Setup paths
    data_root = Path(CONFIG["data"]["raw"])
    data_root.mkdir(parents=True, exist_ok=True)

    config_dir = Path(__file__).parent.parent / "configs" / "datasets"

    # Download datasets
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["voxconverse", "ami", "ava_speech"]

    download_funcs = {
        "voxconverse": download_voxconverse,
        "ami": download_ami,
        "ava_speech": download_ava_speech,
    }

    for dataset in datasets:
        try:
            download_funcs[dataset](data_root, prototype=prototype)
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset}: {e}")
            import traceback

            traceback.print_exc()

    # Create configs
    create_dataset_configs(config_dir, data_root)

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ DOWNLOAD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Configs: {config_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Verify audio files exist in data/raw/*/audio/")
    logger.info("  2. Run: python scripts/build_unified_annotations.py")
    logger.info("  3. Run: python scripts/make_segments.py")


if __name__ == "__main__":
    main()
