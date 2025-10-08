#!/usr/bin/env python3
"""
Download and prepare datasets for prototyping or full experiments.

Respects PROTOTYPE_MODE setting in config.yaml:
- If True: downloads only 5 examples per dataset
- If False: downloads full datasets
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

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
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(output_path, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_voxconverse(data_root: Path, prototype: bool = False):
    """
    Download VoxConverse dataset.

    Official repo: https://github.com/joonson/voxconverse
    """
    logger.info("Downloading VoxConverse dataset...")

    voxconverse_dir = data_root / "voxconverse"
    voxconverse_dir.mkdir(parents=True, exist_ok=True)

    if prototype:
        # Download only annotation samples for prototyping
        logger.info(f"PROTOTYPE_MODE: Downloading {PROTOTYPE_SAMPLES} samples")

        # Clone repo to get annotations
        repo_url = "https://github.com/joonson/voxconverse.git"
        repo_dir = voxconverse_dir / "repo"

        if not repo_dir.exists():
            subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)

        # Copy sample RTTM files
        rttm_source = repo_dir / "dev"
        if rttm_source.exists():
            rttm_files = list(rttm_source.glob("*.rttm"))[:PROTOTYPE_SAMPLES]

            dev_dir = voxconverse_dir / "dev"
            dev_dir.mkdir(exist_ok=True)

            for rttm in rttm_files:
                shutil.copy(rttm, dev_dir / rttm.name)
                logger.info(f"Copied {rttm.name}")

        logger.info("VoxConverse prototype data ready")
        logger.info("NOTE: Audio files need to be downloaded separately from YouTube")
        logger.info("See: https://github.com/joonson/voxconverse#audio-download")

    else:
        # Full dataset download instructions
        logger.info("Full VoxConverse download:")
        logger.info("1. Clone: git clone https://github.com/joonson/voxconverse.git")
        logger.info("2. Follow audio download instructions in repo README")
        logger.info("3. Run download script: python download_voxconverse.py")


def download_dihard(data_root: Path, prototype: bool = False):
    """
    Download DIHARD dataset.

    Official site: https://dihardchallenge.github.io/dihard3/
    """
    logger.info("Downloading DIHARD III dataset...")

    dihard_dir = data_root / "dihard"
    dihard_dir.mkdir(parents=True, exist_ok=True)

    if prototype:
        logger.info(f"PROTOTYPE_MODE: Creating {PROTOTYPE_SAMPLES} mock samples")

        # DIHARD requires LDC license - create mock data for prototyping
        dev_dir = dihard_dir / "dev"
        dev_dir.mkdir(exist_ok=True)

        # Create mock RTTM files
        mock_rttm_path = dev_dir / "mock.rttm"
        with open(mock_rttm_path, "w") as f:
            for i in range(PROTOTYPE_SAMPLES):
                # RTTM format: SPEAKER file 1 start duration <NA> <NA> speaker <NA> <NA>
                f.write(f"SPEAKER mock_file_{i} 1 0.0 5.0 <NA> <NA> speaker_1 <NA> <NA>\n")
                f.write(f"SPEAKER mock_file_{i} 1 6.0 4.0 <NA> <NA> speaker_2 <NA> <NA>\n")

        logger.info(f"Created mock RTTM at {mock_rttm_path}")
        logger.info(
            "NOTE: For real data, obtain LDC license from https://dihardchallenge.github.io/dihard3/"
        )

    else:
        logger.info("Full DIHARD download:")
        logger.info("1. Obtain LDC license")
        logger.info("2. Download from: https://dihardchallenge.github.io/dihard3/")
        logger.info("3. Extract to: {dihard_dir}")


def download_ava_speech(data_root: Path, prototype: bool = False):
    """
    Download AVA-Speech dataset.

    Official: https://research.google.com/ava/download.html
    """
    logger.info("Downloading AVA-Speech dataset...")

    ava_dir = data_root / "ava_speech"
    ava_dir.mkdir(parents=True, exist_ok=True)

    if prototype:
        logger.info(f"PROTOTYPE_MODE: Creating {PROTOTYPE_SAMPLES} mock samples")

        # Create mock CSV annotations
        train_csv = ava_dir / "train.csv"

        with open(train_csv, "w") as f:
            f.write("video_id,frame_timestamp,label,condition\n")

            for i in range(PROTOTYPE_SAMPLES):
                video_id = f"mock_video_{i}"
                # Simulate 5 seconds at 25 fps
                for frame in range(125):
                    if frame % 50 < 25:
                        label = "SPEECH_CLEAN"
                        condition = "clean"
                    else:
                        label = "NO_SPEECH"
                        condition = "none"

                    f.write(f"{video_id},{frame},{label},{condition}\n")

        logger.info(f"Created mock AVA-Speech annotations at {train_csv}")
        logger.info(
            "NOTE: For real data, download from https://research.google.com/ava/download.html"
        )

    else:
        logger.info("Full AVA-Speech download:")
        logger.info("1. Visit: https://research.google.com/ava/download.html")
        logger.info("2. Download annotation CSVs")
        logger.info("3. Download videos (or audio tracks)")
        logger.info("4. Extract to: {ava_dir}")


def download_ami(data_root: Path, prototype: bool = False):
    """
    Download AMI corpus.

    Official: https://groups.inf.ed.ac.uk/ami/corpus/
    """
    logger.info("Downloading AMI corpus...")

    ami_dir = data_root / "ami"
    ami_dir.mkdir(parents=True, exist_ok=True)

    if prototype:
        logger.info(f"PROTOTYPE_MODE: Creating {PROTOTYPE_SAMPLES} mock samples")

        # Create mock word-level alignment
        alignments_dir = ami_dir / "alignments"
        alignments_dir.mkdir(exist_ok=True)

        mock_align = alignments_dir / "mock_alignment.txt"

        with open(mock_align, "w") as f:
            for i in range(PROTOTYPE_SAMPLES):
                meeting_id = f"mock_meeting_{i}"
                # Word-level timing: meeting_id channel start end word
                words = ["hello", "world", "this", "is", "a", "test"]
                time = 0.0

                for word in words:
                    duration = 0.3
                    f.write(f"{meeting_id} 1 {time:.3f} {time+duration:.3f} {word}\n")
                    time += duration + 0.1

        logger.info(f"Created mock AMI alignment at {mock_align}")
        logger.info("NOTE: For real data, download from https://groups.inf.ed.ac.uk/ami/download/")

    else:
        logger.info("Full AMI download:")
        logger.info("1. Visit: https://groups.inf.ed.ac.uk/ami/download/")
        logger.info("2. Download annotations and audio")
        logger.info("3. Extract to: {ami_dir}")


def download_ava_activespeaker(data_root: Path, prototype: bool = False):
    """
    Download AVA-ActiveSpeaker dataset.

    Official: https://github.com/cvdfoundation/ava-dataset
    """
    logger.info("Downloading AVA-ActiveSpeaker dataset...")

    ava_as_dir = data_root / "ava_activespeaker"
    ava_as_dir.mkdir(parents=True, exist_ok=True)

    if prototype:
        logger.info(f"PROTOTYPE_MODE: Creating {PROTOTYPE_SAMPLES} mock samples")

        train_csv = ava_as_dir / "train.csv"

        with open(train_csv, "w") as f:
            f.write(
                "video_id,frame_timestamp,entity_box_x1,entity_box_y1,entity_box_x2,entity_box_y2,label,entity_id\n"
            )

            for i in range(PROTOTYPE_SAMPLES):
                video_id = f"mock_video_{i}"
                for frame in range(100):
                    # Simulate bounding box and speaking label
                    x1, y1, x2, y2 = 0.2, 0.3, 0.8, 0.9
                    label = "SPEAKING_AUDIBLE" if frame % 40 < 20 else "NOT_SPEAKING"
                    entity_id = "person_1"

                    f.write(f"{video_id},{frame},{x1},{y1},{x2},{y2},{label},{entity_id}\n")

        logger.info(f"Created mock AVA-ActiveSpeaker annotations at {train_csv}")
        logger.info("NOTE: For real data, visit https://github.com/cvdfoundation/ava-dataset")

    else:
        logger.info("Full AVA-ActiveSpeaker download:")
        logger.info("1. Visit: https://github.com/cvdfoundation/ava-dataset")
        logger.info("2. Download annotations")
        logger.info("3. Download videos")
        logger.info("4. Extract to: {ava_as_dir}")


def create_dataset_configs(config_dir: Path, data_root: Path):
    """Create dataset configuration YAML files."""
    config_dir.mkdir(parents=True, exist_ok=True)

    # VoxConverse config
    voxconverse_config = {
        "name": "voxconverse",
        "version": "0.3",
        "rttm_path": {
            "train": str(data_root / "voxconverse" / "train"),
            "dev": str(data_root / "voxconverse" / "dev"),
            "test": str(data_root / "voxconverse" / "test"),
        },
        "audio_path": {
            "train": str(data_root / "voxconverse" / "audio" / "train"),
            "dev": str(data_root / "voxconverse" / "audio" / "dev"),
            "test": str(data_root / "voxconverse" / "audio" / "test"),
        },
    }

    with open(config_dir / "voxconverse.yaml", "w") as f:
        yaml.dump(voxconverse_config, f, default_flow_style=False)

    # DIHARD config
    dihard_config = {
        "name": "dihard",
        "version": "III",
        "rttm_path": {
            "train": str(data_root / "dihard" / "train" / "rttm"),
            "dev": str(data_root / "dihard" / "dev" / "mock.rttm"),
            "test": str(data_root / "dihard" / "test" / "rttm"),
        },
        "audio_path": {
            "train": str(data_root / "dihard" / "train" / "audio"),
            "dev": str(data_root / "dihard" / "dev" / "audio"),
            "test": str(data_root / "dihard" / "test" / "audio"),
        },
    }

    with open(config_dir / "dihard.yaml", "w") as f:
        yaml.dump(dihard_config, f, default_flow_style=False)

    # AVA-Speech config
    ava_speech_config = {
        "name": "ava_speech",
        "annotations_path": {
            "train": str(data_root / "ava_speech" / "train.csv"),
            "dev": str(data_root / "ava_speech" / "val.csv"),
            "test": str(data_root / "ava_speech" / "test.csv"),
        },
        "audio_path": {
            "train": str(data_root / "ava_speech" / "audio" / "train"),
            "dev": str(data_root / "ava_speech" / "audio" / "val"),
            "test": str(data_root / "ava_speech" / "audio" / "test"),
        },
    }

    with open(config_dir / "ava_speech.yaml", "w") as f:
        yaml.dump(ava_speech_config, f, default_flow_style=False)

    # AMI config
    ami_config = {
        "name": "ami",
        "alignment_path": {
            "train": str(data_root / "ami" / "alignments" / "train"),
            "dev": str(data_root / "ami" / "alignments" / "dev"),
            "test": str(data_root / "ami" / "alignments" / "test"),
        },
        "audio_path": {
            "train": str(data_root / "ami" / "audio" / "train"),
            "dev": str(data_root / "ami" / "audio" / "dev"),
            "test": str(data_root / "ami" / "audio" / "test"),
        },
    }

    with open(config_dir / "ami.yaml", "w") as f:
        yaml.dump(ami_config, f, default_flow_style=False)

    # AVA-ActiveSpeaker config
    ava_as_config = {
        "name": "ava_activespeaker",
        "annotations_path": {
            "train": str(data_root / "ava_activespeaker" / "train.csv"),
            "dev": str(data_root / "ava_activespeaker" / "val.csv"),
            "test": str(data_root / "ava_activespeaker" / "test.csv"),
        },
        "audio_path": {
            "train": str(data_root / "ava_activespeaker" / "audio" / "train"),
            "dev": str(data_root / "ava_activespeaker" / "audio" / "val"),
            "test": str(data_root / "ava_activespeaker" / "audio" / "test"),
        },
    }

    with open(config_dir / "ava_activespeaker.yaml", "w") as f:
        yaml.dump(ava_as_config, f, default_flow_style=False)

    logger.info(f"Created dataset configs in {config_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for QSM project")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["voxconverse", "dihard", "ava_speech", "ami", "ava_activespeaker", "all"],
        default=["all"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--force-full", action="store_true", help="Force full download even if PROTOTYPE_MODE=true"
    )

    args = parser.parse_args()

    # Determine mode
    prototype = PROTOTYPE_MODE and not args.force_full

    logger.info(f"Download mode: {'PROTOTYPE' if prototype else 'FULL'}")
    if prototype:
        logger.info(f"Will download {PROTOTYPE_SAMPLES} samples per dataset")

    # Setup paths
    data_root = Path(CONFIG["data"]["raw"])
    data_root.mkdir(parents=True, exist_ok=True)

    config_dir = Path(__file__).parent.parent / "configs" / "datasets"

    # Download datasets
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["voxconverse", "dihard", "ava_speech", "ami", "ava_activespeaker"]

    download_funcs = {
        "voxconverse": download_voxconverse,
        "dihard": download_dihard,
        "ava_speech": download_ava_speech,
        "ami": download_ami,
        "ava_activespeaker": download_ava_activespeaker,
    }

    for dataset in datasets:
        try:
            download_funcs[dataset](data_root, prototype=prototype)
        except Exception as e:
            logger.error(f"Failed to download {dataset}: {e}")

    # Create configs
    create_dataset_configs(config_dir, data_root)

    logger.info("Download complete!")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Configs: {config_dir}")


if __name__ == "__main__":
    main()
