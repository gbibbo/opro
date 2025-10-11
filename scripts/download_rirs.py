#!/usr/bin/env python3
"""
Download and setup OpenSLR SLR28 RIRS_NOISES dataset.

Dataset: https://www.openslr.org/28/
Contains simulated and real room impulse responses (RIRs) with various T60 values.

Usage:
    python scripts/download_rirs.py --output_dir data/external/RIRS_NOISES
"""

import argparse
import logging
import subprocess
from pathlib import Path
import tarfile
import urllib.request
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DATASET_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
DATASET_NAME = "rirs_noises.zip"


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=Path, default=Path("data/external/RIRS_NOISES"), help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    zip_path = output_dir / DATASET_NAME
    if zip_path.exists():
        logger.info(f"Dataset already downloaded: {zip_path}")
    else:
        logger.info(f"Downloading {DATASET_URL} to {zip_path}")
        download_url(DATASET_URL, zip_path)
        logger.info("Download complete")

    # Extract
    logger.info(f"Extracting {zip_path}")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    logger.info("Extraction complete")

    # Verify structure
    rir_root = output_dir / "RIRS_NOISES"
    if not rir_root.exists():
        logger.error(f"Expected RIRS_NOISES directory not found in {output_dir}")
        return

    sim_dir = rir_root / "simulated_rirs"
    real_dir = rir_root / "real_rirs_isotropic_noises"

    n_sim = len(list(sim_dir.rglob("*.wav"))) if sim_dir.exists() else 0
    n_real = len(list(real_dir.glob("*.wav"))) if real_dir.exists() else 0

    logger.info(f"Found {n_sim} simulated RIRs in {sim_dir}")
    logger.info(f"Found {n_real} real RIRs in {real_dir}")
    logger.info(f"Dataset ready at {rir_root}")


if __name__ == "__main__":
    main()
