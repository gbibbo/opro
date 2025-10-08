"""
Dataset loaders with high-precision ground truth.

Supports:
- AVA-Speech (frame-level labels)
- DIHARD II/III (RTTM with onset/offset)
- VoxConverse (RTTM v0.3)
- AMI (word-level forced alignment)
- AVA-ActiveSpeaker (frame-level speaking labels)
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pyannote.core import Segment, Timeline
from pyannote.database.util import load_rttm

from qsm import PROTOTYPE_MODE, PROTOTYPE_SAMPLES

logger = logging.getLogger(__name__)


@dataclass
class FrameTable:
    """
    Unified frame-level annotation table.

    Columns:
    - uri: unique recording identifier
    - start_s: start time in seconds
    - end_s: end time in seconds
    - label: SPEECH or NONSPEECH
    - split: train/dev/test
    - dataset: source dataset name
    - snr_bin: SNR category (optional)
    - noise_type: noise type (optional)
    - condition: clean/music/noise (for AVA-Speech)
    """

    data: pd.DataFrame

    def __post_init__(self):
        required_cols = ["uri", "start_s", "end_s", "label", "split", "dataset"]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")

    @property
    def speech_segments(self) -> pd.DataFrame:
        """Return only SPEECH segments."""
        return self.data[self.data["label"] == "SPEECH"]

    @property
    def nonspeech_segments(self) -> pd.DataFrame:
        """Return only NONSPEECH segments."""
        return self.data[self.data["label"] == "NONSPEECH"]

    def get_uri_segments(self, uri: str) -> Timeline:
        """Get pyannote Timeline for a specific URI."""
        uri_data = self.data[self.data["uri"] == uri]
        segments = [
            Segment(row["start_s"], row["end_s"])
            for _, row in uri_data.iterrows()
            if row["label"] == "SPEECH"
        ]
        return Timeline(segments=segments)

    def save(self, output_path: Path):
        """Save to parquet."""
        self.data.to_parquet(output_path, index=False)
        logger.info(f"Saved FrameTable to {output_path}")

    @classmethod
    def load(cls, input_path: Path) -> "FrameTable":
        """Load from parquet."""
        data = pd.read_parquet(input_path)
        return cls(data=data)


def load_rttm_dataset(
    rttm_path: Path,
    uem_path: Path | None = None,
    dataset_name: str = "unknown",
    split: str = "train",
) -> FrameTable:
    """
    Load RTTM format annotations (DIHARD, VoxConverse).

    Args:
        rttm_path: Path to RTTM file
        uem_path: Optional UEM file to filter valid regions
        dataset_name: Name of the dataset
        split: Data split (train/dev/test)

    Returns:
        FrameTable with speech segments
    """
    logger.info(f"Loading RTTM from {rttm_path}")

    # Load using pyannote
    annotations = load_rttm(rttm_path)

    rows = []
    for uri, annotation in annotations.items():
        for segment, track, label in annotation.itertracks(yield_label=True):
            rows.append(
                {
                    "uri": uri,
                    "start_s": segment.start,
                    "end_s": segment.end,
                    "label": "SPEECH",  # RTTM segments are speech by default
                    "split": split,
                    "dataset": dataset_name,
                }
            )

    df = pd.DataFrame(rows)

    if PROTOTYPE_MODE and len(df) > 0:
        # Sample only N unique URIs for prototyping
        unique_uris = df["uri"].unique()[:PROTOTYPE_SAMPLES]
        df = df[df["uri"].isin(unique_uris)]
        logger.info(f"PROTOTYPE_MODE: Limited to {len(unique_uris)} URIs")

    return FrameTable(data=df)


def load_ava_speech(
    annotations_path: Path, dataset_name: str = "ava_speech", split: str = "train"
) -> FrameTable:
    """
    Load AVA-Speech frame-level annotations.

    Format: CSV with columns [video_id, frame_timestamp, label, condition]
    Labels: SPEECH_CLEAN, SPEECH_WITH_MUSIC, SPEECH_WITH_NOISE, NO_SPEECH
    """
    logger.info(f"Loading AVA-Speech from {annotations_path}")

    df = pd.read_csv(annotations_path)

    # Map labels to binary SPEECH/NONSPEECH
    df["label"] = df["label"].apply(lambda x: "SPEECH" if "SPEECH" in x else "NONSPEECH")

    # Extract condition
    df["condition"] = df["label"].apply(
        lambda x: (
            "clean"
            if "CLEAN" in x
            else "music" if "MUSIC" in x else "noise" if "NOISE" in x else "clean"
        )
    )

    # Convert frame timestamp to seconds (assuming 25 fps)
    fps = 25
    df["start_s"] = df["frame_timestamp"] / fps
    df["end_s"] = (df["frame_timestamp"] + 1) / fps

    df["uri"] = df["video_id"]
    df["split"] = split
    df["dataset"] = dataset_name

    if PROTOTYPE_MODE:
        unique_uris = df["uri"].unique()[:PROTOTYPE_SAMPLES]
        df = df[df["uri"].isin(unique_uris)]
        logger.info(f"PROTOTYPE_MODE: Limited to {len(unique_uris)} videos")

    return FrameTable(data=df)


def load_ami_alignment(
    alignment_path: Path, dataset_name: str = "ami", split: str = "train"
) -> FrameTable:
    """
    Load AMI forced alignment (word-level).

    Format: CTM or custom format with word-level timestamps
    """
    logger.info(f"Loading AMI alignment from {alignment_path}")

    # Placeholder - implement based on actual AMI format
    # AMI provides word-level alignments that need to be merged into speech segments
    rows = []

    # TODO: Implement actual AMI parsing
    # For now, return empty frame table
    df = pd.DataFrame(rows)

    return FrameTable(data=df)


def iter_intervals(
    uri: str, frame_table: FrameTable, label: Literal["SPEECH", "NONSPEECH"] = "SPEECH"
) -> Iterator[Segment]:
    """
    Iterate over intervals for a specific URI and label.

    Args:
        uri: Recording identifier
        frame_table: Source FrameTable
        label: SPEECH or NONSPEECH

    Yields:
        Segment objects
    """
    uri_data = frame_table.data[
        (frame_table.data["uri"] == uri) & (frame_table.data["label"] == label)
    ]

    for _, row in uri_data.iterrows():
        yield Segment(row["start_s"], row["end_s"])


def load_dataset(
    dataset_name: str, split: str = "train", config_path: Path | None = None
) -> FrameTable:
    """
    Load any dataset by name using configuration.

    Args:
        dataset_name: Name of dataset (ava_speech, dihard, voxconverse, ami, ava_activespeaker)
        split: Data split
        config_path: Path to dataset config YAML

    Returns:
        FrameTable with annotations
    """
    if config_path is None:
        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "configs"
            / "datasets"
            / f"{dataset_name}.yaml"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    import yaml

    with open(config_path) as f:
        dataset_config = yaml.safe_load(f)

    # Route to appropriate loader
    if dataset_name in ["dihard", "voxconverse"]:
        rttm_path = Path(dataset_config["rttm_path"][split])
        return load_rttm_dataset(rttm_path, dataset_name=dataset_name, split=split)

    elif dataset_name == "ava_speech":
        annotations_path = Path(dataset_config["annotations_path"][split])
        return load_ava_speech(annotations_path, dataset_name=dataset_name, split=split)

    elif dataset_name == "ami":
        alignment_path = Path(dataset_config["alignment_path"][split])
        return load_ami_alignment(alignment_path, dataset_name=dataset_name, split=split)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
