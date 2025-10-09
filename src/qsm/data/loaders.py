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
    audio_dir: Path | None = None,
    dataset_name: str = "unknown",
    split: str = "train",
    include_nonspeech: bool = False,
) -> FrameTable:
    """
    Load RTTM format annotations (DIHARD, VoxConverse).

    Args:
        rttm_path: Path to RTTM file
        uem_path: Optional UEM file to filter valid regions
        audio_dir: Optional directory containing audio files to get durations
        dataset_name: Name of the dataset
        split: Data split (train/dev/test)
        include_nonspeech: If True, generate nonspeech segments from gaps

    Returns:
        FrameTable with speech (and optionally nonspeech) segments
    """
    logger.info(f"Loading RTTM from {rttm_path}")

    # Load using pyannote
    annotations = load_rttm(rttm_path)

    # Load UEM if provided
    uem_timeline = None
    if uem_path is not None and uem_path.exists():
        from pyannote.database.util import load_uem

        logger.info(f"Loading UEM from {uem_path}")
        uem_timeline = load_uem(uem_path)

    # Get audio durations if audio_dir provided
    audio_durations = {}
    if audio_dir is not None and include_nonspeech:
        import soundfile as sf

        for audio_file in audio_dir.glob("*.wav"):
            try:
                info = sf.info(audio_file)
                audio_durations[audio_file.stem] = info.duration
            except Exception as e:
                logger.warning(f"Failed to get duration for {audio_file.name}: {e}")

        logger.info(f"Loaded {len(audio_durations)} audio durations for nonspeech generation")

    rows = []
    for uri, annotation in annotations.items():
        # Get speech timeline
        speech_timeline = annotation.get_timeline()

        # Filter by UEM if provided
        if uem_timeline is not None and uri in uem_timeline:
            valid_regions = uem_timeline[uri]
            speech_timeline = speech_timeline.crop(valid_regions)

        # Add speech segments
        for segment in speech_timeline:
            rows.append(
                {
                    "uri": uri,
                    "start_s": segment.start,
                    "end_s": segment.end,
                    "label": "SPEECH",
                    "split": split,
                    "dataset": dataset_name,
                }
            )

        # Add nonspeech segments if requested
        if include_nonspeech:
            # Determine full extent
            if uem_timeline is not None and uri in uem_timeline:
                full_extent = uem_timeline[uri].extent()
                logger.debug(f"{uri}: Using UEM extent: {full_extent}")
            elif uri in audio_durations:
                # Use audio duration to get full extent
                from pyannote.core import Segment as PyannoteSegment

                full_extent = PyannoteSegment(0, audio_durations[uri])
                logger.debug(f"{uri}: Using audio duration extent: {full_extent}")
            else:
                full_extent = speech_timeline.extent()
                logger.debug(f"{uri}: Using speech timeline extent: {full_extent}")

            # Get gaps (nonspeech) - use extrude to remove speech from full timeline
            nonspeech_timeline = Timeline([full_extent]).extrude(speech_timeline)
            logger.debug(f"{uri}: Generated {len(nonspeech_timeline)} nonspeech gaps")

            for segment in nonspeech_timeline:
                rows.append(
                    {
                        "uri": uri,
                        "start_s": segment.start,
                        "end_s": segment.end,
                        "label": "NONSPEECH",
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

    Format: CSV with columns [video_id, start_time, end_time, label]
    Labels: SPEECH_CLEAN, SPEECH_WITH_MUSIC, SPEECH_WITH_NOISE, NO_SPEECH

    Returns:
        FrameTable with speech/nonspeech labels and acoustic conditions
    """
    logger.info(f"Loading AVA-Speech from {annotations_path}")

    # Read CSV with correct column names
    df = pd.read_csv(
        annotations_path, header=None, names=["video_id", "start_s", "end_s", "original_label"]
    )

    # Extract condition from original label BEFORE mapping
    df["condition"] = df["original_label"].apply(
        lambda x: (
            "clean"
            if "CLEAN" in str(x)
            else "music" if "MUSIC" in str(x) else "noise" if "NOISE" in str(x) else "none"
        )
    )

    # Map labels to binary SPEECH/NONSPEECH
    df["label"] = df["original_label"].apply(
        lambda x: "SPEECH" if "SPEECH" in str(x) and "NO_SPEECH" not in str(x) else "NONSPEECH"
    )

    df["uri"] = df["video_id"]
    df["split"] = split
    df["dataset"] = dataset_name

    # Keep only required columns
    df = df[["uri", "start_s", "end_s", "label", "split", "dataset", "condition"]]

    # Note: PROTOTYPE_MODE filtering should be done by caller if needed
    # to allow filtering by available audio files

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
