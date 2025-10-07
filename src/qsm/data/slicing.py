"""
Segment slicing for specific target durations.

Creates balanced datasets at various temporal resolutions:
{20, 40, 60, 80, 100, 150, 200, 300, 500, 1000} ms
"""

from pathlib import Path
from typing import List, Optional, Literal
import numpy as np
import soundfile as sf
import pandas as pd
from dataclasses import dataclass
import json
import logging
from tqdm import tqdm

from qsm import CONFIG
from .loaders import FrameTable, Segment

logger = logging.getLogger(__name__)


@dataclass
class SegmentMetadata:
    """Metadata for a sliced segment."""
    uri: str
    start_s: float
    end_s: float
    duration_ms: int
    label: str
    dataset: str
    split: str
    condition: Optional[str] = None
    snr_bin: Optional[str] = None
    audio_path: str = ""


def slice_segments_from_interval(
    interval: Segment,
    duration_ms: int,
    max_segments: Optional[int] = None,
    mode: Literal["speech", "nonspeech"] = "speech"
) -> List[Segment]:
    """
    Slice fixed-duration segments from an interval.

    Args:
        interval: Source interval (must be completely speech or nonspeech)
        duration_ms: Target duration in milliseconds
        max_segments: Maximum number of segments to extract
        mode: 'speech' or 'nonspeech'

    Returns:
        List of Segment objects
    """
    duration_s = duration_ms / 1000.0
    interval_duration = interval.duration

    if interval_duration < duration_s:
        return []  # Interval too short

    # Calculate how many segments fit
    num_segments = int(interval_duration / duration_s)

    if max_segments is not None:
        num_segments = min(num_segments, max_segments)

    segments = []
    for i in range(num_segments):
        start = interval.start + i * duration_s
        end = start + duration_s
        segments.append(Segment(start, end))

    return segments


def create_segments(
    frame_table: FrameTable,
    audio_root: Path,
    output_dir: Path,
    durations_ms: List[int],
    balance_by: Optional[List[str]] = None,
    max_segments_per_config: int = 1000
) -> pd.DataFrame:
    """
    Create balanced segment dataset at multiple durations.

    Args:
        frame_table: Source annotations
        audio_root: Root directory containing audio files
        output_dir: Output directory for sliced segments
        durations_ms: List of target durations in milliseconds
        balance_by: List of columns to balance by (e.g., ['condition', 'snr_bin'])
        max_segments_per_config: Max segments per (duration, condition) combination

    Returns:
        DataFrame with segment metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_segments = []

    for duration_ms in durations_ms:
        logger.info(f"Processing duration: {duration_ms}ms")

        # Extract speech segments
        speech_segs = _extract_segments_for_duration(
            frame_table=frame_table.speech_segments,
            duration_ms=duration_ms,
            label="SPEECH",
            max_segments=max_segments_per_config
        )

        # Extract nonspeech segments
        nonspeech_segs = _extract_segments_for_duration(
            frame_table=frame_table.nonspeech_segments,
            duration_ms=duration_ms,
            label="NONSPEECH",
            max_segments=max_segments_per_config
        )

        # Balance
        min_count = min(len(speech_segs), len(nonspeech_segs))
        speech_segs = speech_segs[:min_count]
        nonspeech_segs = nonspeech_segs[:min_count]

        all_segments.extend(speech_segs)
        all_segments.extend(nonspeech_segs)

    # Export audio and metadata
    metadata_records = []

    for seg_meta in tqdm(all_segments, desc="Exporting segments"):
        # Find source audio
        audio_path = audio_root / f"{seg_meta.uri}.wav"

        if not audio_path.exists():
            logger.warning(f"Audio not found: {audio_path}")
            continue

        # Load and slice audio
        try:
            audio, sr = sf.read(audio_path)

            start_sample = int(seg_meta.start_s * sr)
            end_sample = int(seg_meta.end_s * sr)

            segment_audio = audio[start_sample:end_sample]

            # Save segment
            output_filename = f"{seg_meta.dataset}_{seg_meta.uri}_{seg_meta.start_s:.3f}_{seg_meta.duration_ms}ms.wav"
            output_path = output_dir / output_filename

            sf.write(output_path, segment_audio, sr)

            # Update metadata
            seg_meta.audio_path = str(output_path)
            metadata_records.append(seg_meta.__dict__)

        except Exception as e:
            logger.error(f"Failed to process segment {seg_meta.uri}: {e}")
            continue

    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_path = output_dir / "segments_metadata.parquet"
    metadata_df.to_parquet(metadata_path, index=False)

    # Also save JSONL for easy inspection
    jsonl_path = output_dir / "segments_metadata.jsonl"
    with open(jsonl_path, "w") as f:
        for record in metadata_records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Created {len(metadata_records)} segments")
    logger.info(f"Metadata saved to {metadata_path}")

    return metadata_df


def _extract_segments_for_duration(
    frame_table: pd.DataFrame,
    duration_ms: int,
    label: str,
    max_segments: int
) -> List[SegmentMetadata]:
    """Helper to extract segments for a specific duration."""
    segments = []

    for _, row in frame_table.iterrows():
        interval = Segment(row["start_s"], row["end_s"])

        slices = slice_segments_from_interval(
            interval=interval,
            duration_ms=duration_ms,
            max_segments=1,  # One per interval for now
            mode=label.lower()
        )

        for seg in slices:
            segments.append(
                SegmentMetadata(
                    uri=row["uri"],
                    start_s=seg.start,
                    end_s=seg.end,
                    duration_ms=duration_ms,
                    label=label,
                    dataset=row.get("dataset", "unknown"),
                    split=row.get("split", "train"),
                    condition=row.get("condition"),
                    snr_bin=row.get("snr_bin")
                )
            )

        if len(segments) >= max_segments:
            break

    return segments[:max_segments]


def balance_segments(
    segments_df: pd.DataFrame,
    balance_by: List[str]
) -> pd.DataFrame:
    """
    Balance segments across multiple dimensions.

    Args:
        segments_df: DataFrame with segment metadata
        balance_by: Columns to balance by

    Returns:
        Balanced DataFrame
    """
    # Group and find minimum count
    grouped = segments_df.groupby(balance_by)
    min_count = grouped.size().min()

    # Sample equally from each group
    balanced = grouped.apply(
        lambda x: x.sample(n=min(len(x), min_count), random_state=CONFIG["seed"])
    ).reset_index(drop=True)

    logger.info(f"Balanced to {len(balanced)} segments across {balance_by}")

    return balanced
