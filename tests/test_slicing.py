"""
Tests for segment slicing with safety buffers.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from pyannote.core import Segment

# Configure logging to file
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"test_slicing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("Starting test_slicing.py")
logger.info(f"Log file: {log_file}")
logger.info("=" * 80)

from qsm.data.slicing import SegmentMetadata, balance_segments, slice_segments_from_interval


def test_slice_segments_from_interval():
    """Test slicing segments from an interval with safety buffers."""
    logger.info("Running test_slice_segments_from_interval")
    # Create a 10-second interval
    interval = Segment(0.0, 10.0)

    # Slice 100ms segments with 1s safety buffer (default)
    # Valid zone: [1.0, 9.0] = 8 seconds → 80 segments of 100ms
    segments = slice_segments_from_interval(interval=interval, duration_ms=100, mode="speech")

    assert len(segments) == 80  # 8s / 0.1s = 80 segments (with 1s buffer on each side)

    # Check first segment (starts at 1.0s due to safety buffer)
    assert segments[0].start == 1.0
    assert segments[0].end == 1.1

    # Check last segment (ends at 9.0s due to safety buffer)
    assert segments[-1].start == 8.9
    assert segments[-1].end == 9.0
    logger.info("[PASS] test_slice_segments_from_interval")


def test_slice_with_max_segments():
    """Test limiting number of segments."""
    logger.info("Running test_slice_with_max_segments")
    interval = Segment(0.0, 10.0)

    segments = slice_segments_from_interval(
        interval=interval, duration_ms=100, max_segments=10, mode="speech"
    )

    assert len(segments) == 10
    logger.info("[PASS] test_slice_with_max_segments")


def test_slice_interval_too_short():
    """Test that too-short intervals return empty list after safety buffers."""
    logger.info("Running test_slice_interval_too_short")
    # Need at least 2.1s for 100ms segment with 1s buffer on each side
    interval = Segment(0.0, 2.0)  # 2s total, but only 0s valid after buffers

    segments = slice_segments_from_interval(
        interval=interval, duration_ms=100, mode="speech"  # Need 100ms + 2s buffers
    )

    assert len(segments) == 0  # Interval too short after buffers
    logger.info("[PASS] test_slice_interval_too_short")


def test_balance_segments():
    """Test segment balancing."""
    logger.info("Running test_balance_segments")
    # Create unbalanced dataset
    data = []

    # 100 clean speech segments
    for i in range(100):
        data.append(
            {
                "uri": f"file_{i}",
                "start_s": 0.0,
                "end_s": 0.1,
                "duration_ms": 100,
                "label": "SPEECH",
                "condition": "clean",
                "dataset": "test",
                "split": "train",
            }
        )

    # 20 noisy speech segments
    for i in range(20):
        data.append(
            {
                "uri": f"file_noise_{i}",
                "start_s": 0.0,
                "end_s": 0.1,
                "duration_ms": 100,
                "label": "SPEECH",
                "condition": "noise",
                "dataset": "test",
                "split": "train",
            }
        )

    df = pd.DataFrame(data)

    # Balance by condition
    balanced = balance_segments(df, balance_by=["condition"])

    # Should have equal counts
    clean_count = len(balanced[balanced["condition"] == "clean"])
    noise_count = len(balanced[balanced["condition"] == "noise"])

    assert clean_count == noise_count == 20
    logger.info("[PASS] test_balance_segments")


def test_segment_metadata():
    """Test SegmentMetadata dataclass."""
    logger.info("Running test_segment_metadata")
    meta = SegmentMetadata(
        uri="test_file",
        start_s=0.0,
        end_s=0.1,
        duration_ms=100,
        label="SPEECH",
        dataset="test",
        split="train",
        condition="clean",
    )

    assert meta.uri == "test_file"
    assert meta.duration_ms == 100
    assert meta.label == "SPEECH"
    assert meta.condition == "clean"
    logger.info("[PASS] test_segment_metadata")


def test_slice_various_durations():
    """Test slicing at multiple target durations with safety buffers."""
    logger.info("Running test_slice_various_durations")
    # Use 5-second interval: valid zone is [1.0, 4.0] = 3 seconds after buffers
    interval = Segment(0.0, 5.0)

    test_cases = [
        (20, 150),  # 3000ms / 20ms = 150
        (40, 75),  # 3000ms / 40ms = 75
        (100, 30),  # 3000ms / 100ms = 30
        (500, 6),  # 3000ms / 500ms = 6
        (1000, 3),  # 3000ms / 1000ms = 3
    ]

    for duration_ms, expected_count in test_cases:
        segments = slice_segments_from_interval(
            interval=interval, duration_ms=duration_ms, mode="speech"
        )

        assert (
            len(segments) == expected_count
        ), f"Failed for {duration_ms}ms: expected {expected_count}, got {len(segments)}"

        # Verify all segments have correct duration
        for seg in segments:
            assert abs(seg.duration - duration_ms / 1000.0) < 1e-6

        # Verify all segments are within safety buffer zone [1.0, 4.0]
        for seg in segments:
            assert seg.start >= 1.0, f"Segment starts before buffer: {seg.start}"
            assert seg.end <= 4.0, f"Segment ends after buffer: {seg.end}"

    logger.info("[PASS] test_slice_various_durations")


def test_speech_nonspeech_mode():
    """Test that mode parameter is accepted."""
    logger.info("Running test_speech_nonspeech_mode")
    # Use 5s interval: valid zone is [1.0, 4.0] = 3s → 30 segments of 100ms
    interval = Segment(0.0, 5.0)

    # Should work for both modes
    speech_segs = slice_segments_from_interval(interval, 100, mode="speech")
    nonspeech_segs = slice_segments_from_interval(interval, 100, mode="nonspeech")

    assert len(speech_segs) == len(nonspeech_segs) == 30
    logger.info("[PASS] test_speech_nonspeech_mode")


logger.info("=" * 80)
logger.info("Completed test_slicing.py")
logger.info("=" * 80)
