"""
Tests for segment slicing.
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from pyannote.core import Segment

from qsm.data.slicing import (
    slice_segments_from_interval,
    balance_segments,
    SegmentMetadata
)


def test_slice_segments_from_interval():
    """Test slicing segments from an interval."""
    # Create a 10-second interval
    interval = Segment(0.0, 10.0)

    # Slice 100ms segments
    segments = slice_segments_from_interval(
        interval=interval,
        duration_ms=100,
        mode="speech"
    )

    assert len(segments) == 100  # 10s / 0.1s = 100 segments

    # Check first segment
    assert segments[0].start == 0.0
    assert segments[0].end == 0.1

    # Check last segment
    assert segments[-1].start == 9.9
    assert segments[-1].end == 10.0


def test_slice_with_max_segments():
    """Test limiting number of segments."""
    interval = Segment(0.0, 10.0)

    segments = slice_segments_from_interval(
        interval=interval,
        duration_ms=100,
        max_segments=10,
        mode="speech"
    )

    assert len(segments) == 10


def test_slice_interval_too_short():
    """Test that too-short intervals return empty list."""
    interval = Segment(0.0, 0.05)  # 50ms

    segments = slice_segments_from_interval(
        interval=interval,
        duration_ms=100,  # Need 100ms
        mode="speech"
    )

    assert len(segments) == 0


def test_balance_segments():
    """Test segment balancing."""
    # Create unbalanced dataset
    data = []

    # 100 clean speech segments
    for i in range(100):
        data.append({
            "uri": f"file_{i}",
            "start_s": 0.0,
            "end_s": 0.1,
            "duration_ms": 100,
            "label": "SPEECH",
            "condition": "clean",
            "dataset": "test",
            "split": "train"
        })

    # 20 noisy speech segments
    for i in range(20):
        data.append({
            "uri": f"file_noise_{i}",
            "start_s": 0.0,
            "end_s": 0.1,
            "duration_ms": 100,
            "label": "SPEECH",
            "condition": "noise",
            "dataset": "test",
            "split": "train"
        })

    df = pd.DataFrame(data)

    # Balance by condition
    balanced = balance_segments(df, balance_by=["condition"])

    # Should have equal counts
    clean_count = len(balanced[balanced["condition"] == "clean"])
    noise_count = len(balanced[balanced["condition"] == "noise"])

    assert clean_count == noise_count == 20


def test_segment_metadata():
    """Test SegmentMetadata dataclass."""
    meta = SegmentMetadata(
        uri="test_file",
        start_s=0.0,
        end_s=0.1,
        duration_ms=100,
        label="SPEECH",
        dataset="test",
        split="train",
        condition="clean"
    )

    assert meta.uri == "test_file"
    assert meta.duration_ms == 100
    assert meta.label == "SPEECH"
    assert meta.condition == "clean"


def test_slice_various_durations():
    """Test slicing at multiple target durations."""
    interval = Segment(0.0, 1.0)  # 1 second

    test_cases = [
        (20, 50),   # 1000ms / 20ms = 50
        (40, 25),   # 1000ms / 40ms = 25
        (100, 10),  # 1000ms / 100ms = 10
        (500, 2),   # 1000ms / 500ms = 2
    ]

    for duration_ms, expected_count in test_cases:
        segments = slice_segments_from_interval(
            interval=interval,
            duration_ms=duration_ms,
            mode="speech"
        )

        assert len(segments) == expected_count, f"Failed for {duration_ms}ms"

        # Verify all segments have correct duration
        for seg in segments:
            assert abs(seg.duration - duration_ms / 1000.0) < 1e-6


def test_speech_nonspeech_mode():
    """Test that mode parameter is accepted."""
    interval = Segment(0.0, 1.0)

    # Should work for both modes
    speech_segs = slice_segments_from_interval(interval, 100, mode="speech")
    nonspeech_segs = slice_segments_from_interval(interval, 100, mode="nonspeech")

    assert len(speech_segs) == len(nonspeech_segs) == 10
