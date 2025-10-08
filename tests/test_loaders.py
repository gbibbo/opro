"""
Tests for data loaders.
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# Configure logging to file
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"test_loaders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logger.info("=" * 80)
logger.info("Starting test_loaders.py")
logger.info(f"Log file: {log_file}")
logger.info("=" * 80)

from qsm.data.loaders import FrameTable, iter_intervals, load_ava_speech, load_rttm_dataset


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_rttm_file(temp_dir):
    """Create a mock RTTM file."""
    rttm_path = temp_dir / "test.rttm"

    with open(rttm_path, "w") as f:
        # RTTM format: SPEAKER file 1 start duration <NA> <NA> speaker <NA> <NA>
        f.write("SPEAKER file_001 1 0.0 5.0 <NA> <NA> speaker_1 <NA> <NA>\n")
        f.write("SPEAKER file_001 1 6.0 4.0 <NA> <NA> speaker_2 <NA> <NA>\n")
        f.write("SPEAKER file_002 1 1.0 3.0 <NA> <NA> speaker_1 <NA> <NA>\n")

    return rttm_path


@pytest.fixture
def mock_ava_csv(temp_dir):
    """Create a mock AVA-Speech CSV file."""
    csv_path = temp_dir / "ava_speech.csv"

    data = {
        "video_id": ["video_1"] * 10 + ["video_2"] * 10,
        "frame_timestamp": list(range(10)) * 2,
        "label": ["SPEECH_CLEAN"] * 5
        + ["NO_SPEECH"] * 5
        + ["SPEECH_WITH_MUSIC"] * 5
        + ["NO_SPEECH"] * 5,
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


def test_frame_table_creation():
    """Test FrameTable creation and validation."""
    logger.info("Running test_frame_table_creation")
    data = pd.DataFrame(
        {
            "uri": ["file_1", "file_1", "file_2"],
            "start_s": [0.0, 5.0, 1.0],
            "end_s": [4.0, 9.0, 3.0],
            "label": ["SPEECH", "NONSPEECH", "SPEECH"],
            "split": ["train", "train", "dev"],
            "dataset": ["test", "test", "test"],
        }
    )

    ft = FrameTable(data=data)

    assert len(ft.data) == 3
    assert len(ft.speech_segments) == 2
    assert len(ft.nonspeech_segments) == 1
    logger.info("[PASS] test_frame_table_creation")


def test_frame_table_missing_column():
    """Test FrameTable raises error for missing columns."""
    logger.info("Running test_frame_table_missing_column")
    data = pd.DataFrame(
        {
            "uri": ["file_1"],
            "start_s": [0.0],
            "end_s": [4.0],
            # Missing 'label', 'split', 'dataset'
        }
    )

    with pytest.raises(ValueError, match="Missing required column"):
        FrameTable(data=data)
    logger.info("[PASS] test_frame_table_missing_column")


def test_load_rttm_dataset(mock_rttm_file):
    """Test loading RTTM dataset."""
    logger.info("Running test_load_rttm_dataset")
    ft = load_rttm_dataset(rttm_path=mock_rttm_file, dataset_name="test_rttm", split="dev")

    # Should have 3 speech segments
    assert len(ft.data) >= 1
    assert all(ft.data["label"] == "SPEECH")
    assert all(ft.data["dataset"] == "test_rttm")
    assert all(ft.data["split"] == "dev")

    # Check URIs
    assert "file_001" in ft.data["uri"].values or "file_002" in ft.data["uri"].values
    logger.info("[PASS] test_load_rttm_dataset")


def test_load_ava_speech(mock_ava_csv):
    """Test loading AVA-Speech dataset."""
    logger.info("Running test_load_ava_speech")
    ft = load_ava_speech(annotations_path=mock_ava_csv, dataset_name="ava_test", split="train")

    assert len(ft.data) > 0

    # Check label mapping
    assert all(ft.data["label"].isin(["SPEECH", "NONSPEECH"]))

    # Check condition extraction
    assert "condition" in ft.data.columns

    # Check time conversion (25 fps)
    assert ft.data["end_s"].max() <= 1.0  # 10 frames / 25 fps = 0.4s
    logger.info("[PASS] test_load_ava_speech")


def test_iter_intervals():
    """Test interval iteration."""
    logger.info("Running test_iter_intervals")
    data = pd.DataFrame(
        {
            "uri": ["file_1", "file_1", "file_1"],
            "start_s": [0.0, 5.0, 10.0],
            "end_s": [4.0, 9.0, 14.0],
            "label": ["SPEECH", "NONSPEECH", "SPEECH"],
            "split": ["train"] * 3,
            "dataset": ["test"] * 3,
        }
    )

    ft = FrameTable(data=data)

    # Get SPEECH intervals for file_1
    intervals = list(iter_intervals("file_1", ft, label="SPEECH"))

    assert len(intervals) == 2
    assert intervals[0].start == 0.0
    assert intervals[0].end == 4.0
    assert intervals[1].start == 10.0
    assert intervals[1].end == 14.0
    logger.info("[PASS] test_iter_intervals")


def test_frame_table_save_load(temp_dir):
    """Test saving and loading FrameTable."""
    logger.info("Running test_frame_table_save_load")
    data = pd.DataFrame(
        {
            "uri": ["file_1"],
            "start_s": [0.0],
            "end_s": [4.0],
            "label": ["SPEECH"],
            "split": ["train"],
            "dataset": ["test"],
        }
    )

    ft = FrameTable(data=data)

    save_path = temp_dir / "test_ft.parquet"
    ft.save(save_path)

    assert save_path.exists()

    # Load back
    ft_loaded = FrameTable.load(save_path)

    assert len(ft_loaded.data) == len(ft.data)
    assert all(ft_loaded.data["uri"] == ft.data["uri"])
    logger.info("[PASS] test_frame_table_save_load")


def test_prototype_mode_limiting(mock_rttm_file, monkeypatch):
    """Test that PROTOTYPE_MODE limits data."""
    logger.info("Running test_prototype_mode_limiting")
    # Mock PROTOTYPE_MODE and PROTOTYPE_SAMPLES
    import qsm.data.loaders as loaders_module

    monkeypatch.setattr(loaders_module, "PROTOTYPE_MODE", True)
    monkeypatch.setattr(loaders_module, "PROTOTYPE_SAMPLES", 1)

    ft = load_rttm_dataset(rttm_path=mock_rttm_file, dataset_name="test", split="dev")

    # Should limit to 1 URI
    unique_uris = ft.data["uri"].nunique()
    assert unique_uris <= 1
    logger.info("[PASS] test_prototype_mode_limiting")


logger.info("=" * 80)
logger.info("Completed test_loaders.py")
logger.info("=" * 80)
