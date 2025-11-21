"""
Reverberation module using Room Impulse Response (RIR) convolution.

Uses OpenSLR SLR28 RIRS_NOISES dataset.
Dataset: https://www.openslr.org/28/
"""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


class RIRDatabase:
    """
    RIR database loader and manager.

    Loads RIRs from OpenSLR SLR28 structure:
      {rir_root}/simulated_rirs/*/Room*.wav  -> simulated
      {rir_root}/real_rirs_isotropic_noises/*.wav -> real

    Metadata should include:
      - rir_id: unique identifier
      - T60: reverberation time (sec)
      - room_dims: (optional) room dimensions
      - source_dist: (optional) source-mic distance
    """

    def __init__(self, rir_root: Path, metadata_path: Path | None = None):
        """
        Initialize RIR database.

        Args:
            rir_root: Root directory of RIR dataset
            metadata_path: Optional path to metadata JSON with T60 annotations
        """
        self.rir_root = Path(rir_root)
        self.metadata_path = metadata_path
        self.rirs: dict[str, dict] = {}
        self._load_database()

    def _load_database(self):
        """Scan RIR directory and load metadata."""
        # Simulated RIRs
        sim_dir = self.rir_root / "simulated_rirs"
        if sim_dir.exists():
            for rir_file in sim_dir.rglob("*.wav"):
                rir_id = str(rir_file.relative_to(self.rir_root))
                self.rirs[rir_id] = {
                    "path": rir_file,
                    "type": "simulated",
                    "T60": None,  # Will be populated from metadata if available
                }

        # Real RIRs
        real_dir = self.rir_root / "real_rirs_isotropic_noises"
        if real_dir.exists():
            for rir_file in real_dir.glob("*.wav"):
                rir_id = str(rir_file.relative_to(self.rir_root))
                self.rirs[rir_id] = {
                    "path": rir_file,
                    "type": "real",
                    "T60": None,
                }

        # Load metadata if available
        if self.metadata_path and Path(self.metadata_path).exists():
            with open(self.metadata_path) as f:
                metadata = json.load(f)
            for rir_id, meta in metadata.items():
                if rir_id in self.rirs:
                    self.rirs[rir_id].update(meta)

    def get_rir(self, rir_id: str, sr: int = 16000) -> np.ndarray:
        """
        Load RIR audio.

        Args:
            rir_id: RIR identifier
            sr: Target sample rate (will resample if needed)

        Returns:
            RIR audio array (samples,)
        """
        if rir_id not in self.rirs:
            raise KeyError(f"RIR {rir_id} not found in database")

        rir_path = self.rirs[rir_id]["path"]
        rir, rir_sr = sf.read(rir_path, dtype="float32")

        # Convert to mono if stereo
        if rir.ndim > 1:
            rir = rir.mean(axis=1)

        # Resample if needed
        if rir_sr != sr:
            num_samples = int(len(rir) * sr / rir_sr)
            rir = signal.resample(rir, num_samples)

        return rir

    def get_by_t60(self, t60_min: float, t60_max: float) -> list[str]:
        """
        Get RIR IDs within T60 range.

        Args:
            t60_min: Minimum T60 (sec)
            t60_max: Maximum T60 (sec)

        Returns:
            List of RIR IDs
        """
        return [
            rir_id
            for rir_id, meta in self.rirs.items()
            if meta.get("T60") is not None and t60_min <= meta["T60"] <= t60_max
        ]

    def list_all(self) -> list[str]:
        """List all available RIR IDs."""
        return list(self.rirs.keys())


def apply_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    normalize: bool = True,
    rir_gain: float = 1.0,
) -> np.ndarray:
    """
    Apply RIR to audio via convolution.

    Args:
        audio: Input audio (samples,)
        rir: Room impulse response (samples,)
        normalize: If True, normalize output to preserve RMS energy
        rir_gain: Gain factor for RIR before convolution (default 1.0)

    Returns:
        Reverberant audio (same length as input, truncated convolution)
    """
    # Scale RIR
    rir_scaled = rir * rir_gain

    # Convolve
    reverb = signal.fftconvolve(audio, rir_scaled, mode="full")

    # Truncate to original length
    reverb = reverb[: len(audio)]

    # Normalize to preserve energy
    if normalize:
        rms_orig = np.sqrt(np.mean(audio**2))
        rms_reverb = np.sqrt(np.mean(reverb**2))
        if rms_reverb > 1e-8:
            reverb = reverb * (rms_orig / rms_reverb)

    return reverb.astype(audio.dtype)


def load_rir_database(rir_root: Path, metadata_path: Path | None = None) -> RIRDatabase:
    """
    Convenience function to load RIR database.

    Args:
        rir_root: Root directory of RIR dataset
        metadata_path: Optional metadata JSON

    Returns:
        RIRDatabase instance
    """
    return RIRDatabase(rir_root, metadata_path)
