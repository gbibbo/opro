"""
WebRTC-VAD implementation.

WebRTC Voice Activity Detector supports frame durations of 10, 20, or 30ms
and aggressiveness levels 0-3 (0=least aggressive, 3=most aggressive).

Reference: https://github.com/wiseman/py-webrtcvad
"""

import time
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad

from .base import VADModel, VADPrediction


class WebRTCVAD(VADModel):
    """WebRTC Voice Activity Detector."""

    VALID_FRAME_DURATIONS = [10, 20, 30]  # milliseconds
    VALID_AGGRESSIVENESS = [0, 1, 2, 3]

    def __init__(
        self,
        frame_duration_ms: int = 30,
        aggressiveness: int = 1,
        sample_rate: int = 16000,
        hysteresis_frames: int = 0,
    ):
        """
        Initialize WebRTC-VAD.

        Args:
            frame_duration_ms: Frame duration in ms (10, 20, or 30)
            aggressiveness: Aggressiveness level (0-3)
            sample_rate: Sample rate in Hz (8000, 16000, 32000, or 48000)
            hysteresis_frames: Number of consecutive frames for decision smoothing (0=disabled)

        Raises:
            ValueError: If parameters are invalid
        """
        if frame_duration_ms not in self.VALID_FRAME_DURATIONS:
            raise ValueError(
                f"frame_duration_ms must be one of {self.VALID_FRAME_DURATIONS}, got {frame_duration_ms}"
            )

        if aggressiveness not in self.VALID_AGGRESSIVENESS:
            raise ValueError(
                f"aggressiveness must be one of {self.VALID_AGGRESSIVENESS}, got {aggressiveness}"
            )

        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"sample_rate must be 8000, 16000, 32000, or 48000, got {sample_rate}")

        self._frame_duration_ms = frame_duration_ms
        self._aggressiveness = aggressiveness
        self._sample_rate = sample_rate
        self._hysteresis_frames = hysteresis_frames

        # Initialize VAD
        self.vad = webrtcvad.Vad(aggressiveness)

        # Calculate frame size in samples
        self._frame_size = int(sample_rate * frame_duration_ms / 1000)

    @property
    def frame_duration_ms(self) -> int:
        """Frame duration in milliseconds."""
        return self._frame_duration_ms

    @property
    def name(self) -> str:
        """Model name for logging/results."""
        return f"webrtc_vad_{self._frame_duration_ms}ms_agg{self._aggressiveness}"

    def predict_frames(self, audio: np.ndarray, sample_rate: int) -> list[bool]:
        """
        Predict speech presence frame-by-frame.

        Args:
            audio: Audio samples (1D numpy array, float32 or int16)
            sample_rate: Sample rate in Hz

        Returns:
            List of boolean decisions (True = speech, False = nonspeech) per frame
        """
        if sample_rate != self._sample_rate:
            raise ValueError(
                f"Sample rate mismatch: expected {self._sample_rate}, got {sample_rate}"
            )

        # Convert to int16 if needed (WebRTC expects int16 PCM)
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = (audio * 32767).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)

        # Pad audio to multiple of frame size
        remainder = len(audio) % self._frame_size
        if remainder != 0:
            padding = self._frame_size - remainder
            audio = np.pad(audio, (0, padding), mode="constant")

        # Process frame by frame
        num_frames = len(audio) // self._frame_size
        frame_decisions = []

        for i in range(num_frames):
            start = i * self._frame_size
            end = start + self._frame_size
            frame = audio[start:end].tobytes()

            # WebRTC VAD returns True for speech, False for nonspeech
            is_speech = self.vad.is_speech(frame, sample_rate)
            frame_decisions.append(is_speech)

        # Apply hysteresis if enabled
        if self._hysteresis_frames > 0:
            frame_decisions = self._apply_hysteresis(frame_decisions, self._hysteresis_frames)

        return frame_decisions

    def predict(self, audio_path: Path) -> VADPrediction:
        """
        Predict speech presence in audio file.

        Args:
            audio_path: Path to audio file (WAV, mono, 16kHz expected)

        Returns:
            VADPrediction with label, confidence, and latency
        """
        start_time = time.time()

        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (basic resampling, for production use librosa.resample)
        if sr != self._sample_rate:
            # Simple nearest-neighbor resampling (sufficient for VAD)
            ratio = self._sample_rate / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            audio = audio[indices]
            sr = self._sample_rate

        # Get frame decisions
        frame_decisions = self.predict_frames(audio, sr)

        # Aggregate: percentage of speech frames
        speech_ratio = sum(frame_decisions) / len(frame_decisions) if frame_decisions else 0.0

        # Decision: SPEECH if majority of frames are speech
        label = "SPEECH" if speech_ratio >= 0.5 else "NONSPEECH"
        confidence = speech_ratio if label == "SPEECH" else (1.0 - speech_ratio)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        return VADPrediction(
            label=label,
            confidence=confidence,
            latency_ms=latency_ms,
            frame_decisions=frame_decisions,
        )

    @staticmethod
    def _apply_hysteresis(decisions: list[bool], n_frames: int) -> list[bool]:
        """
        Apply hysteresis smoothing: require n consecutive frames to change state.

        Args:
            decisions: Raw frame decisions
            n_frames: Number of consecutive frames required

        Returns:
            Smoothed decisions
        """
        if n_frames <= 0:
            return decisions

        smoothed = []
        current_state = decisions[0] if decisions else False
        consecutive_count = 0

        for decision in decisions:
            if decision == current_state:
                consecutive_count = 0
            else:
                consecutive_count += 1
                if consecutive_count >= n_frames:
                    current_state = decision
                    consecutive_count = 0

            smoothed.append(current_state)

        return smoothed
