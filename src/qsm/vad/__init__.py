"""
Voice Activity Detection (VAD) baselines.

Implements low-latency VAD models for comparison with Qwen:
- WebRTC-VAD (10/20/30ms frames)
- Silero-VAD (32-96ms frames)
"""

from .base import VADModel, VADPrediction
from .webrtc import WebRTCVAD
from .silero import SileroVAD

__all__ = [
    "VADModel",
    "VADPrediction",
    "WebRTCVAD",
    "SileroVAD",
]
