"""
Band-limited filtering module.

Implements telephony band-pass (300-3400 Hz) and ablation filters (LP/HP).
Uses zero-phase Butterworth IIR filters for clean frequency response.
"""

from typing import Literal

import numpy as np
from scipy import signal

FilterType = Literal["bandpass", "lowpass", "highpass"]


def apply_filter(
    audio: np.ndarray,
    sr: int,
    filter_type: FilterType,
    lowcut: float = None,
    highcut: float = None,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth filter to audio.

    Args:
        audio: Input audio (samples,)
        sr: Sample rate (Hz)
        filter_type: "bandpass", "lowpass", or "highpass"
        lowcut: Low cutoff frequency (Hz) for bandpass/highpass
        highcut: High cutoff frequency (Hz) for bandpass/lowpass
        order: Filter order (default 4)

    Returns:
        Filtered audio
    """
    nyquist = sr / 2.0

    if filter_type == "bandpass":
        if lowcut is None or highcut is None:
            raise ValueError("bandpass requires both lowcut and highcut")
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = signal.butter(order, [low, high], btype="band", output="sos")

    elif filter_type == "lowpass":
        if highcut is None:
            raise ValueError("lowpass requires highcut")
        high = highcut / nyquist
        sos = signal.butter(order, high, btype="low", output="sos")

    elif filter_type == "highpass":
        if lowcut is None:
            raise ValueError("highpass requires lowcut")
        low = lowcut / nyquist
        sos = signal.butter(order, low, btype="high", output="sos")

    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    # Apply zero-phase filtering (forward-backward)
    filtered = signal.sosfiltfilt(sos, audio)
    return filtered.astype(audio.dtype)


def apply_bandpass(
    audio: np.ndarray,
    sr: int,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply telephony band-pass filter (300-3400 Hz by default).

    Standard telephony band per ITU-T.

    Args:
        audio: Input audio (samples,)
        sr: Sample rate (Hz)
        lowcut: Low cutoff (default 300 Hz)
        highcut: High cutoff (default 3400 Hz)
        order: Filter order (default 4)

    Returns:
        Band-limited audio
    """
    return apply_filter(
        audio, sr, filter_type="bandpass", lowcut=lowcut, highcut=highcut, order=order
    )


def apply_lowpass(
    audio: np.ndarray,
    sr: int,
    highcut: float = 3400.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply low-pass filter (ablation: no high-pass component).

    Args:
        audio: Input audio (samples,)
        sr: Sample rate (Hz)
        highcut: High cutoff (default 3400 Hz)
        order: Filter order (default 4)

    Returns:
        Low-pass filtered audio
    """
    return apply_filter(audio, sr, filter_type="lowpass", highcut=highcut, order=order)


def apply_highpass(
    audio: np.ndarray,
    sr: int,
    lowcut: float = 300.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply high-pass filter (ablation: no low-pass component).

    Args:
        audio: Input audio (samples,)
        sr: Sample rate (Hz)
        lowcut: Low cutoff (default 300 Hz)
        order: Filter order (default 4)

    Returns:
        High-pass filtered audio
    """
    return apply_filter(audio, sr, filter_type="highpass", lowcut=lowcut, order=order)
