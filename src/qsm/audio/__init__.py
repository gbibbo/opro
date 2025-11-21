"""
Psychoacoustic audio manipulation modules for QSM evaluation.

Modules:
  - slicing: Duration-based segmentation and padding
  - noise: White noise / SNR sweep
  - filters: Band-limited filtering (telephony, LP, HP)
  - reverb: Reverberation (RIR convolution)
"""

from .filters import apply_bandpass, apply_highpass, apply_lowpass
from .noise import add_white_noise, mix_at_snr
from .reverb import apply_rir, load_rir_database
from .slicing import (
    create_duration_variants,
    extract_segment_center,
    pad_audio_center,
    slice_and_pad,
)

__all__ = [
    # Slicing/segmentation
    "extract_segment_center",
    "pad_audio_center",
    "slice_and_pad",
    "create_duration_variants",
    # Noise
    "add_white_noise",
    "mix_at_snr",
    # Filters
    "apply_bandpass",
    "apply_lowpass",
    "apply_highpass",
    # Reverb
    "apply_rir",
    "load_rir_database",
]
