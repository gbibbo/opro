"""
Duration-based audio slicing/segmentation.

Extracts segments of different durations from audio files, applying
center extraction and padding for consistent psychoacoustic evaluation.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_segment_center(
    audio: np.ndarray,
    duration_ms: int,
    sr: int = 16000,
) -> np.ndarray:
    """
    Extract a segment of specified duration from the CENTER of audio.

    Args:
        audio: Input audio array
        duration_ms: Target duration in milliseconds
        sr: Sample rate (Hz)

    Returns:
        Extracted segment (centered within input audio)

    Example:
        >>> audio_1000ms = np.random.randn(16000)  # 1000ms at 16kHz
        >>> segment_100ms = extract_segment_center(audio_1000ms, 100, sr=16000)
        >>> len(segment_100ms)  # 1600 samples = 100ms
        1600
    """
    # Calculate samples for target duration
    target_samples = int(duration_ms * sr / 1000.0)
    current_samples = len(audio)

    if target_samples >= current_samples:
        # If requesting same or longer duration, return entire audio
        return audio

    # Center extraction: extract from middle of audio
    start_idx = (current_samples - target_samples) // 2
    end_idx = start_idx + target_samples

    return audio[start_idx:end_idx]


def pad_audio_center(
    audio: np.ndarray,
    target_duration_ms: int,
    sr: int = 16000,
    noise_amplitude: float = 0.0001,
    seed: int = 42,
) -> np.ndarray:
    """
    Pad audio to target duration by centering it in low-amplitude noise.

    Args:
        audio: Input audio array
        target_duration_ms: Target duration in milliseconds
        sr: Sample rate (Hz)
        noise_amplitude: Amplitude of padding noise
        seed: Random seed for reproducibility

    Returns:
        Padded audio array centered in noise padding

    Example:
        >>> audio_100ms = np.random.randn(1600)  # 100ms at 16kHz
        >>> padded = pad_audio_center(audio_100ms, 2000, sr=16000)
        >>> len(padded)  # 32000 samples = 2000ms
        32000
    """
    target_samples = int(target_duration_ms * sr / 1000.0)
    current_samples = len(audio)

    if current_samples >= target_samples:
        # Truncate if longer
        return audio[:target_samples]

    # Calculate padding needed
    total_padding = target_samples - current_samples
    padding_left = total_padding // 2
    padding_right = total_padding - padding_left

    # Generate low-amplitude noise
    rng = np.random.default_rng(seed)
    noise_left = rng.normal(0, noise_amplitude, padding_left).astype(np.float32)
    noise_right = rng.normal(0, noise_amplitude, padding_right).astype(np.float32)

    # Concatenate: [NOISE_LEFT] + [AUDIO] + [NOISE_RIGHT]
    padded = np.concatenate([noise_left, audio, noise_right])

    return padded


def slice_and_pad(
    audio: np.ndarray,
    duration_ms: int,
    padding_ms: int = 2000,
    sr: int = 16000,
    noise_amplitude: float = 0.0001,
    seed: int = 42,
) -> np.ndarray:
    """
    Extract segment of specified duration from center, then pad to target length.

    This is a convenience function combining extract_segment_center() and pad_audio_center().

    Args:
        audio: Input audio array
        duration_ms: Duration to extract (ms)
        padding_ms: Target duration after padding (ms)
        sr: Sample rate (Hz)
        noise_amplitude: Amplitude of padding noise
        seed: Random seed

    Returns:
        Extracted and padded audio segment

    Example:
        >>> audio_1000ms = np.random.randn(16000)  # 1000ms at 16kHz
        >>> segment = slice_and_pad(audio_1000ms, duration_ms=100, padding_ms=2000)
        >>> len(segment)  # 32000 samples = 2000ms
        32000
        >>> # The 100ms segment is centered in 2000ms of low-amplitude noise
    """
    # Step 1: Extract segment from center
    segment = extract_segment_center(audio, duration_ms, sr)

    # Step 2: Pad to target duration
    padded = pad_audio_center(segment, padding_ms, sr, noise_amplitude, seed)

    return padded


def extract_from_padded_1000ms(
    audio_padded_2000ms: np.ndarray,
    duration_ms: int,
    sr: int = 16000,
) -> np.ndarray:
    """
    Extract segment from a 1000ms audio that's already padded to 2000ms.

    Assumes the 1000ms content is centered in the 2000ms container.

    Args:
        audio_padded_2000ms: Input audio (1000ms centered in 2000ms padding)
        duration_ms: Target duration to extract (ms)
        sr: Sample rate (Hz)

    Returns:
        Extracted segment from the 1000ms content region

    Example:
        >>> # Audio with 1000ms centered in 2000ms padding
        >>> audio_2000ms = np.random.randn(32000)  # 2000ms at 16kHz
        >>> segment_100ms = extract_from_padded_1000ms(audio_2000ms, 100, sr=16000)
        >>> len(segment_100ms)  # 1600 samples = 100ms
        1600
    """
    target_samples = int(duration_ms * sr / 1000.0)
    total_samples = len(audio_padded_2000ms)
    padding_samples = int(2000 * sr / 1000.0)  # 2000ms in samples

    if total_samples == padding_samples:
        # Audio is in 2000ms container, 1000ms is centered
        content_samples = int(1000 * sr / 1000.0)  # 1000ms
        content_start = (padding_samples - content_samples) // 2
        content_end = content_start + content_samples
        content_audio = audio_padded_2000ms[content_start:content_end]
    else:
        # Audio is raw 1000ms or other length
        content_audio = audio_padded_2000ms

    # Extract target duration from center of content
    if target_samples >= len(content_audio):
        # If requesting full content or more, return entire content
        return content_audio

    # Center extraction
    start_idx = (len(content_audio) - target_samples) // 2
    end_idx = start_idx + target_samples

    return content_audio[start_idx:end_idx]


def create_duration_variants(
    audio_1000ms: np.ndarray,
    durations_ms: list[int],
    padding_ms: int = 2000,
    sr: int = 16000,
    noise_amplitude: float = 0.0001,
    seed_base: int = 42,
) -> dict[int, np.ndarray]:
    """
    Create multiple duration variants from a single 1000ms audio segment.

    Each variant is extracted from the center and padded to 2000ms.

    Args:
        audio_1000ms: Input audio (should be 1000ms)
        durations_ms: List of durations to generate (e.g., [20, 40, 60, 80, 100, 200, 500, 1000])
        padding_ms: Target duration after padding (default: 2000ms)
        sr: Sample rate (Hz)
        noise_amplitude: Amplitude of padding noise
        seed_base: Base random seed

    Returns:
        Dictionary mapping duration_ms -> padded audio array

    Example:
        >>> audio_1000ms = np.random.randn(16000)  # 1000ms at 16kHz
        >>> variants = create_duration_variants(
        ...     audio_1000ms,
        ...     durations_ms=[20, 40, 60, 80, 100, 200, 500, 1000]
        ... )
        >>> len(variants)
        8
        >>> variants[100].shape  # 100ms segment padded to 2000ms
        (32000,)
    """
    variants = {}

    for duration_ms in durations_ms:
        # Generate unique seed for this duration
        seed = seed_base + hash(duration_ms) % 10000

        # Extract and pad
        variant = slice_and_pad(
            audio_1000ms,
            duration_ms=duration_ms,
            padding_ms=padding_ms,
            sr=sr,
            noise_amplitude=noise_amplitude,
            seed=seed,
        )

        variants[duration_ms] = variant

    return variants
