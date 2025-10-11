"""
White noise / SNR sweep module.

Adds calibrated white noise to audio segments at target SNR levels.
Computes RMS over the effective segment (excludes padding) to avoid inflating SNR.
"""

import numpy as np


def compute_rms(audio: np.ndarray) -> float:
    """
    Compute RMS energy of audio signal.

    Args:
        audio: Audio array (samples,)

    Returns:
        RMS value
    """
    return np.sqrt(np.mean(audio ** 2))


def add_white_noise(
    audio: np.ndarray,
    snr_db: float,
    effective_start: int = 0,
    effective_end: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Mix white noise into audio at target SNR.

    **Important**: SNR is computed relative to the *effective segment*
    (audio[effective_start:effective_end]), but noise is applied to the
    entire container to avoid giving "cues" about segment boundaries.

    Args:
        audio: Input audio array (samples,). Can be padded container.
        snr_db: Target SNR in dB. Positive = signal louder than noise.
        effective_start: Start index of the effective segment (default 0).
        effective_end: End index of the effective segment (default None = end).
        seed: Random seed for noise generation (reproducibility).

    Returns:
        (mixed_audio, meta_dict)
        - mixed_audio: Audio with noise added
        - meta_dict: {"snr_db", "rms_signal", "rms_noise", "seed"}
    """
    if effective_end is None:
        effective_end = len(audio)

    # Compute RMS of effective segment only
    effective_segment = audio[effective_start:effective_end]
    rms_signal = compute_rms(effective_segment)

    # Generate noise for the entire container
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(audio)).astype(audio.dtype)

    if rms_signal < 1e-8:
        # Silent segment: use minimal noise level instead of SNR-based mixing
        import warnings
        warnings.warn(f"Effective segment has near-zero RMS ({rms_signal:.2e}); using minimal noise")

        target_rms_noise = 1e-4  # Minimal noise level
        current_rms_noise = compute_rms(noise)
        noise = noise * (target_rms_noise / current_rms_noise)

        meta = {
            "snr_db": None,  # SNR undefined for silent segments
            "rms_signal": float(rms_signal),
            "rms_noise": float(target_rms_noise),
            "seed": seed,
            "silent_segment": True,
        }
        return audio + noise, meta

    # Scale noise to achieve target SNR
    # SNR_dB = 20*log10(RMS_signal / RMS_noise)
    # => RMS_noise = RMS_signal / 10^(SNR_dB/20)
    target_rms_noise = rms_signal / (10 ** (snr_db / 20.0))
    current_rms_noise = compute_rms(noise)
    noise = noise * (target_rms_noise / current_rms_noise)

    # Mix
    mixed = audio + noise

    meta = {
        "snr_db": snr_db,
        "rms_signal": float(rms_signal),
        "rms_noise": float(target_rms_noise),
        "seed": seed,
    }

    return mixed, meta


def mix_at_snr(
    audio: np.ndarray,
    snr_db: float,
    sr: int,
    padding_ms: int = 2000,
    effective_dur_ms: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    High-level convenience wrapper for adding noise to padded containers.

    Assumes the effective segment is centered in a padded container of
    total duration = padding_ms. If effective_dur_ms is given, computes
    the effective region; else assumes the entire audio is effective.

    Args:
        audio: Padded audio array (samples,).
        snr_db: Target SNR in dB.
        sr: Sample rate (Hz).
        padding_ms: Total duration of the container in ms (default 2000).
        effective_dur_ms: Duration of the effective segment in ms (optional).
        seed: Random seed.

    Returns:
        (mixed_audio, meta_dict)
    """
    total_samples = len(audio)
    expected_samples = int(sr * padding_ms / 1000.0)

    if total_samples != expected_samples:
        raise ValueError(
            f"Audio length {total_samples} does not match expected "
            f"{expected_samples} for padding_ms={padding_ms}, sr={sr}."
        )

    if effective_dur_ms is not None:
        # Compute effective region (centered)
        effective_samples = int(sr * effective_dur_ms / 1000.0)
        effective_start = (total_samples - effective_samples) // 2
        effective_end = effective_start + effective_samples
    else:
        # Entire audio is effective
        effective_start = 0
        effective_end = total_samples

    return add_white_noise(
        audio,
        snr_db=snr_db,
        effective_start=effective_start,
        effective_end=effective_end,
        seed=seed,
    )
