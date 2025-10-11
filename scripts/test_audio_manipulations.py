#!/usr/bin/env python3
"""
Test audio manipulation modules (noise, filters, reverb).

Validates that SNR, band-limiting, and RIR convolution produce expected results.

Usage:
    python scripts/test_audio_manipulations.py
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import logging

from qsm.audio import (
    mix_at_snr,
    apply_bandpass,
    apply_lowpass,
    apply_highpass,
    apply_rir,
    load_rir_database,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_snr_mixing():
    """Test SNR mixing produces correct SNR."""
    logger.info("Testing SNR mixing...")

    sr = 16000
    duration_s = 0.1  # 100 ms effective
    padding_ms = 2000
    total_samples = int(sr * padding_ms / 1000.0)
    effective_samples = int(sr * duration_s)

    # Create test signal (sine wave)
    t = np.linspace(0, duration_s, effective_samples)
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Pad to 2000 ms
    audio = np.zeros(total_samples, dtype=np.float32)
    start_idx = (total_samples - effective_samples) // 2
    audio[start_idx : start_idx + effective_samples] = signal

    # Mix at SNR = 0 dB
    target_snr = 0.0
    noisy, meta = mix_at_snr(
        audio,
        snr_db=target_snr,
        sr=sr,
        padding_ms=padding_ms,
        effective_dur_ms=duration_s * 1000,
        seed=42,
    )

    # Verify SNR
    rms_signal = meta["rms_signal"]
    rms_noise = meta["rms_noise"]
    actual_snr = 20 * np.log10(rms_signal / rms_noise)

    logger.info(f"  Target SNR: {target_snr:.2f} dB")
    logger.info(f"  Actual SNR: {actual_snr:.2f} dB")
    logger.info(f"  RMS signal: {rms_signal:.6f}")
    logger.info(f"  RMS noise: {rms_noise:.6f}")

    assert abs(actual_snr - target_snr) < 0.1, f"SNR mismatch: {actual_snr:.2f} != {target_snr:.2f}"
    logger.info("✓ SNR test passed")


def test_band_filters():
    """Test band-pass filtering."""
    logger.info("Testing band-pass filters...")

    sr = 16000
    duration_s = 1.0
    t = np.linspace(0, duration_s, int(sr * duration_s))

    # Create multi-frequency signal
    # 200 Hz (below telephony), 1000 Hz (in band), 5000 Hz (above telephony)
    signal = (
        np.sin(2 * np.pi * 200 * t) +
        np.sin(2 * np.pi * 1000 * t) +
        np.sin(2 * np.pi * 5000 * t)
    ).astype(np.float32)

    # Apply telephony filter (300-3400 Hz)
    filtered = apply_bandpass(signal, sr, lowcut=300.0, highcut=3400.0)

    # Compute FFT
    fft_orig = np.fft.rfft(signal)
    fft_filt = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(signal), 1/sr)

    # Check attenuation
    def get_power_at_freq(fft, freqs, target_freq, bandwidth=50):
        idx = np.argmin(np.abs(freqs - target_freq))
        return np.abs(fft[idx]) ** 2

    power_200_orig = get_power_at_freq(fft_orig, freqs, 200)
    power_200_filt = get_power_at_freq(fft_filt, freqs, 200)
    power_1000_orig = get_power_at_freq(fft_orig, freqs, 1000)
    power_1000_filt = get_power_at_freq(fft_filt, freqs, 1000)
    power_5000_orig = get_power_at_freq(fft_orig, freqs, 5000)
    power_5000_filt = get_power_at_freq(fft_filt, freqs, 5000)

    attenuation_200 = 10 * np.log10(power_200_filt / power_200_orig) if power_200_orig > 0 else -np.inf
    attenuation_1000 = 10 * np.log10(power_1000_filt / power_1000_orig) if power_1000_orig > 0 else -np.inf
    attenuation_5000 = 10 * np.log10(power_5000_filt / power_5000_orig) if power_5000_orig > 0 else -np.inf

    logger.info(f"  200 Hz attenuation: {attenuation_200:.2f} dB (expect < -20 dB)")
    logger.info(f"  1000 Hz attenuation: {attenuation_1000:.2f} dB (expect ~ 0 dB)")
    logger.info(f"  5000 Hz attenuation: {attenuation_5000:.2f} dB (expect < -20 dB)")

    assert attenuation_200 < -10, "200 Hz not attenuated enough"
    assert attenuation_1000 > -3, "1000 Hz attenuated too much"
    assert attenuation_5000 < -10, "5000 Hz not attenuated enough"

    logger.info("✓ Band filter test passed")


def test_rir_convolution():
    """Test RIR convolution."""
    logger.info("Testing RIR convolution...")

    sr = 16000
    duration_s = 0.5

    # Create impulse
    audio = np.zeros(int(sr * duration_s), dtype=np.float32)
    audio[0] = 1.0

    # Create synthetic RIR (exponential decay)
    t60 = 0.5  # 500 ms
    rir_duration_s = 1.0
    rir_samples = int(sr * rir_duration_s)
    t = np.arange(rir_samples) / sr
    rir = np.exp(-6.91 * t / t60).astype(np.float32)  # -60 dB at T60

    # Apply RIR
    reverb = apply_rir(audio, rir, normalize=False)

    # Check that reverb is longer (has tail)
    assert len(reverb) == len(audio), "RIR should truncate to input length"
    assert np.max(np.abs(reverb[len(audio)//2:])) > 0, "No reverb tail detected"

    logger.info(f"  RIR length: {len(rir)} samples ({len(rir)/sr:.2f} s)")
    logger.info(f"  Output length: {len(reverb)} samples")
    logger.info(f"  Max amplitude: {np.max(np.abs(reverb)):.4f}")

    logger.info("✓ RIR convolution test passed")


def main():
    logger.info("=" * 60)
    logger.info("Audio Manipulation Module Tests")
    logger.info("=" * 60)

    test_snr_mixing()
    test_band_filters()
    test_rir_convolution()

    logger.info("=" * 60)
    logger.info("All tests passed! ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
