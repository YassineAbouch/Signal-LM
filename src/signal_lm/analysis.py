"""Deterministic physical measurements, independent of neural weights."""
import numpy as np
from .signal import Signal


def analyze(signal: Signal, *, include_spectrum: bool = False) -> dict:
    x = signal.samples
    n = len(x)
    means = x.mean(axis=0)
    centered = x - means
    # Periodic Hann window; coherent-gain-corrected one-sided amplitude.
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    magnitude = np.abs(np.fft.rfft(centered * window[:, None], axis=0))
    amplitude = magnitude / window.sum()
    amplitude[1:-1 if n % 2 == 0 else None] *= 2
    frequencies = np.fft.rfftfreq(n, 1 / signal.sample_rate_hz)
    measurements = []
    for ch, name in enumerate(signal.channels):
        ac_rms = float(np.sqrt(np.mean(centered[:, ch] ** 2)))
        # Locate peaks before one-sided scaling: scaling adjacent bins but not
        # Nyquist can otherwise move a Nyquist peak to its Hann side lobe.
        peak_bin = int(np.argmax(magnitude[1:, ch])) + 1
        measurements.append({
            "channel": name, "unit": signal.units[ch],
            "mean": float(means[ch]),
            "rms": float(np.sqrt(np.mean(x[:, ch] ** 2))),
            "ac_rms": ac_rms,
            "peak_abs": float(np.max(np.abs(x[:, ch]))),
            "peak_to_peak": float(np.ptp(x[:, ch])),
            "dominant_frequency_hz": float(frequencies[peak_bin]) if ac_rms > 0 else None,
            "dominant_amplitude": float(amplitude[peak_bin, ch]) if ac_rms > 0 else 0.0,
        })
    result = {
        "schema_version": 1,
        "sample_rate_hz": signal.sample_rate_hz,
        "sample_count": n,
        "window_duration_s": n / signal.sample_rate_hz,
        "frequency_bin_spacing_hz": signal.sample_rate_hz / n,
        "spectral_method": "mean_removed_periodic_hann_one_sided_amplitude",
        "measurements": measurements,
        "prediction": {"status": "unavailable", "reason": "no_trained_specialist"},
    }
    if include_spectrum:
        result["spectrum"] = {"frequency_hz": frequencies.tolist(),
                              "amplitude_by_channel": amplitude.T.tolist()}
    return result
