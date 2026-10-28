"""Generators: physical properties, SNR definition, reproducibility, aliasing guard."""
import math

import numpy as np
import pytest

from signal_lm import Signal, analyze
from signal_lm.data import synthetic as syn

FS = 4096
N = 4096


def test_sine_frequency_amplitude_measured_by_analysis():
    x, events = syn.sine(N, FS, frequency_hz=120, amplitude=2.0)
    m = analyze(Signal(x, FS))["measurements"][0]
    assert m["dominant_frequency_hz"] == pytest.approx(120)
    assert m["dominant_amplitude"] == pytest.approx(2.0, rel=1e-6)
    assert m["ac_rms"] == pytest.approx(2 / math.sqrt(2))
    assert events == []


def test_frequency_above_nyquist_margin_is_rejected_unless_aliasing_requested():
    with pytest.raises(ValueError):
        syn.sine(N, FS, frequency_hz=0.95 * FS / 2)
    x, _ = syn.sine(N, FS, frequency_hz=0.95 * FS / 2, allow_aliasing=True)
    assert np.isfinite(x).all()


def test_harmonics_drops_components_above_nyquist():
    kept = syn.harmonics_kept(FS, fundamental_hz=500, n_harmonics=10)
    assert kept == 3  # 500, 1000, 1500 <= 0.9 * 2048
    x, _ = syn.harmonics(N, FS, fundamental_hz=500, n_harmonics=10, amplitude=1.0, decay=0.5)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(N, 1 / FS)
    assert spectrum[freqs == 2000].max() < 1e-6 * spectrum.max()


def test_chirp_frequency_rises():
    x, _ = syn.chirp(N, FS, f_start_hz=100, f_end_hz=1000)
    first = analyze(Signal(x[: N // 4], FS))["measurements"][0]["dominant_frequency_hz"]
    last = analyze(Signal(x[-N // 4:], FS))["measurements"][0]["dominant_frequency_hz"]
    assert first < 400 < last


def test_amplitude_modulation_envelope_bounds():
    x, _ = syn.amplitude_modulation(N, FS, carrier_hz=500, modulation_hz=5, amplitude=1.0, depth=0.5)
    assert np.max(np.abs(x)) == pytest.approx(1.5, abs=0.02)


def test_impulses_events_and_positions():
    x, events = syn.impulses(N, FS, positions_samples=[10, 2000], amplitude=3.0, width_samples=4, kind="rect")
    assert events == [("impulse", 10, 14), ("impulse", 2000, 2004)]
    assert x[10:14].tolist() == [3.0] * 4 and x[9] == 0 and x[14] == 0
    for kind in ("rect", "triangle", "half_sine"):
        y, _ = syn.impulses(N, FS, positions_samples=[5], amplitude=1.0, width_samples=1, kind=kind)
        assert y[5] == pytest.approx(1.0)


def test_periodic_impulse_positions_spacing():
    positions = syn.periodic_impulse_positions(N, FS, rate_hz=16)
    assert positions[:3] == [0, 256, 512] and len(positions) == 16


def test_damped_oscillation_decays_and_event_bounds():
    x, events = syn.damped_oscillation(N, FS, frequency_hz=300, onset_sample=1000, amplitude=1.0,
                                       decay_rate_per_s=50)
    assert np.all(x[:1000] == 0)
    (label, start, end), = events
    assert label == "damped_burst" and start == 1000
    assert np.max(np.abs(x[end:])) < 0.02


@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0])
def test_colored_noise_slope(exponent):
    rng = np.random.default_rng(0)
    x, _ = syn.colored_noise(1 << 16, FS, rng=rng, exponent=exponent, std=1.0)
    assert np.std(x) == pytest.approx(1.0, rel=1e-6)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(1 << 16, 1 / FS)
    band = (freqs > 20) & (freqs < 1500)
    slope = np.polyfit(np.log(freqs[band]), np.log(spectrum[band]), 1)[0]
    assert slope == pytest.approx(-exponent, abs=0.15)


def test_band_noise_is_band_limited():
    x, _ = syn.band_noise(1 << 14, FS, rng=np.random.default_rng(1), low_hz=500, high_hz=800)
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(1 << 14, 1 / FS)
    assert spectrum[(freqs < 450) | (freqs > 850)].max() < 1e-9 * spectrum.max()


def test_drift_step_and_saturation():
    d, _ = syn.drift(N, FS, slope_per_s=2.0, offset=1.0)
    assert d[0] == 1.0 and d[-1] == pytest.approx(1.0 + 2.0 * (N - 1) / FS)
    s, events = syn.level_step(N, FS, step_sample=100, step_amplitude=-2.0)
    assert s[99] == 0 and s[100] == -2.0 and events == [("level_change", 100, N)]
    clipped, fraction = syn.saturate(np.linspace(-2, 2, 401), 1.0)
    assert clipped.max() == 1.0 and clipped.min() == -1.0 and fraction == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("target", [-10, -5, 0, 5, 10, 20])
def test_snr_definition_is_met_exactly(target):
    clean, _ = syn.sine(N, FS, frequency_hz=200, amplitude=0.3)
    noise, _ = syn.white_noise(N, FS, rng=np.random.default_rng(3), std=5.0)
    scaled = syn.scale_noise_to_snr(clean, noise, target)
    measured = syn.snr_db(clean, scaled)
    assert measured == pytest.approx(target, abs=1e-9)
    assert measured == pytest.approx(10 * math.log10(np.mean(clean ** 2) / np.mean(scaled ** 2)))


def test_active_snr_differs_from_global_for_short_events():
    spec = {"n_samples": N, "sample_rate_hz": FS, "seed": 11,
            "components": [{"type": "impulses", "positions_samples": [2000], "amplitude": 2.0,
                            "width_samples": 40, "kind": "rect"}],
            "noise": {"type": "white_noise", "snr_db": 0.0, "snr_reference": "active"}}
    rendered = syn.render(spec)
    assert rendered["measured"]["snr_db_measured_active"] == pytest.approx(0.0, abs=1e-9)
    # Same energy spread over the full record: global SNR is far lower.
    assert rendered["measured"]["snr_db_measured"] < -15
    assert rendered["events"] == [("impulse", 2000, 2040)]


def test_render_is_reproducible_and_keeps_clean_noise_mix():
    spec = {"n_samples": N, "sample_rate_hz": FS, "seed": 5,
            "components": [{"type": "sine", "frequency_hz": 100, "amplitude": 1.0}],
            "noise": {"type": "colored_noise", "exponent": 1.0, "snr_db": 3.0}}
    a, b = syn.render(spec), syn.render(spec)
    for key in ("clean", "noise", "mix"):
        np.testing.assert_array_equal(a[key], b[key])
        assert a[key].shape == (N, 1)
    np.testing.assert_allclose(a["mix"], a["clean"] + a["noise"])
    assert a["measured"]["snr_db_measured"] == pytest.approx(3.0, abs=1e-9)
    different = syn.render(dict(spec, seed=6))
    assert not np.array_equal(a["noise"], different["noise"])
    np.testing.assert_array_equal(a["clean"], different["clean"])


def test_example_specs_vary_parameters_and_share_group_per_parent():
    config = {"dataset_id": "t", "specialist": "vibration", "sample_rate_hz": FS, "n_samples": N,
              "seed": 1, "n_parents": 3, "snr_db_levels": [0, 10, None],
              "noise": {"type": "white_noise"},
              "recipes": [{"recipe_id": "sine", "task_type": "event_classification", "labels": ["sine"],
                           "components": [{"type": "sine", "frequency_hz": [50, 500], "amplitude": [0.1, 1.0]}]}]}
    specs = syn.build_example_specs(config)
    assert len(specs) == 9
    freqs = {s["components"][0]["frequency_hz"] for s in specs}
    assert len(freqs) == 3 and all(50 <= f <= 500 for f in freqs)
    groups = {s["group_id"] for s in specs}
    assert len(groups) == 3
    assert len({s["record_id"] for s in specs}) == 9
    assert syn.build_example_specs(config) == specs  # deterministic
