import numpy as np
import pytest

from signal_lm import Signal, analyze, load_csv


@pytest.mark.parametrize("n", [1000, 1001])
def test_sine_amplitude_frequency_and_rms(n):
    t = np.arange(n) / n
    result = analyze(Signal(3 + 2 * np.sin(2 * np.pi * 50 * t), n), include_spectrum=True)
    m = result["measurements"][0]
    assert m["dominant_frequency_hz"] == pytest.approx(50)
    assert m["dominant_amplitude"] == pytest.approx(2)
    assert m["ac_rms"] == pytest.approx(np.sqrt(2))
    assert m["rms"] == pytest.approx(np.sqrt(11))
    assert m["mean"] == pytest.approx(3)
    assert result["prediction"]["status"] == "unavailable"


def test_nyquist_amplitude_is_not_doubled():
    result = analyze(Signal(2 * (-1.) ** np.arange(1000), 1000))
    assert result["measurements"][0]["dominant_amplitude"] == pytest.approx(2)
    assert result["measurements"][0]["dominant_frequency_hz"] == 500


def test_dc_has_no_nonzero_dominant_frequency():
    m = analyze(Signal(np.full(32, 4.), 100))["measurements"][0]
    assert m["dominant_frequency_hz"] is None
    assert m["rms"] == 4


def test_multichannel_and_input_not_mutated():
    t = np.arange(1024) / 1024
    x = np.column_stack([np.sin(2*np.pi*30*t), np.sin(2*np.pi*90*t)])
    original = x.copy()
    result = analyze(Signal(x, 1024, ("x", "y"), ("g", "g")))
    assert [m["dominant_frequency_hz"] for m in result["measurements"]] == [30, 90]
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("samples,rate", [([], 100), ([1, 2, 3], 100),
    ([1, 2, np.nan, 4], 100), ([1, 2, 3, 4], 0), ([1, 2, 3, 4], np.inf)])
def test_invalid_inputs(samples, rate):
    with pytest.raises(ValueError):
        Signal(samples, rate)


def test_csv_explicit_column_selection(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("time;x;y\n0;1;9\n1;2;8\n2;3;7\n3;4;6\n", encoding="utf-8")
    s = load_csv(path, sample_rate_hz=1, columns=["y"], delimiter=";")
    np.testing.assert_array_equal(s.samples[:, 0], [9, 8, 7, 6])
    assert s.channels == ("y",)
    with pytest.raises(ValueError):
        load_csv(path, sample_rate_hz=1, columns=["missing"], delimiter=";")


def test_missing_csv_value_fails(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x\n1\n2\nNA\n4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 4"):
        load_csv(path, sample_rate_hz=1, columns=["x"])
