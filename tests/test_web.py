import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest
import numpy as np

from signal_lm.web import Handler, analyze_request


def test_web_demo_physical_measurements():
    result = analyze_request({"frequency": 120, "amplitude": 2, "sample_rate": 4096})
    measurement = result["measurements"][0]
    assert measurement["dominant_frequency_hz"] == 120
    assert measurement["rms"] == pytest.approx(2**0.5)
    assert result["prediction"]["status"] == "unavailable"
    assert result["sample_count"] == 4096
    assert len(result["waveform"]["time_s"]) == 1800


def test_web_csv_explicit_channels_and_units():
    result = analyze_request({"source": "csv", "sample_rate": 100,
        "csv": "time;x;y\n0;1;4\n1;2;3\n2;3;2\n3;4;1\n",
        "delimiter": ";", "columns": ["y"], "units": ["g"]})
    assert result["measurements"][0]["channel"] == "y"
    assert result["measurements"][0]["unit"] == "g"
    assert result["waveform"]["values"] == [[4, 3, 2, 1]]


def test_composite_signal_preserves_sum_rms_and_spectral_amplitudes():
    components = [{"frequency": 50, "amplitude": 2, "phase_deg": 0},
                  {"frequency": 120, "amplitude": 1, "phase_deg": 90}]
    result = analyze_request({"components": components, "sample_rate": 4096})
    assert result["measurements"][0]["rms"] == pytest.approx(np.sqrt(2.5))
    assert result["measurements"][0]["dominant_frequency_hz"] == 50
    spectrum = result["spectrum"]["amplitude_by_channel"][0]
    assert spectrum[50] == pytest.approx(2)
    assert spectrum[120] == pytest.approx(1)
    waveform = result["waveform"]
    np.testing.assert_allclose(np.sum(waveform["components"], axis=0), waveform["values"][0])
    assert waveform["components"][1][0] == pytest.approx(1)
    assert result["synthetic"]["components"] == components
    assert result["prediction"]["status"] == "unavailable"


def test_composite_opposite_phases_cancel_without_normalization():
    result = analyze_request({"components": [
        {"frequency": 50, "amplitude": 2, "phase_deg": 90},
        {"frequency": 50, "amplitude": 2, "phase_deg": -90}]})
    assert result["measurements"][0]["rms"] == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("components", [[], [{}] * 17, [None],
    [{"frequency": 2048}], [{"amplitude": 0}], [{"phase_deg": float("nan")}],
    [{"phase_deg": 361}]])
def test_composite_rejects_invalid_components(components):
    with pytest.raises(ValueError):
        analyze_request({"components": components})


@pytest.mark.parametrize("payload", [
    {"frequency": 2048}, {"sample_rate": 0}, {"amplitude": float("nan")},
    {"source": "csv", "csv": "x\n1\n2\nNaN\n4", "columns": ["x"]},
    {"source": "csv", "csv": "x\n1\n2\n3\n4", "columns": ["time"]},
])
def test_web_rejects_invalid_measurements(payload):
    with pytest.raises(ValueError):
        analyze_request(payload)


def test_http_routes_errors_and_origin():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        client.request("GET", "/")
        response = client.getresponse()
        assert response.status == 200
        assert b"Signal-LM" in response.read()
        client.request("GET", "/model")
        response = client.getresponse()
        assert response.status == 200
        assert "Architecture du réseau" in response.read().decode()
        client.request("GET", "/model.js")
        response = client.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type").startswith("text/javascript")
        response.read()
        client.request("GET", "/api/doctor")
        response = client.getresponse()
        assert response.status == 200
        details = json.loads(response.read())
        assert details["status"] == "untrained"
        assert details["training_steps"] == 0
        assert details["parameters"] == 22304
        assert details["config"]["widths"] == [16, 32, 64]
        assert details["weight_bytes"] == 89216
        assert details["receptive_field_samples"] == 19
        assert len(details["architecture"]) == 3
        assert details["model_source"] == "fresh_default_initialization"
        client.request("POST", "/api/analyze", json.dumps({"frequency": 120}),
                       {"Content-Type": "application/json"})
        response = client.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["measurements"][0]["dominant_frequency_hz"] == 120
        client.request("POST", "/api/analyze", "{}", {"Origin": "https://example.com"})
        response = client.getresponse()
        assert response.status == 403
        response.read()
        client.request("POST", "/api/analyze", "not-json")
        response = client.getresponse()
        assert response.status == 400
        response.read()
        client.request("GET", "/../model.py")
        response = client.getresponse()
        assert response.status == 404
        response.read()
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()
