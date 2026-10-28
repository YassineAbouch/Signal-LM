from datetime import datetime, timezone

import numpy as np
import pytest

from signal_lm import audio


def test_audio_preserves_stereo_units_zone_and_utc():
    rate = 48000
    t = np.arange(rate) / rate
    values = np.column_stack([.5 * np.sin(2 * np.pi * 120 * t),
                              .25 * np.sin(2 * np.pi * 240 * t)])
    end = datetime(2026, 10, 28, 12, 0, 1, tzinfo=timezone.utc)
    result = audio.analyze_audio(values, rate, "zone_01", "test", end, 48, 0)
    assert [m["dominant_frequency_hz"] for m in result["measurements"]] == [120, 240]
    assert result["measurements"][0]["rms"] == pytest.approx(.5 / np.sqrt(2))
    assert result["measurements"][0]["unit"] == "FS"
    assert result["capture"]["window_start_utc"] == "2026-10-28T12:00:00+00:00"
    assert result["capture"]["window_end_utc"] == end.isoformat()
    assert result["capture"]["zone"] == "zone_01"
    assert result["capture"]["persisted"] is False
    assert result["prediction"]["status"] == "unavailable"


class FakeStream:
    def start_stream(self):
        pass

    def close(self):
        self.closed = True


class FakePA:
    callback = None

    def get_device_info_by_index(self, index):
        return {"isLoopbackDevice": True, "defaultSampleRate": 4096,
                "maxInputChannels": 1, "name": "synthetic-output"}

    def open(self, **kwargs):
        self.callback = kwargs["stream_callback"]
        self.stream = FakeStream()
        return self.stream

    def terminate(self):
        self.terminated = True


def test_capture_lifecycle_ring_buffer_and_discontinuities(monkeypatch):
    pa = FakePA()
    class Backend:
        paFloat32 = 1
        paContinue = 0
        PyAudio = lambda self: pa
    monkeypatch.setattr(audio, "backend", lambda: Backend())
    capture = audio.AudioCapture()
    try:
        info = capture.start(1, "test-zone")
        token = info["session"]
        assert capture.frame(token)["waiting"]
        with pytest.raises(RuntimeError):
            capture.start(1, "other-zone")
        with pytest.raises(ValueError):
            capture.stop("another-tab")
        t = np.arange(4096) / 4096
        raw = np.sin(2 * np.pi * 120 * t).astype(np.float32).tobytes()
        for _ in range(3):
            pa.callback(raw, 4096, {}, 0)
        assert capture.frames == 4096
        assert capture.frame(token)["measurements"][0]["dominant_frequency_hz"] == 120
        pa.callback(np.zeros(1024, dtype=np.float32).tobytes(), 1024, {}, 1)
        assert capture.discontinuities == 1
        assert capture.frame(token)["waiting"]
        capture.stop(token)
        assert pa.stream.closed and pa.terminated
        assert not capture.blocks and capture.session is None
        with pytest.raises(ValueError):
            capture.frame(token)
        token = capture.start(1, "new-session")["session"]
        capture.heartbeat -= 16
        capture.expire_idle()
        assert capture.session is None
        assert not capture.blocks
    finally:
        capture.close()


@pytest.mark.parametrize("zone", ["", " ", "x" * 81, None])
def test_capture_rejects_invalid_zone_without_opening_device(zone):
    capture = audio.AudioCapture()
    with pytest.raises(ValueError):
        capture.start(1, zone)
    assert capture.pa is None
