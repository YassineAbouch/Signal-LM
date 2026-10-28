import io
import shutil
import subprocess
import wave

import numpy as np
import pytest

from signal_lm.audio_file import analyze_file

requires_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                    reason="FFmpeg/ffprobe unavailable")


def wav_bytes():
    rate = 4096
    t = np.arange(rate) / rate
    # Two different windows, stereo with independent amplitudes.
    x = np.concatenate([np.sin(2*np.pi*50*t), np.sin(2*np.pi*120*t)])
    samples = (np.column_stack([.5*x, .25*x]) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(samples.tobytes())
    return output.getvalue()


@requires_ffmpeg
def test_wav_window_offset_and_physical_properties():
    result = analyze_file(wav_bytes(), "test.wav", "zone_audio", 1)
    assert result["sample_rate_hz"] == 4096
    assert result["sample_count"] == 4096
    assert len(result["measurements"]) == 2
    assert [m["dominant_frequency_hz"] for m in result["measurements"]] == [120, 120]
    assert result["measurements"][0]["rms"] == pytest.approx(.5 / np.sqrt(2), abs=5e-5)
    assert result["measurements"][1]["dominant_amplitude"] == pytest.approx(.25, abs=5e-5)
    assert result["audio_file"]["window_start_s"] == 1
    assert result["audio_file"]["window_end_s"] == 2
    assert result["audio_file"]["acquisition_time_utc"] is None
    assert result["audio_file"]["persisted"] is False
    assert "capture" not in result
    assert result["prediction"]["status"] == "unavailable"


@requires_ffmpeg
def test_avi_audio_track_is_decoded():
    avi = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", "pipe:0",
        "-c:a", "pcm_s16le", "-f", "avi", "pipe:1"], input=wav_bytes(),
        capture_output=True, check=True, timeout=15).stdout
    result = analyze_file(avi, "test.avi", "zone_01")
    assert result["measurements"][0]["dominant_frequency_hz"] == 50
    assert result["audio_file"]["name"] == "test.avi"


@requires_ffmpeg
def test_bad_media_and_position_after_end_are_rejected():
    with pytest.raises(ValueError):
        analyze_file(b"not audio", "bad.wav", "zone_01")
    with pytest.raises(ValueError, match="position"):
        analyze_file(wav_bytes(), "test.wav", "zone_01", 10)


@pytest.mark.parametrize("start", [-1, float("nan"), 86401])
def test_invalid_position_rejected(start):
    with pytest.raises(ValueError):
        analyze_file(b"audio", "test.wav", "zone_01", start)


def test_missing_decoder_error_is_actionable(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="FFmpeg"):
        analyze_file(b"audio", "test.wav", "zone_01")
