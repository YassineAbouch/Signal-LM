"""Import real recordings (CSV, WAV) into a dataset root with a manifest.

* CSV goes through ``signal_lm.load_csv`` (explicit columns, explicit rate).
* WAV is read with the standard library: PCM 8/16/24/32-bit and float32.
  Integer samples are divided by 2**(bits-1) so values lie in [-1, 1); the
  unit is recorded as ``unspecified`` because a WAV is not calibrated to a
  physical quantity. That scaling is recorded in ``transforms``.
* No resampling. Original files are never modified; the import records the
  original path, size and hash in ``provenance``.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from ..signal import load_csv
from .schema import Record, UNKNOWN, sha256_of_file
from .store import add_imported_record


def read_wav(path: str | Path) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Return (samples[n, channels] float64 in [-1, 1), sample_rate_hz, info)."""
    with wave.open(str(path), "rb") as stream:
        n_channels = stream.getnchannels()
        width = stream.getsampwidth()
        rate = stream.getframerate()
        n_frames = stream.getnframes()
        raw = stream.readframes(n_frames)
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128) / 128.0
        bits = 8
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
        bits = 16
    elif width == 3:
        buffer = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (buffer[:, 0].astype(np.int32) | (buffer[:, 1].astype(np.int32) << 8)
                | (buffer[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        data = ints.astype(np.float64) / float(1 << 23)
        bits = 24
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
        bits = 32
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes.")
    samples = data.reshape(-1, n_channels)
    info = {"bits_per_sample": bits, "n_channels": n_channels, "n_frames": n_frames,
            "scaling": f"int / 2**{bits - 1}", "sample_rate_hz": rate}
    return samples, float(rate), info


def import_wav(root: Path, path: str | Path, *, record_id: str, source_id: str, specialist: str,
               task_type: str, group_id: str, labels: list[str] | None = None,
               channel_names: list[str] | None = None, license: str = UNKNOWN,
               machine_id: str = UNKNOWN, session_id: str = UNKNOWN,
               acquisition_id: str = UNKNOWN, extra_provenance: dict[str, Any] | None = None,
               notes: str = "") -> Record:
    path = Path(path)
    samples, rate, info = read_wav(path)
    names = channel_names or [f"wav_ch_{i}" for i in range(samples.shape[1])]
    provenance = {"origin": "wav", "original_path": str(path), "original_sha256": sha256_of_file(path),
                  "original_bytes": path.stat().st_size, **info, **(extra_provenance or {})}
    return add_imported_record(
        root, record_id, samples, source_id=source_id, specialist=specialist, task_type=task_type,
        sample_rate_hz=rate, channel_names=names, units=["unspecified"] * samples.shape[1],
        group_id=group_id, labels=labels, provenance=provenance, license=license,
        machine_id=machine_id, session_id=session_id, acquisition_id=acquisition_id,
        transforms=[{"op": "wav_pcm_to_float", "scaling": info["scaling"]}], notes=notes)


def import_csv(root: Path, path: str | Path, *, record_id: str, source_id: str, specialist: str,
               task_type: str, group_id: str, sample_rate_hz: float, columns: list[str],
               units: list[str] | None = None, delimiter: str = ",", labels: list[str] | None = None,
               license: str = UNKNOWN, machine_id: str = UNKNOWN, session_id: str = UNKNOWN,
               acquisition_id: str = UNKNOWN, extra_provenance: dict[str, Any] | None = None,
               notes: str = "") -> Record:
    path = Path(path)
    signal = load_csv(path, sample_rate_hz=sample_rate_hz, columns=columns, units=units, delimiter=delimiter)
    provenance = {"origin": "csv", "original_path": str(path), "original_sha256": sha256_of_file(path),
                  "original_bytes": path.stat().st_size, "columns": list(columns),
                  "delimiter": delimiter, **(extra_provenance or {})}
    return add_imported_record(
        root, record_id, signal.samples, source_id=source_id, specialist=specialist, task_type=task_type,
        sample_rate_hz=signal.sample_rate_hz, channel_names=list(signal.channels),
        units=list(signal.units), group_id=group_id, labels=labels, provenance=provenance,
        license=license, machine_id=machine_id, session_id=session_id, acquisition_id=acquisition_id,
        transforms=[], notes=notes)
