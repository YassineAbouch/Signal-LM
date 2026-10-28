"""Canonical input: samples x channels, regular sampling, physical units."""
from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Signal:
    samples: np.ndarray
    sample_rate_hz: float
    channels: tuple[str, ...] = ()
    units: tuple[str, ...] = ()

    def __post_init__(self):
        x = np.array(self.samples, dtype=np.float64, copy=True)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or x.shape[0] < 4 or x.shape[1] < 1:
            raise ValueError("Expected at least 4 samples, shaped [samples, channels].")
        if not np.isfinite(x).all():
            raise ValueError("Samples must be finite; missing values need explicit handling.")
        rate = float(self.sample_rate_hz)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("sample_rate_hz must be positive and finite.")
        channels = tuple(self.channels) or tuple(f"ch_{i}" for i in range(x.shape[1]))
        units = tuple(self.units) or ("unspecified",) * x.shape[1]
        if len(channels) != x.shape[1] or len(units) != x.shape[1]:
            raise ValueError("Channel names and units must match channel count.")
        if len(set(channels)) != len(channels):
            raise ValueError("Channel names must be unique.")
        x.setflags(write=False)
        object.__setattr__(self, "samples", x)
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "units", units)


def load_csv(path: str | Path, *, sample_rate_hz: float,
             columns: list[str], units: list[str] | None = None,
             delimiter: str = ",") -> Signal:
    """Read explicit numeric columns; never guess that time is a signal channel."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Select at least one unique signal column.")
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        if not reader.fieldnames or any(c not in reader.fieldnames for c in columns):
            raise ValueError("Selected columns not found in CSV header.")
        rows = []
        for line, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[c]) for c in columns])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value on CSV line {line}.") from exc
    return Signal(rows, sample_rate_hz, tuple(columns), tuple(units or ()))
