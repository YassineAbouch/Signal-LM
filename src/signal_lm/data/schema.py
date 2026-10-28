"""Manifest schema: one JSONL line per recording, samples x channels on disk.

Nothing here interprets a signal. A record only states what a file is, where
it comes from, what is known about it and what is *not* known. Unknown
metadata is written explicitly as ``UNKNOWN`` rather than guessed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

MANIFEST_SCHEMA_VERSION = 1
UNKNOWN = "unknown"

SPECIALISTS = ("vibration", "acoustic", "music", "slow_series")
TASK_TYPES = ("event_detection", "event_classification",
              "anomaly_detection", "multilabel_classification")
SPLITS = ("train", "validation", "test", "unassigned")


@dataclass(frozen=True)
class EventAnnotation:
    """Temporal annotation in samples, [start, end) relative to the record."""
    label: str
    start_sample: int
    end_sample: int

    def validate(self, n_samples: int) -> None:
        if not isinstance(self.start_sample, int) or not isinstance(self.end_sample, int):
            raise ValueError("Event bounds must be integers (samples).")
        if not 0 <= self.start_sample < self.end_sample <= n_samples:
            raise ValueError(f"Event {self.label!r} bounds [{self.start_sample}, "
                             f"{self.end_sample}) outside [0, {n_samples}).")


@dataclass
class Record:
    record_id: str
    source_id: str
    specialist: str
    task_type: str
    file_path: str                 # relative to the dataset root
    sample_rate_hz: float
    n_samples: int
    channel_names: list[str]
    units: list[str]               # "unspecified" when not calibrated (SPEC_V1)
    labels: list[str] = field(default_factory=list)
    machine_id: str = UNKNOWN
    session_id: str = UNKNOWN
    acquisition_id: str = UNKNOWN
    split: str = "unassigned"
    is_synthetic: bool = False
    generation: dict[str, Any] = field(default_factory=dict)   # params + seed
    provenance: dict[str, Any] = field(default_factory=dict)   # origin, url, version
    license: str = UNKNOWN
    file_sha256: str = UNKNOWN
    snr_db_target: float | None = None
    snr_db_measured: float | None = None
    snr_db_measured_active: float | None = None
    events: list[EventAnnotation] = field(default_factory=list)
    parent_record_id: str | None = None
    group_id: str = UNKNOWN        # unit kept together across splits
    transforms: list[dict[str, Any]] = field(default_factory=list)  # traceable ops
    notes: str = ""
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def validate(self) -> list[str]:
        """Return a list of problems (empty when the record is well-formed)."""
        problems = []
        if not self.record_id:
            problems.append("record_id is empty")
        if self.specialist not in SPECIALISTS:
            problems.append(f"specialist {self.specialist!r} not in {SPECIALISTS}")
        if self.task_type not in TASK_TYPES:
            problems.append(f"task_type {self.task_type!r} not in {TASK_TYPES}")
        if self.split not in SPLITS:
            problems.append(f"split {self.split!r} not in {SPLITS}")
        if not (isinstance(self.sample_rate_hz, (int, float)) and self.sample_rate_hz > 0):
            problems.append("sample_rate_hz must be > 0")
        if not (isinstance(self.n_samples, int) and self.n_samples >= 4):
            problems.append("n_samples must be an integer >= 4")
        if not self.channel_names:
            problems.append("channel_names is empty")
        if len(self.channel_names) != len(self.units):
            problems.append("units must match channel_names length")
        if len(set(self.channel_names)) != len(self.channel_names):
            problems.append("channel_names must be unique")
        if self.is_synthetic and "seed" not in self.generation:
            problems.append("synthetic record without generation.seed")
        if self.group_id == UNKNOWN:
            problems.append("group_id unknown: cannot split without leakage")
        for ev in self.events:
            try:
                ev.validate(self.n_samples)
            except ValueError as exc:
                problems.append(str(exc))
        if self.task_type == "event_detection" and not self.events and "no_event" not in self.labels:
            problems.append("event_detection record has no events and no 'no_event' label")
        return problems

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Record":
        data = dict(data)
        data["events"] = [EventAnnotation(**e) if isinstance(e, dict) else e
                          for e in data.get("events", [])]
        return cls(**data)


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(records: Iterable[Record], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.to_json() + "\n")
            count += 1
    return count


def read_manifest(path: str | Path) -> list[Record]:
    return list(iter_manifest(path))


def iter_manifest(path: str | Path) -> Iterator[Record]:
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Record.from_dict(json.loads(line))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid manifest line {line_no}: {exc}") from exc
