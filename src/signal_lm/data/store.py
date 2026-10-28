"""Dataset on disk: NPZ arrays (samples x channels) + JSONL manifest.

Layout of a dataset root::

    <root>/
      manifest.jsonl          one Record per line
      config.json             the generation / import config that produced it
      records/<record_id>.npz keys: mix (always), clean, noise (synthetic only)

Arrays are stored as float32 in their original physical scale. No
normalisation is applied here; windows keep their amplitude (AGENTS.md).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..signal import Signal
from .schema import EventAnnotation, Record, UNKNOWN, sha256_of_file, write_manifest
from . import synthetic

ARRAY_DTYPE = np.float32


def _as_samples_x_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("Arrays must be [samples, channels].")
    return x


def save_arrays(root: Path, record_id: str, **arrays: np.ndarray) -> Path:
    path = root / "records" / f"{record_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: _as_samples_x_channels(v).astype(ARRAY_DTYPE) for k, v in arrays.items()}
    if "mix" not in payload:
        raise ValueError("A record needs a 'mix' array (the observed signal).")
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    return path


def load_arrays(root: Path, record: Record) -> dict[str, np.ndarray]:
    with np.load(root / record.file_path) as data:
        return {k: data[k].astype(np.float64) for k in data.files}


def load_signal(root: Path, record: Record, key: str = "mix") -> Signal:
    arrays = load_arrays(root, record)
    return Signal(arrays[key], record.sample_rate_hz, tuple(record.channel_names), tuple(record.units))


# --------------------------------------------------------------------------- synthetic dataset

def generate_dataset(config: dict[str, Any], root: str | Path, *, overwrite: bool = False) -> list[Record]:
    """Render every example spec of ``config`` under ``root`` and write the manifest."""
    root = Path(root)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"{root} is not empty; pass overwrite=True to regenerate.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    records = []
    for spec in synthetic.build_example_specs(config):
        rendered = synthetic.render(spec)
        path = save_arrays(root, spec["record_id"], mix=rendered["mix"],
                           clean=rendered["clean"], noise=rendered["noise"])
        generation = {k: spec[k] for k in ("components", "noise", "saturation_limit", "seed",
                                           "recipe_id", "parent_index")}
        generation["generator_version"] = 1
        generation["measured"] = rendered["measured"]
        record = Record(
            record_id=spec["record_id"], source_id=config["dataset_id"],
            specialist=spec["specialist"], task_type=spec["task_type"],
            file_path=str(path.relative_to(root)).replace("\\", "/"),
            sample_rate_hz=spec["sample_rate_hz"], n_samples=spec["n_samples"],
            channel_names=spec["channel_names"], units=spec["units"],
            labels=spec["labels"], machine_id="synthetic", session_id=spec["recipe_id"],
            acquisition_id=spec["group_id"], split="unassigned", is_synthetic=True,
            generation=generation,
            provenance={"origin": "signal_lm.data.synthetic", "config_dataset_id": config["dataset_id"]},
            license="project-internal-synthetic", file_sha256=sha256_of_file(path),
            snr_db_target=spec["noise"]["snr_db"] if spec["noise"] else None,
            snr_db_measured=rendered["measured"]["snr_db_measured"],
            snr_db_measured_active=rendered["measured"]["snr_db_measured_active"],
            events=[EventAnnotation(label, start, end) for label, start, end in rendered["events"]],
            parent_record_id=None, group_id=spec["group_id"],
            notes=config.get("notes", ""),
        )
        records.append(record)
    write_manifest(records, root / "manifest.jsonl")
    return records


def regenerate_record(record: Record) -> dict[str, Any]:
    """Re-render a synthetic record from its stored parameters (for verification)."""
    if not record.is_synthetic:
        raise ValueError("Only synthetic records can be regenerated.")
    g = record.generation
    spec = {"n_samples": record.n_samples, "sample_rate_hz": record.sample_rate_hz,
            "components": g["components"], "noise": g["noise"],
            "saturation_limit": g.get("saturation_limit"), "seed": g["seed"]}
    return synthetic.render(spec)


# --------------------------------------------------------------------------- imported dataset

def add_imported_record(root: Path, record_id: str, samples: np.ndarray, *, source_id: str,
                        specialist: str, task_type: str, sample_rate_hz: float,
                        channel_names: list[str], units: list[str] | None,
                        group_id: str, labels: list[str] | None = None,
                        provenance: dict[str, Any] | None = None, license: str = UNKNOWN,
                        machine_id: str = UNKNOWN, session_id: str = UNKNOWN,
                        acquisition_id: str = UNKNOWN, transforms: list[dict] | None = None,
                        events: list[EventAnnotation] | None = None, notes: str = "") -> Record:
    x = _as_samples_x_channels(samples)
    path = save_arrays(root, record_id, mix=x)
    return Record(
        record_id=record_id, source_id=source_id, specialist=specialist, task_type=task_type,
        file_path=str(path.relative_to(root)).replace("\\", "/"),
        sample_rate_hz=float(sample_rate_hz), n_samples=int(x.shape[0]),
        channel_names=list(channel_names), units=list(units or ["unspecified"] * x.shape[1]),
        labels=list(labels or []), machine_id=machine_id, session_id=session_id,
        acquisition_id=acquisition_id, split="unassigned", is_synthetic=False,
        provenance=provenance or {}, license=license, file_sha256=sha256_of_file(path),
        events=list(events or []), group_id=group_id, transforms=list(transforms or []), notes=notes,
    )
