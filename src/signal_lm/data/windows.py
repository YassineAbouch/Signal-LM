"""Fixed-length windowing of parent records, after splitting.

Windows are new records whose ``parent_record_id`` points to the source
record; they inherit ``split`` and ``group_id`` and carry a ``transforms``
entry describing the cut. Amplitude is never rescaled. Temporal events are
clipped to the window and relabelled relative to its start; the window's
``labels`` reflect which events overlap it (plus ``no_event`` for
event_detection windows that contain none).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import EventAnnotation, Record, sha256_of_file
from .store import load_arrays, save_arrays


def window_record(root: Path, record: Record, *, window_samples: int, hop_samples: int,
                  min_event_overlap: float = 0.5, drop_partial: bool = True) -> list[Record]:
    if record.split == "unassigned":
        raise ValueError(f"{record.record_id}: assign splits before windowing.")
    if window_samples < 4 or hop_samples < 1:
        raise ValueError("window_samples >= 4 and hop_samples >= 1 required.")
    arrays = load_arrays(root, record)
    n = record.n_samples
    windows = []
    index = 0
    for start in range(0, n, hop_samples):
        end = start + window_samples
        if end > n:
            if drop_partial or start == 0:
                break
            end = n
        cut = {k: v[start:end] for k, v in arrays.items()}
        events = []
        for ev in record.events:
            overlap = min(ev.end_sample, end) - max(ev.start_sample, start)
            if overlap <= 0:
                continue
            duration = ev.end_sample - ev.start_sample
            if overlap / duration < min_event_overlap and overlap < window_samples:
                continue
            events.append(EventAnnotation(ev.label, max(ev.start_sample, start) - start,
                                          min(ev.end_sample, end) - start))
        if record.task_type == "event_detection":
            labels = sorted({e.label for e in events}) or ["no_event"]
        else:
            labels = list(record.labels)
        wid = f"{record.record_id}__w{index:04d}"
        path = save_arrays(root, wid, **cut)
        windows.append(Record(
            record_id=wid, source_id=record.source_id, specialist=record.specialist,
            task_type=record.task_type, file_path=str(path.relative_to(root)).replace("\\", "/"),
            sample_rate_hz=record.sample_rate_hz, n_samples=end - start,
            channel_names=list(record.channel_names), units=list(record.units), labels=labels,
            machine_id=record.machine_id, session_id=record.session_id,
            acquisition_id=record.acquisition_id, split=record.split, is_synthetic=record.is_synthetic,
            generation={"inherited_from": record.record_id, "seed": record.generation.get("seed"),
                        "saturation_limit": record.generation.get("saturation_limit")}
            if record.is_synthetic else {},
            provenance=dict(record.provenance), license=record.license,
            file_sha256=sha256_of_file(path), snr_db_target=record.snr_db_target,
            snr_db_measured=_window_snr(cut), snr_db_measured_active=None,
            events=events, parent_record_id=record.record_id, group_id=record.group_id,
            transforms=list(record.transforms) + [{"op": "window", "start_sample": start,
                                                   "end_sample": end, "hop_samples": hop_samples}],
            notes=record.notes,
        ))
        index += 1
    return windows


def _window_snr(cut: dict[str, np.ndarray]) -> float | None:
    if "clean" not in cut or "noise" not in cut:
        return None
    p_s, p_n = float(np.mean(cut["clean"] ** 2)), float(np.mean(cut["noise"] ** 2))
    if p_s <= 0 or p_n <= 0:
        return None
    return float(10 * np.log10(p_s / p_n))
