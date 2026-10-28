"""Explicit, idempotent storage of user-labeled audio segments; never trains."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from .audio_file import analyze_file
from .data.schema import read_manifest
from .data.store import add_imported_record

LEARNING_ROOT = Path("data/manual_audio_annotations_v1")
lock = threading.RLock()


def stats(root=None):
    root = Path(root) if root is not None else LEARNING_ROOT
    with lock:
        manifest = root/"manifest.jsonl"
        records = read_manifest(manifest) if manifest.exists() else []
        return {"n_records": len(records), "labels": dict(Counter(label for r in records for label in r.labels)),
                "unassigned": sum(r.split == "unassigned" for r in records),
                "root": str(root), "training_started": False}


def save_annotation(raw, filename, *, zone, window_start, selection_start, selection_end,
                    label, meaning, machine_id="", root=None):
    root = Path(root) if root is not None else LEARNING_ROOT
    values = {"label": label, "meaning": meaning, "zone": zone, "machine_id": machine_id}
    for name, limit in [("label", 80), ("meaning", 1000), ("zone", 80), ("machine_id", 80)]:
        value = values[name]
        if not isinstance(value, str) or len(value.strip()) > limit or (name != "machine_id" and not value.strip()):
            raise ValueError(f"Champ {name} requis, maximum {limit} caractères.")
        values[name] = value.strip()
    start, end = float(selection_start), float(selection_end)
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= 1:
        raise ValueError("Sélection attendue dans la fenêtre affichée : 0 ≤ début < fin ≤ 1 s.")
    _, signal = analyze_file(raw, filename, values["zone"], window_start, return_signal=True)
    duration = len(signal.samples)/signal.sample_rate_hz
    if end > duration + 1e-9:
        raise ValueError("La sélection dépasse la fin du fichier.")
    first, last = round(start*signal.sample_rate_hz), round(end*signal.sample_rate_hz)
    if last-first < 4:
        raise ValueError("Sélectionnez au moins quatre échantillons.")
    original_hash = hashlib.sha256(raw).hexdigest()
    group = "machine:"+values["machine_id"] if values["machine_id"] else "acquisition:"+original_hash
    metadata = {**values, "source_sha256": original_hash, "window_start_s": float(window_start),
                "start_sample": first, "end_sample": last, "source_filename": str(filename)[:240]}
    fingerprint = hashlib.sha256(json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    record_id = "manual_"+fingerprint[:32]
    with lock:
        root.mkdir(parents=True, exist_ok=True)
        manifest = root/"manifest.jsonl"
        records = read_manifest(manifest) if manifest.exists() else []
        for record in records:
            if record.provenance.get("original_sha256") == original_hash and record.group_id != group:
                raise ValueError("Ce fichier est déjà rattaché à un autre groupe. Conservez le même identifiant machine.")
            if record.record_id == record_id:
                if record.provenance.get("annotation_fingerprint") != fingerprint:
                    raise ValueError("Conflit d’identifiant d’annotation.")
                return {"record_id": record_id, "created": False, "split": record.split,
                        "n_samples": record.n_samples, "stats": stats(root), "training_started": False}
        if (root/"records"/f"{record_id}.npz").exists():
            raise FileExistsError("Un segment sans manifeste existe déjà : conservé pour vérification.")
        config = {"dataset_id": "manual_audio_annotations_v1", "seed": None,
                  "origin": "explicit_user_annotations", "review_status": "unverified",
                  "automatic_training": False}
        config_path = root/"config.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Configuration existante incompatible ; aucune donnée remplacée.")
        else:
            with config_path.open("x", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2)
        record = add_imported_record(root, record_id, signal.samples[first:last], source_id="manual_audio",
            specialist="acoustic", task_type="event_classification", sample_rate_hz=signal.sample_rate_hz,
            channel_names=list(signal.channels), units=list(signal.units), group_id=group,
            labels=[values["label"]], machine_id=values["machine_id"] or "unknown",
            acquisition_id=original_hash, license="unknown",
            provenance={"origin": "manual_audio_annotation", "original_sha256": original_hash,
                "original_filename": str(filename)[:240], "original_bytes": len(raw),
                "annotation_fingerprint": fingerprint, "annotation": values,
                "review_status": "unverified_user_annotation", "acquisition_time_utc": None,
                "annotated_at_utc": datetime.now(timezone.utc).isoformat()},
            transforms=[{"op": "ffmpeg_pcm_to_float32", "resampled": False},
                        {"op": "annotated_segment", "source_window_start_s": float(window_start),
                         "start_sample": first, "end_sample": last}],
            notes="User-provided label/meaning, not verified ground truth. Review, split by group and window before training.")
        problems = record.validate()
        if problems:
            raise ValueError("; ".join(problems))
        with manifest.open("a", encoding="utf-8") as stream:
            stream.write(record.to_json()+"\n")
        return {"record_id": record_id, "created": True, "n_samples": record.n_samples,
                "sample_rate_hz": record.sample_rate_hz, "channels": len(record.channel_names),
                "split": "unassigned", "group_id": group, "stats": stats(root), "training_started": False}
