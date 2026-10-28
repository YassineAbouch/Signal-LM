"""Read-only experimental inference using the fixed real MIMII checkpoint."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .analysis import analyze
from .data import load_signal
from .data.schema import read_manifest, sha256_of_file
from .training.engine import load_specialist

CHECKPOINT = Path("artifacts/experiments/mimii_fan_real_v1/final.pt")
DATA_ROOT = Path("data/mimii_fan_0db_windows_real_v1")


@lru_cache(maxsize=1)
def loaded_model():
    if not CHECKPOINT.exists():
        raise RuntimeError("Checkpoint réel absent sur ce poste. Lancez d’abord l’expérience MIMII.")
    return load_specialist(CHECKPOINT)


def examples():
    _, payload = loaded_model()
    if not (DATA_ROOT/"manifest.jsonl").exists():
        return []
    if sha256_of_file(DATA_ROOT/"manifest.jsonl") != payload["provenance"]["manifest_sha256"]:
        raise ValueError("Le manifeste MIMII ne correspond plus au checkpoint.")
    records = sorted(read_manifest(DATA_ROOT/"manifest.jsonl"), key=lambda r: r.record_id)
    result = []
    for split in ["validation", "test"]:
        for label in ["normal", "abnormal"]:
            record = next((r for r in records if r.split == split and r.parent_record_id and r.labels == [label]), None)
            if record:
                result.append(record)
    return result


def info():
    _, payload = loaded_model()
    return {"status": "trained_experimental", "input_contract": payload["input_contract"],
        "threshold": payload["threshold"], "checkpoint": CHECKPOINT.name,
        "training_steps": payload["training_steps"],
        "held_out_clip_false_positive_rate": 1., "confidence_calibrated": False,
        "examples": [{"id": r.record_id, "split": r.split, "label": r.labels[0],
                      "machine_id": r.machine_id} for r in examples()]}


def predict(signal):
    model, payload = loaded_model()
    contract = payload["input_contract"]
    if (signal.sample_rate_hz != contract["sample_rate_hz"]
            or signal.samples.shape != (contract["n_samples"], len(contract["channel_names"]))
            or list(signal.channels) != contract["channel_names"]
            or list(signal.units) != contract["units"]):
        raise ValueError(f"Modèle incompatible : attendu {contract['sample_rate_hz']:g} Hz, "
                         f"{len(contract['channel_names'])} canaux et {contract['n_samples']} échantillons. "
                         "Aucun rééchantillonnage ni duplication de canaux implicite. Utilisez un exemple MIMII.")
    x = (signal.samples - np.asarray(payload["normalization"]["mean"])) / np.asarray(payload["normalization"]["scale"])
    with torch.no_grad():
        embedding = model(torch.tensor(x, dtype=torch.float32).unsqueeze(0))["embedding"]
        score = float(((embedding-payload["center"])**2).sum())
    if not np.isfinite(score):
        raise ValueError("Score non fini")
    return {"status": "experimental", "score": score, "threshold": payload["threshold"],
            "above_threshold": score > payload["threshold"], "confidence": None,
            "warning": "Non fiable : 100 % de faux positifs sur les clips normaux de la machine test.",
            "scope": "one_second_window", "trained": True, "weights_updated": False}


def test_example(record_id):
    record = next((r for r in examples() if r.record_id == record_id), None)
    if record is None:
        raise ValueError("Exemple inconnu")
    if sha256_of_file(DATA_ROOT/record.file_path) != record.file_sha256:
        raise ValueError("Intégrité de l’exemple invalide")
    signal = load_signal(DATA_ROOT, record)
    analysis = analyze(signal, include_spectrum=True)
    indices = np.unique(np.linspace(0, len(signal.samples)-1, 1800).astype(int))
    analysis["waveform"] = {"time_s": (indices/signal.sample_rate_hz).tolist(),
        "values": signal.samples[indices].T.tolist(), "display_samples": len(indices)}
    return {"model": predict(signal), "analysis": analysis,
            "example": {"id": record.record_id, "source_label": record.labels[0], "split": record.split}}
