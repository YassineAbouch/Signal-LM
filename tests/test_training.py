import json
from pathlib import Path

import numpy as np
import pytest
import torch

from signal_lm import data
from signal_lm.data.schema import EventAnnotation, write_manifest
from signal_lm.data.store import add_imported_record
from signal_lm.model import load_virgin
from signal_lm.training.dataset import (ManifestDataset, encode_target, fit_normalization,
                                      manifest_records, select_records, vocabulary)
from signal_lm.training.engine import evaluate_checkpoint, load_specialist, train
from signal_lm.training.tasks import build_task_model, task_config


def fixture_dataset(root, task="event_classification"):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for split_index, split in enumerate(["train", "validation", "test"]):
        for i in range(4):
            record_id = f"{split}-{i}"
            t = np.arange(64)/256
            x = np.sin(2*np.pi*(20 + 20*(i % 2))*t + .03*split_index + .01*i)
            labels, events = ["sine" if i % 2 == 0 else "chirp"], []
            if task == "multilabel_classification":
                labels = ["sine"] + (["chirp"] if i % 2 else [])
            elif task == "event_detection":
                events = [EventAnnotation("impulse", 10, 20)] if i % 2 else []
                labels = ["impulse"] if events else ["no_event"]
            elif task == "anomaly_detection":
                labels = ["reference"] if i % 2 == 0 else ["deviation"]
                x *= 1 if i % 2 == 0 else 3
            record = add_imported_record(root, record_id, x[:, None], source_id="fixture",
                specialist="vibration", task_type=task, sample_rate_hz=256, channel_names=["x"],
                units=["a.u."], group_id=record_id, labels=labels, events=events,
                machine_id=f"machine-{split_index}", license="project-internal-synthetic")
            record.split = split
            record.is_synthetic = True
            record.generation = {"seed": i, "components": []}
            record.snr_db_target = [None, -10, 0, 20][i]
            records.append(record)
    write_manifest(records, root/"manifest.jsonl")
    (root/"config.json").write_text(json.dumps({"dataset_id": "fixture", "seed": 17}))
    (root/"split_summary.json").write_text(json.dumps({"seed": 7,
        "holdout_groups": {"test-0": "test"}, "reasons": {"test-0": "unseen machine"}}))
    return records


@pytest.mark.parametrize("task", ["event_classification", "multilabel_classification",
                                  "event_detection", "anomaly_detection"])
def test_fit_has_no_test_access_and_final_evaluation_is_frozen(tmp_path, monkeypatch, task):
    torch.set_num_threads(1)
    root = tmp_path/"dataset"
    fixture_dataset(root, task)
    output = tmp_path/"specialist.pt"
    original_loader = data.load_signal
    accessed = []

    def guarded(root, record):
        accessed.append(record.split)
        assert record.split != "test", "fit must not read test tensors"
        return original_loader(root, record)

    monkeypatch.setattr(data, "load_signal", guarded)
    payload = train(root, output, specialist="vibration", task_type=task, level="parents", epochs=1, batch_size=2)
    assert "test" not in accessed
    assert payload["provenance"]["dataset_seed"] == 17
    assert len(payload["provenance"]["manifest_sha256"]) == 64
    assert payload["normalization"]["split"] == "train"
    assert all(name.startswith("train-") for name in payload["normalization"]["record_ids"])
    assert "test" not in payload["evaluations"]
    assert payload["training_steps"] > 0
    if task == "anomaly_detection":
        assert payload["normalization"]["record_ids"] == ["train-0", "train-2"]
        assert payload["calibration"]["record_ids"] == ["validation-0", "validation-2"]
    with pytest.raises(FileExistsError):
        train(root, output, specialist="vibration", task_type=task, level="parents")
    with pytest.raises(ValueError):
        load_virgin(output)
    model, restored = load_specialist(output)
    fresh, _ = load_specialist(output)
    x = torch.randn(2, 64, 1)
    with torch.no_grad():
        for key, values in model(x).items():
            torch.testing.assert_close(values, fresh(x)[key], rtol=0, atol=0)
    monkeypatch.setattr(data, "load_signal", original_loader)
    final = evaluate_checkpoint(root, output, tmp_path/"evaluated.pt", splits=["test"])
    assert final["threshold"] == restored["threshold"]
    assert final["normalization"] == restored["normalization"]
    for key in restored["state_dict"]:
        assert torch.equal(final["state_dict"][key], restored["state_dict"][key])
    report = final["evaluations"]["test"]
    assert report["overall"]["n_records"] == 4
    assert set(report["by_snr_db_target"]) == {"clean", "-10 dB", "0 dB", "20 dB"}
    assert "unseen machine" in report["by_holdout_rule"]


def test_normalization_preserves_arrays_and_rejects_validation_fit(tmp_path):
    records = fixture_dataset(tmp_path)
    selected = select_records(records, "train", "vibration", "event_classification", "parents")
    labels = vocabulary(selected, "event_classification")
    raw = ManifestDataset(tmp_path, selected, labels)
    before = [(tmp_path/r.file_path).read_bytes() for r in selected]
    stats = fit_normalization(raw)
    values = np.concatenate([raw.physical_signal(i).samples for i in range(len(raw))])
    assert stats["mean"] == pytest.approx(values.mean(axis=0))
    assert stats["scale"] == pytest.approx(values.std(axis=0))
    assert before == [(tmp_path/r.file_path).read_bytes() for r in selected]
    val = ManifestDataset(tmp_path, [r for r in records if r.split == "validation"], labels)
    with pytest.raises(ValueError, match="train only"):
        fit_normalization(val)
    with pytest.raises(ValueError):
        vocabulary(val.records, "event_classification")


def test_temporal_targets_use_half_open_bounds(tmp_path):
    records = fixture_dataset(tmp_path, "event_detection")
    target = encode_target(records[1], ["impulse"])
    assert target.shape == (64, 1)
    assert target.sum() == 10
    assert target[10] == 1 and target[19] == 1 and target[20] == 0
    assert encode_target(records[0], ["impulse"]).sum() == 0
    model = build_task_model(task_config(1, "event_detection", 1, 42), "event_detection", 1)
    assert model(torch.zeros(2, 64, 1))["logits"].shape == (2, 64, 1)


def test_mixed_lengths_and_group_leakage_fail_closed(tmp_path):
    records = fixture_dataset(tmp_path)
    records[1].n_samples = 32
    with pytest.raises(ValueError, match="window"):
        ManifestDataset(tmp_path, records[:2], ["sine", "chirp"])
    records[1].n_samples = 64
    records[4].group_id = records[0].group_id
    write_manifest(records, tmp_path/"manifest.jsonl")
    with pytest.raises(ValueError, match="leakage"):
        manifest_records(tmp_path)


def test_manifest_change_blocks_final_evaluation(tmp_path):
    root = tmp_path/"data"
    records = fixture_dataset(root)
    checkpoint = tmp_path/"model.pt"
    train(root, checkpoint, specialist="vibration", task_type="event_classification", level="parents")
    records[-1].notes = "changed"
    write_manifest(records, root/"manifest.jsonl")
    with pytest.raises(ValueError, match="provenance"):
        evaluate_checkpoint(root, checkpoint, tmp_path/"final.pt")


def test_imported_real_data_does_not_train_without_explicit_option(tmp_path, monkeypatch):
    records = fixture_dataset(tmp_path, "anomaly_detection")
    for record in records:
        record.is_synthetic = False
        record.generation = {}
    write_manifest(records, tmp_path/"manifest.jsonl")
    def no_loading(*args):
        raise AssertionError("real samples must not load before explicit authorization")
    monkeypatch.setattr(data, "load_signal", no_loading)
    with pytest.raises(ValueError, match="allow-real-training"):
        train(tmp_path, tmp_path/"real.pt", specialist="vibration", task_type="anomaly_detection", level="parents")
