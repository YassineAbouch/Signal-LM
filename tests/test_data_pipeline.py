"""Manifest schema, dataset generation, split without leakage, windows, quality, importers."""
import csv
import json
import wave

import numpy as np
import pytest

from signal_lm.data import schema, split as split_mod, windows
from signal_lm.data.importers import import_csv, import_wav, read_wav
from signal_lm.data.quality import run_quality, write_report
from signal_lm.data.schema import EventAnnotation, Record, read_manifest, write_manifest
from signal_lm.data.store import generate_dataset, load_arrays, load_signal, regenerate_record
from signal_lm.data.cli import main as cli_main

FS = 2048
N = 2048


def small_config(**overrides):
    config = {
        "dataset_id": "unit", "specialist": "vibration", "sample_rate_hz": FS, "n_samples": N,
        "channel_names": ["acc"], "units": ["a.u."], "seed": 123, "n_parents": 4,
        "snr_db_levels": [-5, 10, None], "noise": {"type": "white_noise"},
        "recipes": [
            {"recipe_id": "sine", "task_type": "event_classification", "labels": ["sine"],
             "components": [{"type": "sine", "frequency_hz": [50, 500], "amplitude": [0.2, 1.0]}]},
            {"recipe_id": "impulses", "task_type": "event_detection", "labels": ["periodic_impulses"],
             "components": [{"type": "impulses", "rate_hz": [8, 32], "amplitude": [1, 3],
                             "width_samples": {"int_uniform": [1, 4]}, "kind": "half_sine", "label": "impulse"}]},
            {"recipe_id": "mix", "task_type": "multilabel_classification", "labels": ["sine", "chirp"],
             "components": [{"type": "sine", "frequency_hz": [50, 200], "amplitude": 0.5},
                            {"type": "chirp", "f_start_hz": [100, 200], "f_end_hz": [500, 800], "amplitude": 0.5}]},
        ],
    }
    config.update(overrides)
    return config


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    records = generate_dataset(small_config(), root)
    return root, records


def test_record_validation_catches_problems():
    record = Record(record_id="r", source_id="s", specialist="vibration", task_type="event_detection",
                    file_path="x.npz", sample_rate_hz=100, n_samples=100, channel_names=["a"], units=["g"],
                    group_id="g1", events=[EventAnnotation("e", 50, 200)])
    problems = record.validate()
    assert any("outside" in p for p in problems)
    bad = Record(record_id="", source_id="s", specialist="nope", task_type="event_classification",
                 file_path="x", sample_rate_hz=0, n_samples=2, channel_names=["a", "a"], units=["g"],
                 is_synthetic=True)
    messages = " | ".join(bad.validate())
    for fragment in ("record_id", "specialist", "sample_rate_hz", "n_samples", "unique", "units", "seed", "group_id"):
        assert fragment in messages
    good = Record(record_id="r", source_id="s", specialist="acoustic", task_type="event_classification",
                  file_path="x.npz", sample_rate_hz=16000, n_samples=16000, channel_names=["m"],
                  units=["unspecified"], labels=["x"], group_id="g")
    assert good.validate() == []


def test_manifest_round_trip(tmp_path):
    record = Record(record_id="r", source_id="s", specialist="music", task_type="multilabel_classification",
                    file_path="x.npz", sample_rate_hz=44100.0, n_samples=44100, channel_names=["l", "r"],
                    units=["unspecified"] * 2, labels=["a", "b"], group_id="g",
                    events=[EventAnnotation("a", 0, 10)], generation={"seed": 1}, is_synthetic=True)
    path = tmp_path / "m.jsonl"
    assert write_manifest([record], path) == 1
    back = read_manifest(path)
    assert back == [record]
    assert json.loads(path.read_text().splitlines()[0])["events"][0]["start_sample"] == 0


def test_generate_dataset_layout_and_manifest(dataset):
    root, records = dataset
    assert len(records) == 3 * 4 * 3
    assert (root / "config.json").exists() and (root / "manifest.jsonl").exists()
    for record in records:
        assert record.validate() == []
        assert record.is_synthetic and record.file_sha256 != schema.UNKNOWN
        arrays = load_arrays(root, record)
        assert set(arrays) == {"mix", "clean", "noise"}
        assert arrays["mix"].shape == (N, 1)
        np.testing.assert_allclose(arrays["mix"], arrays["clean"] + arrays["noise"], atol=1e-5)
        assert record.units == ["a.u."]
    detection = [r for r in records if r.task_type == "event_detection"]
    assert all(r.events and all(e.label == "impulse" for e in r.events) for r in detection)
    clean = [r for r in records if r.snr_db_target is None]
    assert all(r.snr_db_measured is None for r in clean)
    noisy = [r for r in records if r.snr_db_target is not None]
    assert all(abs(r.snr_db_measured - r.snr_db_target) < 1e-9 for r in noisy)
    assert generate_dataset(small_config(), root, overwrite=True)[0].file_sha256 == records[0].file_sha256
    with pytest.raises(FileExistsError):
        generate_dataset(small_config(), root)


def test_regeneration_matches_stored_files(dataset):
    root, records = dataset
    for record in records[:5]:
        fresh = regenerate_record(record)
        stored = load_arrays(root, record)
        for key in ("clean", "noise", "mix"):
            np.testing.assert_array_equal(stored[key], fresh[key].astype(np.float32).astype(np.float64))


def test_load_signal_preserves_amplitude(dataset):
    root, records = dataset
    record = next(r for r in records if r.session_id == "sine" and r.snr_db_target is None)
    signal = load_signal(root, record)
    expected = record.generation["components"][0]["amplitude"]
    assert np.max(np.abs(signal.samples)) == pytest.approx(expected, rel=1e-3)
    assert signal.channels == ("acc",) and signal.units == ("a.u.",)


def test_split_keeps_groups_together_and_is_deterministic(dataset):
    root, records = dataset
    summary = split_mod.assign_splits(records, seed=3)
    assert summary["n_groups"] == 12 and split_mod.check_leakage(records) == []
    by_group = {}
    for r in records:
        by_group.setdefault(r.group_id, set()).add(r.split)
    assert all(len(s) == 1 for s in by_group.values())
    assert set(summary["counts"]) == {"train", "validation", "test"}
    assert all(r.split in ("train", "validation", "test") for r in records)
    # every recipe appears in train (stratified) and the assignment is reproducible
    first = {r.record_id: r.split for r in records}
    split_mod.assign_splits(records, seed=3)
    assert {r.record_id: r.split for r in records} == first
    split_mod.assign_splits(records, seed=4)
    assert {r.record_id: r.split for r in records} != first


def test_holdout_rule_forces_group_to_test(dataset):
    root, records = dataset
    target = next(r for r in records if r.session_id == "sine")
    f = target.generation["components"][0]["frequency_hz"]
    rule = [{"field": "generation.components.0.frequency_hz", "range": [f - 1, f + 1],
             "split": "test", "reason": "unseen frequency"}]
    summary = split_mod.assign_splits(records, seed=0, holdout=rule)
    assert summary["holdout_groups"] == {target.group_id: "test"}
    assert all(r.split == "test" for r in records if r.group_id == target.group_id)


def test_leakage_detector_flags_split_group_and_hash():
    base = dict(source_id="s", specialist="vibration", task_type="event_classification", file_path="x",
                sample_rate_hz=10.0, n_samples=10, channel_names=["a"], units=["g"])
    a = Record(record_id="a", group_id="g", split="train", file_sha256="h1", **base)
    b = Record(record_id="b", group_id="g", split="test", file_sha256="h2", **base)
    c = Record(record_id="c", group_id="other", split="test", file_sha256="h1", **base)
    problems = split_mod.check_leakage([a, b, c])
    assert any("group g spans" in p for p in problems)
    assert any("identical file" in p for p in problems)
    w = Record(record_id="w", group_id="g", split="validation", parent_record_id="a", **base)
    assert any("parent split" in p for p in split_mod.check_leakage([a, w]))


def test_windows_inherit_split_and_relabel_events(dataset):
    root, records = dataset
    split_mod.assign_splits(records, seed=1)
    parent = next(r for r in records if r.task_type == "event_detection" and r.snr_db_target == 10)
    cut = windows.window_record(root, parent, window_samples=512, hop_samples=256)
    assert len(cut) == (N - 512) // 256 + 1
    for w in cut:
        assert w.split == parent.split and w.group_id == parent.group_id
        assert w.parent_record_id == parent.record_id and w.n_samples == 512
        assert w.transforms[-1]["op"] == "window"
        for e in w.events:
            assert 0 <= e.start_sample < e.end_sample <= 512
        assert w.labels == (sorted({e.label for e in w.events}) or ["no_event"])
        arrays = load_arrays(root, w)
        start = w.transforms[-1]["start_sample"]
        np.testing.assert_array_equal(arrays["mix"], load_arrays(root, parent)["mix"][start:start + 512])
    unsplit = Record(**{**parent.__dict__, "split": "unassigned", "record_id": "u"})
    with pytest.raises(ValueError):
        windows.window_record(root, unsplit, window_samples=512, hop_samples=256)


def test_quality_passes_on_generated_set_and_catches_corruption(dataset, tmp_path):
    root, records = dataset
    split_mod.assign_splits(records, seed=1)
    problems, report = run_quality(root, records, verify_regeneration=True)
    assert [p for p in problems if p["severity"] == "error"] == []
    assert report["n_records"] == len(records) and report["records_by_split"]
    json_path, md_path = write_report(root, report)
    assert json_path.exists() and "Rapport de qualité" in md_path.read_text(encoding="utf-8")

    # Corrupt one file: NaN inside, and tamper the manifest sample count of another.
    victim = records[0]
    arrays = load_arrays(root, victim)
    arrays["mix"][5, 0] = np.nan
    with (root / victim.file_path).open("wb") as stream:
        np.savez_compressed(stream, **{k: v.astype(np.float32) for k, v in arrays.items()})
    records[1].n_samples += 1
    moved = next(r for r in records if r.split == "train")
    moved.split = "test"  # leaves its siblings (same group) in train
    records[3].events = [EventAnnotation("x", 0, 10 ** 6)]
    problems, _ = run_quality(root, records, verify_hash=True)
    checks = {p["check"] for p in problems}
    assert {"finite", "hash", "shape", "leakage", "schema"} <= checks


def test_class_balance_flags_imbalance():
    from signal_lm.data.quality import class_balance
    base = dict(source_id="s", specialist="acoustic", task_type="event_classification", file_path="x",
                sample_rate_hz=10.0, n_samples=10, channel_names=["a"], units=["g"], group_id="g", split="train")
    records = [Record(record_id=f"a{i}", labels=["a"], **base) for i in range(12)]
    records.append(Record(record_id="b", labels=["b"], **base))
    info = class_balance(records)["acoustic/event_classification"]["train"]
    assert info["imbalanced"] and info["labels"] == {"a": 12, "b": 1}


def _write_wav(path, samples, rate, width):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(samples.shape[1])
        stream.setsampwidth(width)
        stream.setframerate(rate)
        if width == 2:
            stream.writeframes((samples * 32767).astype("<i2").tobytes())
        else:
            stream.writeframes(((samples * 127) + 128).astype(np.uint8).tobytes())


def test_wav_import_records_unspecified_units_and_scaling(tmp_path):
    t = np.arange(8000) / 8000
    stereo = np.column_stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.25 * np.sin(2 * np.pi * 880 * t)])
    path = tmp_path / "tone.wav"
    _write_wav(path, stereo, 8000, 2)
    samples, rate, info = read_wav(path)
    assert rate == 8000 and samples.shape == (8000, 2) and info["bits_per_sample"] == 16
    assert np.max(np.abs(samples[:, 0])) == pytest.approx(0.5, abs=1e-3)
    root = tmp_path / "real"
    record = import_wav(root, path, record_id="tone", source_id="unit", specialist="acoustic",
                        task_type="event_classification", group_id="session_a", labels=["tone"])
    assert record.units == ["unspecified", "unspecified"] and not record.is_synthetic
    assert record.transforms[0]["op"] == "wav_pcm_to_float"
    assert record.provenance["original_sha256"] and record.license == schema.UNKNOWN
    assert record.validate() == []
    assert load_arrays(root, record)["mix"].shape == (8000, 2)
    mono8 = tmp_path / "m8.wav"
    _write_wav(mono8, stereo[:, :1], 8000, 1)
    assert read_wav(mono8)[2]["bits_per_sample"] == 8


def test_csv_import_uses_explicit_columns(tmp_path):
    path = tmp_path / "s.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["t", "ax", "ay"])
        for i in range(100):
            writer.writerow([i / 100, np.sin(i / 5), np.cos(i / 5)])
    root = tmp_path / "real"
    record = import_csv(root, path, record_id="csv1", source_id="unit", specialist="vibration",
                        task_type="anomaly_detection", group_id="machine_1", sample_rate_hz=100,
                        columns=["ax", "ay"], units=["g", "g"], delimiter=";", machine_id="machine_1")
    assert record.channel_names == ["ax", "ay"] and record.units == ["g", "g"] and record.n_samples == 100
    assert record.validate() == []
    with pytest.raises(ValueError):
        import_csv(root, path, record_id="csv2", source_id="unit", specialist="vibration",
                   task_type="anomaly_detection", group_id="m", sample_rate_hz=100, columns=["missing"])


def test_cli_end_to_end(tmp_path, capsys):
    config = tmp_path / "cfg.json"
    config.write_text(json.dumps(small_config()), encoding="utf-8")
    split_cfg = tmp_path / "split.json"
    split_cfg.write_text(json.dumps({"seed": 2, "fractions": {"train": 0.5, "validation": 0.25, "test": 0.25}}))
    root = tmp_path / "ds"
    cli_main(["generate", "--config", str(config), "--root", str(root)])
    cli_main(["split", "--root", str(root), "--config", str(split_cfg)])
    cli_main(["window", "--root", str(root), "--window-samples", "1024", "--hop-samples", "1024"])
    cli_main(["validate", "--root", str(root), "--strict"])
    cli_main(["show", "--root", str(root)])
    out = capsys.readouterr().out
    assert '"n_errors": 0' in out and (root / "quality_report.md").exists()
    assert (root / "split_summary.json").exists()
    with pytest.raises(SystemExit):
        cli_main(["window", "--root", str(root), "--window-samples", "1024", "--hop-samples", "1024"])
