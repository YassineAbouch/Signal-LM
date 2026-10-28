"""One predeclared real-data pilot experiment, with test opened after freezing."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from signal_lm.data.schema import read_manifest, sha256_of_file, write_manifest
from signal_lm.data.quality import run_quality, write_report
from signal_lm.data.windows import window_record
from .dataset import ManifestDataset, manifest_records, select_records
from .engine import load_specialist, predictions, save_specialist, train
from .metrics import grouped_metrics

PROTOCOL = {
    "experiment": "mimii_fan_0db_real_v1", "specialist": "acoustic", "task_type": "anomaly_detection",
    "epochs": 10, "batch_size": 16, "seed": 42, "learning_rate": .001,
    "window_samples": 16000, "hop_samples": 16000, "sample_rate_hz": 16000,
    "channels": 8, "reference_quantile": .95, "device": "cuda",
    "clip_score": "mean_of_nonoverlapping_window_scores",
    "baseline": "unstandardized_physical_RMS_over_all_channels",
    "train_machines": ["fan_id_00", "fan_id_02"],
    "validation_machines": ["fan_id_04"], "test_machines": ["fan_id_06"],
    "selection": "minimum_reference_validation_loss", "hyperparameter_search": False,
    "test_policy": "deterministic_preparation_then_evaluation_only_after_freeze",
}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def binary_metrics(targets, scores, threshold):
    targets, scores = np.asarray(targets, dtype=bool), np.asarray(scores, dtype=float)
    if not len(targets) or scores.shape != targets.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid binary evaluation inputs")
    predicted = scores > threshold
    tp, fp = int(np.sum(targets & predicted)), int(np.sum(~targets & predicted))
    fn, tn = int(np.sum(targets & ~predicted)), int(np.sum(~targets & ~predicted))
    positive, negative = scores[targets], scores[~targets]
    auc = None
    if len(positive) and len(negative):
        auc = float(np.mean((positive[:, None] > negative[None, :]).astype(float)
                            + .5 * (positive[:, None] == negative[None, :])))
    return {"n": len(targets), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn) if tp+fn else None,
        "false_positive_rate": fp/(fp+tn) if fp+tn else None,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        "accuracy": (tp+tn)/len(targets), "roc_auc": auc, "threshold": threshold}


def aggregate_clips(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["parent_record_id"]].append(row)
    result = []
    for parent, windows in sorted(groups.items()):
        if len({r["target"] for r in windows}) != 1 or len({r["machine_id"] for r in windows}) != 1:
            raise ValueError("Inconsistent clip annotations")
        result.append({"record_id": parent, "machine_id": windows[0]["machine_id"],
            "target": windows[0]["target"], "n_windows": len(windows),
            "neural_score": float(np.mean([r["neural_score"] for r in windows])),
            "rms_score": float(np.mean([r["rms_score"] for r in windows]))})
    return result


def thresholds_from_validation(windows, clips):
    result = {}
    for level, rows in [("window", windows), ("clip", clips)]:
        reference = [r for r in rows if r["target"] == 0]
        if not reference:
            raise ValueError("Missing reference validation examples")
        for score in ["neural_score", "rms_score"]:
            result[f"{level}_{score}"] = float(np.quantile([r[score] for r in reference], .95))
    return result


def prepare_windows(source, root, progress):
    records = manifest_records(source)
    expected = {"train": set(PROTOCOL["train_machines"]), "validation": set(PROTOCOL["validation_machines"]),
                "test": set(PROTOCOL["test_machines"])}
    for record in records:
        if (record.parent_record_id is not None or record.specialist != "acoustic"
                or record.task_type != "anomaly_detection" or record.is_synthetic
                or record.sample_rate_hz != 16000 or len(record.channel_names) != 8
                or record.group_id != record.machine_id
                or record.machine_id not in expected[record.split]):
            raise ValueError("Source does not match the declared MIMII pilot protocol")
    root.mkdir(parents=True, exist_ok=False)
    (root/"records").mkdir()
    source_config = json.loads((source/"config.json").read_text(encoding="utf-8"))
    write_json(root/"config.json", {**source_config, "dataset_id": "mimii_fan_0db_windows_real_v1",
        "parent_dataset_id": source_config["dataset_id"], "parent_manifest_sha256": sha256_of_file(source/"manifest.jsonl"),
        "window_samples": 16000, "hop_samples": 16000})
    shutil.copy2(source/"split_summary.json", root/"split_summary.json")

    def prepare(record):
        relative = Path(record.file_path)
        original = (source/relative).resolve()
        if not original.is_relative_to(source.resolve()) or sha256_of_file(original) != record.file_sha256:
            raise ValueError("Source integrity failure")
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        return window_record(root, record, window_samples=16000, hop_samples=16000)

    children = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(prepare, records), 1):
            children.extend(result)
            if index % 20 == 0:
                progress({"phase": "window_preparation", "parents": index, "total_parents": len(records)})
    write_manifest(records + children, root/"manifest.jsonl")
    problems, quality = run_quality(root, records + children)
    write_report(root, quality)
    if any(p["severity"] == "error" for p in problems):
        raise ValueError("Window quality check failed")
    return records + children, quality


def score_split(root, records, split, model, payload):
    selected = select_records(records, split, "acoustic", "anomaly_detection", "windows")
    dataset = ManifestDataset(root, selected, payload["labels"], payload["normalization"],
                              payload["input_contract"], cache=True)
    center = payload["center"].cuda()
    indices = {r.record_id: i for i, r in enumerate(selected)}
    rows = []
    for record, target, score in predictions(model, dataset, 16, "cuda", "anomaly_detection", center):
        physical = dataset.physical_signal(indices[record.record_id]).samples
        rows.append({"record_id": record.record_id, "parent_record_id": record.parent_record_id,
            "machine_id": record.machine_id, "group_id": record.group_id, "target": int(target),
            "neural_score": float(score), "rms_score": float(np.sqrt(np.mean(physical**2)))})
    dataset.verify_integrity()
    return rows


def reports(windows, clips, thresholds):
    result = {}
    for level, rows in [("window", windows), ("clip", clips)]:
        result[level] = {score: binary_metrics([r["target"] for r in rows], [r[score] for r in rows],
            thresholds[f"{level}_{score}"]) for score in ["neural_score", "rms_score"]}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the authorized, fixed MIMII real-data pilot")
    parser.add_argument("--source", type=Path, default=Path("data/mimii_fan_0db"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("This declared protocol requires CUDA")
    torch.set_num_threads(1)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protocol = {**PROTOCOL, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256_of_file(args.source/"manifest.jsonl"),
        "source_config_sha256": sha256_of_file(args.source/"config.json"),
        "code_sha256": {name: sha256_of_file(Path(__file__).with_name(name))
                        for name in ["real_run.py", "engine.py", "dataset.py", "tasks.py", "metrics.py"]}}
    write_json(args.run_dir/"protocol.json", protocol)

    def progress(event):
        event = {**event, "elapsed_s": round(time.monotonic()-started, 1)}
        print(json.dumps(event), flush=True)
        with (args.run_dir/"progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")

    records, quality = prepare_windows(args.source, args.root, progress)
    write_json(args.run_dir/"quality_summary.json", {"n_records": quality["n_records"],
        "n_windows": quality["n_windows"], "problems": quality["problems"]})
    progress({"phase": "training", "epochs": 10, "reference_only": True})
    trained = args.run_dir/"trained.pt"
    payload = train(args.root, trained, specialist="acoustic", task_type="anomaly_detection",
        level="windows", epochs=10, batch_size=16, seed=42, learning_rate=.001,
        device="cuda", reference_quantile=.95, allow_real_training=True, cache_arrays=True,
        progress=lambda event: progress({"phase": "epoch", **event}))
    model, payload = load_specialist(trained, "cuda")
    validation = score_split(args.root, records, "validation", model, payload)
    validation_clips = aggregate_clips(validation)
    thresholds = thresholds_from_validation(validation, validation_clips)
    thresholds["window_neural_score"] = payload["threshold"]
    write_json(args.run_dir/"validation_scores.json", validation)
    frozen = {"trained_checkpoint_sha256": sha256_of_file(trained),
        "protocol_sha256": sha256_of_file(args.run_dir/"protocol.json"), "thresholds": thresholds,
        "calibration_split": "validation", "reference_only": True,
        "validation_reference_clip_ids": [r["record_id"] for r in validation_clips if r["target"] == 0],
        "test_opened_for_model_or_threshold_selection": False}
    write_json(args.run_dir/"frozen_before_test.json", frozen)
    progress({"phase": "frozen_test_evaluation", "selected_epoch": payload["selected_epoch"]})
    test = score_split(args.root, records, "test", model, payload)
    test_clips = aggregate_clips(test)
    write_json(args.run_dir/"test_scores.json", test)
    result = {"protocol": protocol, "selected_epoch": payload["selected_epoch"],
        "training_steps": payload["training_steps"], "total_optimizer_steps": payload["total_optimizer_steps"],
        "validation": reports(validation, validation_clips, thresholds),
        "test": reports(test, test_clips, thresholds), "thresholds": thresholds,
        "independent_test_machines": 1, "independent_test_clips": len(test_clips),
        "limitations": "Pilot only; no calibrated confidence; one held-out test machine; no mechanical diagnosis.",
        "elapsed_s": round(time.monotonic()-started, 1)}
    payload["real_experiment"] = result
    selected_by_id = {r.record_id: r for r in records}
    counts = []
    for row in test:
        truth, predicted = bool(row["target"]), row["neural_score"] > payload["threshold"]
        counts.append({"record": selected_by_id[row["record_id"]],
            "tp": np.array([truth and predicted], dtype=np.int64),
            "fp": np.array([not truth and predicted], dtype=np.int64),
            "fn": np.array([truth and not predicted], dtype=np.int64)})
    payload["evaluations"]["test"] = grouped_metrics(counts, "anomaly_detection", payload["labels"],
                                                      payload["provenance"]["split_summary"])
    payload["evaluation_parent_checkpoint_sha256"] = sha256_of_file(trained)
    final = args.run_dir/"final.pt"
    save_specialist(payload, final)
    restored, final_payload = load_specialist(final)
    original, original_payload = load_specialist(trained)
    assert final_payload["normalization"] == original_payload["normalization"]
    assert final_payload["threshold"] == original_payload["threshold"]
    assert all(torch.equal(value, original_payload["state_dict"][key]) for key, value in final_payload["state_dict"].items())
    probe = torch.linspace(-1, 1, 16000*8).reshape(1, 16000, 8)
    with torch.no_grad():
        torch.testing.assert_close(restored(probe)["embedding"], original(probe)["embedding"], rtol=0, atol=0)
    result["checkpoint"] = str(final.resolve())
    result["checkpoint_sha256"] = sha256_of_file(final)
    result["cpu_roundtrip_exact"] = True
    if sha256_of_file(args.source/"manifest.jsonl") != protocol["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    write_json(args.run_dir/"result.json", result)
    progress({"phase": "complete", "checkpoint": str(final), "test_clip_metrics": result["test"]["clip"]})


if __name__ == "__main__":
    main()
