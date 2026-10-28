"""Train/validation selection, frozen final evaluation, specialized checkpoints."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from signal_lm.data.schema import sha256_of_file
from signal_lm.model import ModelConfig
from .dataset import (ManifestDataset, anomaly_target, fit_normalization, manifest_records,
                      select_records, vocabulary)
from .metrics import grouped_metrics
from .tasks import build_task_model, task_config, task_loss

FORMAT = "signal_lm.specialist.v1"


def loader(dataset, batch_size, shuffle=False, seed=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def predictions(model, dataset, batch_size, device, task_type, center=None):
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, y, indices in loader(dataset, batch_size):
            output = model(x.to(device))
            value = ((output["embedding"] - center) ** 2).sum(dim=1) if task_type == "anomaly_detection" else output["logits"]
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite model output")
            for index, target, score in zip(indices.tolist(), y, value.cpu()):
                outputs.append((dataset.records[index], target.numpy(), score.numpy()))
    return outputs


def evaluate(model, dataset, batch_size, device, task_type, labels, threshold, summary, center=None):
    rows = []
    for record, target, score in predictions(model, dataset, batch_size, device, task_type, center):
        row = {"record": record}
        if task_type == "event_classification":
            confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
            confusion[int(target), int(np.argmax(score))] = 1
            row["confusion"] = confusion
        else:
            predicted = np.atleast_1d(score > threshold if task_type == "anomaly_detection" else score > np.log(threshold/(1-threshold)))
            truth = np.atleast_1d(target.astype(bool))
            if task_type == "event_detection":
                axes = (0,)
            else:
                axes = ()
            row.update(tp=np.sum(predicted & truth, axis=axes, dtype=np.int64),
                       fp=np.sum(predicted & ~truth, axis=axes, dtype=np.int64),
                       fn=np.sum(~predicted & truth, axis=axes, dtype=np.int64))
        rows.append(row)
    return grouped_metrics(rows, task_type, labels, summary)


def save_specialist(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)


def load_specialist(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != FORMAT or payload.get("status") != "trained"
            or payload.get("training_steps", 0) < 1):
        raise ValueError("Expected a trained specialist checkpoint, not a virgin checkpoint")
    model = build_task_model(ModelConfig(**payload["model_config"]), payload["task_type"], len(payload["labels"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def _json(root, name):
    path = root / name
    if not path.exists():
        raise ValueError(f"Required dataset provenance missing: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def train(root, output, *, specialist, task_type, level="windows", epochs=1,
          batch_size=32, seed=42, learning_rate=1e-3, device="cpu", reference_quantile=.95,
          allow_real_training=False):
    root = Path(root)
    if Path(output).exists():
        raise FileExistsError(output)
    if epochs < 1 or batch_size < 1 or not 0 < learning_rate < 1 or not 0 < reference_quantile < 1:
        raise ValueError("Invalid epochs, batch size, learning rate or reference quantile")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    records = manifest_records(root)
    config = _json(root, "config.json")
    summary = _json(root, "split_summary.json")
    digest = sha256_of_file(root / "manifest.jsonl")
    train_records = select_records(records, "train", specialist, task_type, level,
                                   reference_only=task_type == "anomaly_detection")
    if not allow_real_training and any(not r.is_synthetic for r in train_records):
        raise ValueError("Real-data training requires explicit --allow-real-training; import alone never trains")
    labels = vocabulary(train_records, task_type)
    raw_train = ManifestDataset(root, train_records, labels)
    normalization = fit_normalization(raw_train)
    contract = raw_train.contract

    def dataset(split, reference_only=False):
        selected = select_records(records, split, specialist, task_type, level, reference_only)
        return ManifestDataset(root, selected, labels, normalization, contract)

    train_set = dataset("train", task_type == "anomaly_detection")
    validation_set = dataset("validation", task_type == "anomaly_detection")
    # Fit task vocabulary and input contract before loading any validation tensors.
    model_config = task_config(len(contract["units"]), task_type, len(labels), seed)
    model = build_task_model(model_config, task_type, len(labels)).to(device)
    center = None
    if task_type == "anomaly_detection":
        with torch.no_grad():
            total = torch.zeros(model_config.embedding_dim, device=device)
            for x, _, _ in loader(train_set, batch_size):
                total += model(x.to(device))["embedding"].sum(dim=0)
            center = total / len(train_set)
            center = torch.where(center.abs() < .1, torch.where(center < 0, -.1, .1), center)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history, best_loss, steps, best_state, selected_epoch, best_steps = [], float("inf"), 0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.
        for x, y, _ in loader(train_set, batch_size, True, seed + epoch):
            optimizer.zero_grad(set_to_none=True)
            loss = task_loss(model(x.to(device)), y.to(device), task_type, center)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Non-finite gradient")
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(x)
            steps += 1
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for x, y, _ in loader(validation_set, batch_size):
                val_loss += float(task_loss(model(x.to(device)), y.to(device), task_type, center).cpu()) * len(x)
        val_loss /= len(validation_set)
        if not np.isfinite(val_loss):
            raise ValueError("Non-finite validation loss")
        history.append({"epoch": epoch, "train_loss": train_loss/len(train_set), "validation_loss": val_loss})
        if val_loss < best_loss:
            best_loss, selected_epoch, best_steps = val_loss, epoch, steps
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    threshold, calibration = .5, {"method": "fixed_probability_threshold", "value": .5,
                                  "confidence_calibrated": False}
    if task_type == "anomaly_detection":
        scores = [float(score) for _, _, score in predictions(model, validation_set, batch_size, device, task_type, center)]
        threshold = float(np.quantile(scores, reference_quantile))
        calibration = {"method": "validation_reference_score_quantile", "split": "validation",
            "quantile": reference_quantile, "n_reference_records": len(scores),
            "value": threshold, "confidence_calibrated": False,
            "record_ids": [r.record_id for r in validation_set.records]}
    if task_type == "event_classification":
        threshold, calibration = None, {"method": "argmax", "confidence_calibrated": False}
    evaluations = {split: evaluate(model, dataset(split), batch_size, device, task_type,
                    labels, threshold, summary, center) for split in ("train", "validation")}
    if digest != sha256_of_file(root / "manifest.jsonl"):
        raise ValueError("Manifest changed during fitting")
    payload = {"format": FORMAT, "format_version": 1, "status": "trained", "specialist": specialist,
        "task_type": task_type, "labels": labels, "label_to_index": {label: i for i, label in enumerate(labels)},
        "model_config": asdict(model_config),
        "state_dict": best_state, "training_steps": best_steps, "total_optimizer_steps": steps,
        "normalization": normalization, "input_contract": contract, "level": level,
        "center": center.detach().cpu() if center is not None else None,
        "threshold": threshold, "calibration": calibration, "selected_epoch": selected_epoch,
        "selection": "minimum_validation_loss; reference_only_for_anomaly", "history": history,
        "training_config": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
            "learning_rate": learning_rate, "weight_decay": 1e-4, "device": device},
        "provenance": {"dataset_id": config.get("dataset_id", root.name),
            "dataset_seed": config.get("seed"), "training_seed": seed,
            "source_ids": sorted({r.source_id for r in train_records}),
            "manifest_sha256": digest, "config_sha256": sha256_of_file(root/"config.json"),
            "split_summary_sha256": sha256_of_file(root/"split_summary.json"),
            "split_summary": summary, "licenses": sorted({r.license for r in train_records})},
        "environment": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
        "evaluations": evaluations, "test_used_for_selection": False,
        "limitations": "Pipeline baseline only; no industrial validation; anomaly one-class embedding distance; event metrics sample-level."}
    save_specialist(payload, output)
    return payload


def evaluate_checkpoint(root, checkpoint, output, splits=("test",), device="cpu", batch_size=32):
    if Path(output).exists():
        raise FileExistsError(output)
    root = Path(root)
    model, payload = load_specialist(checkpoint, device)
    for filename, key in [("manifest.jsonl", "manifest_sha256"), ("config.json", "config_sha256"),
                          ("split_summary.json", "split_summary_sha256")]:
        if sha256_of_file(root/filename) != payload["provenance"][key]:
            raise ValueError(f"Dataset provenance changed: {filename}")
    records = manifest_records(root)
    center = payload["center"].to(device) if payload["center"] is not None else None
    for split in splits:
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown evaluation split")
        selected = select_records(records, split, payload["specialist"], payload["task_type"], payload["level"])
        dataset = ManifestDataset(root, selected, payload["labels"], payload["normalization"], payload["input_contract"])
        payload["evaluations"][split] = evaluate(model, dataset, batch_size, device, payload["task_type"],
            payload["labels"], payload["threshold"], payload["provenance"]["split_summary"], center)
    payload["evaluation_parent_checkpoint_sha256"] = sha256_of_file(checkpoint)
    save_specialist(payload, output)
    return payload
