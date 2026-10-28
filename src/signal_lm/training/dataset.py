"""Manifest-backed, fixed-shape batches and train-only target/normalization fits."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from signal_lm import data
from signal_lm.data.schema import sha256_of_file


def manifest_records(root):
    records = data.read_manifest(Path(root) / "manifest.jsonl")
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("Duplicate record IDs")
    groups = {}
    by_id = {r.record_id: r for r in records}
    for r in records:
        problems = r.validate()
        if problems or r.split == "unassigned":
            raise ValueError(f"{r.record_id}: {problems or 'unassigned split'}")
        if r.group_id in groups and groups[r.group_id] != r.split:
            raise ValueError(f"Group leakage: {r.group_id}")
        groups[r.group_id] = r.split
        if r.parent_record_id:
            parent = by_id.get(r.parent_record_id)
            if parent is None or parent.split != r.split or parent.group_id != r.group_id:
                raise ValueError(f"Parent leakage: {r.record_id}")
    # Known all-zero synthetic no-event windows may legitimately share a hash.
    # Their constancy is verified when loading each selected array.
    hashes = {}
    for r in records:
        if r.file_sha256 != "unknown":
            hashes.setdefault(r.file_sha256, []).append(r)
    for members in hashes.values():
        if len({r.split for r in members}) > 1 and not all(constant_candidate(r) for r in members):
            raise ValueError("Identical content across splits")
    return records


def constant_candidate(record):
    return (record.parent_record_id is not None and record.is_synthetic
            and record.snr_db_target is None and record.labels == ["no_event"])


def anomaly_target(record, reference_labels=("reference", "normal"), deviation_labels=("deviation", "abnormal")):
    reference = bool(set(record.labels) & set(reference_labels))
    deviation = bool(set(record.labels) & set(deviation_labels))
    if reference == deviation:
        raise ValueError(f"{record.record_id}: need exactly one reference/deviation state")
    return int(deviation)


def select_records(records, split, specialist, task_type, level="windows", reference_only=False):
    selected = [r for r in records if (r.split, r.specialist, r.task_type) == (split, specialist, task_type)
                and (r.parent_record_id is not None) == (level == "windows")]
    if level not in {"windows", "parents"}:
        raise ValueError("level must be windows or parents")
    if reference_only:
        selected = [r for r in selected if anomaly_target(r) == 0]
    if not selected:
        raise ValueError(f"No records for {split}/{specialist}/{task_type}/{level}")
    return selected


def vocabulary(train_records, task_type):
    if any(r.split != "train" for r in train_records):
        raise ValueError("Vocabulary must be fitted only on train")
    if task_type == "anomaly_detection":
        return ["reference", "deviation"]
    labels = sorted({e.label for r in train_records for e in r.events} if task_type == "event_detection"
                    else {label for r in train_records for label in r.labels})
    if not labels:
        raise ValueError("No training labels")
    return labels


def encode_target(record, labels):
    lookup = {label: index for index, label in enumerate(labels)}
    if record.task_type == "anomaly_detection":
        return torch.tensor(anomaly_target(record), dtype=torch.long)
    if record.task_type == "event_detection":
        target = torch.zeros(record.n_samples, len(labels))
        for event in record.events:
            if event.label not in lookup:
                raise ValueError(f"Unknown event label outside train vocabulary: {event.label}")
            target[event.start_sample:event.end_sample, lookup[event.label]] = 1
        return target
    if any(label not in lookup for label in record.labels):
        raise ValueError("Unknown label outside train vocabulary")
    if record.task_type == "event_classification":
        if len(record.labels) != 1:
            raise ValueError("Exclusive classification needs exactly one label")
        return torch.tensor(lookup[record.labels[0]], dtype=torch.long)
    if record.task_type != "multilabel_classification":
        raise ValueError("Unsupported task")
    target = torch.zeros(len(labels))
    for label in record.labels:
        target[lookup[label]] = 1
    return target


class ManifestDataset(Dataset):
    def __init__(self, root, records, labels, normalization=None, contract=None, cache=False):
        self.root = Path(root).resolve()
        self.records, self.labels, self.normalization = records, labels, normalization
        self.cache = {} if cache else None
        hashes = {}
        for r in data.read_manifest(self.root/"manifest.jsonl"):
            hashes.setdefault(r.file_sha256, set()).add(r.split)
        self.constant_hashes = {digest for digest, splits in hashes.items()
                                if digest != "unknown" and len(splits) > 1}
        if not records:
            raise ValueError("Empty dataset")
        first = records[0]
        self.contract = contract or {"n_samples": first.n_samples, "sample_rate_hz": first.sample_rate_hz,
            "channel_names": first.channel_names, "units": first.units}
        for r in records:
            expected = {"n_samples": r.n_samples, "sample_rate_hz": r.sample_rate_hz,
                        "channel_names": r.channel_names, "units": r.units}
            if expected != self.contract:
                raise ValueError("Incompatible length/rate/channels/units: window before batching; no masked padding")
            path = (self.root / r.file_path).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Dataset file escapes root")

    def __len__(self):
        return len(self.records)

    def physical_signal(self, index):
        if self.cache is not None and index in self.cache:
            return self.cache[index]
        record = self.records[index]
        if record.file_sha256 == "unknown" or sha256_of_file(self.root / record.file_path) != record.file_sha256:
            raise ValueError(f"Integrity check failed: {record.record_id}")
        signal = data.load_signal(self.root, record)
        if signal.samples.shape != (record.n_samples, len(record.channel_names)):
            raise ValueError("Array shape differs from manifest")
        if record.file_sha256 in self.constant_hashes and np.ptp(signal.samples, axis=0).any():
            raise ValueError("Hash exemption requires a constant no-event window")
        if self.cache is not None:
            self.cache[index] = signal
        return signal

    def verify_integrity(self):
        for record in self.records:
            if sha256_of_file(self.root / record.file_path) != record.file_sha256:
                raise ValueError(f"Integrity check failed: {record.record_id}")

    def __getitem__(self, index):
        values = self.physical_signal(index).samples.copy()
        if self.normalization:
            values = (values - np.array(self.normalization["mean"])) / np.array(self.normalization["scale"])
        return torch.tensor(values, dtype=torch.float32), encode_target(self.records[index], self.labels), index


def fit_normalization(dataset):
    if any(r.split != "train" for r in dataset.records):
        raise ValueError("Normalization must use train only")
    if dataset.normalization is not None:
        raise ValueError("Fit statistics on physical values only")
    if any(r.task_type == "anomaly_detection" and anomaly_target(r) for r in dataset.records):
        raise ValueError("Anomaly normalization uses reference train only")
    count, mean, m2 = 0, np.zeros(len(dataset.contract["units"])), np.zeros(len(dataset.contract["units"]))
    for index in range(len(dataset)):
        x = dataset.physical_signal(index).samples
        n, local_mean = len(x), x.mean(axis=0)
        local_m2 = ((x - local_mean) ** 2).sum(axis=0)
        delta = local_mean - mean
        m2 += local_m2 + delta ** 2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    scale = np.sqrt(m2 / count)
    constant = scale < 1e-12
    scale[constant] = 1
    return {"method": "per_channel_train_zscore", "split": "train", "mean": mean.tolist(),
            "scale": scale.tolist(), "constant_channels": constant.tolist(), "sample_count": count,
            "record_ids": [r.record_id for r in dataset.records],
            "group_ids": sorted({r.group_id for r in dataset.records})}
