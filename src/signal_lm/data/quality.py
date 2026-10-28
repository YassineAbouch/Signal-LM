"""Quality controls and quality report for a dataset root.

Checks (each problem is a dict with ``record_id``, ``check``, ``detail``):

* manifest: schema validity, file exists, hash matches, n_samples / channel
  count / dtype consistent with the array, units present.
* arrays: NaN / inf, constant channels, truncated (trailing zeros),
  saturated (fraction of samples at the record's absolute maximum).
* annotations: event bounds inside the record.
* splits: unassigned, group / parent / hash leakage (see ``split``).
* classes: imbalance per (specialist, task_type) reported, not failed.
* synthetic: measured SNR vs target, dominant frequency vs the generating
  sine parameters (via ``signal_lm.analyze``, physical measurement only),
  regeneration from stored parameters reproduces the stored arrays.

Nothing here normalises or alters the data.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis import analyze
from ..signal import Signal
from .schema import Record, sha256_of_file
from .split import check_leakage
from .store import load_arrays, regenerate_record

SATURATION_FRACTION_WARN = 0.01     # >1 % of samples at the absolute peak
TRUNCATION_TAIL_FRACTION = 0.05     # last 5 % exactly zero
SNR_TOLERANCE_DB = 0.05
FREQ_TOLERANCE_BINS = 1.5           # dominant-frequency tolerance in FFT bins
IMBALANCE_RATIO_WARN = 5.0


def _problem(record_id: str, check: str, detail: str, severity: str = "error") -> dict[str, str]:
    return {"record_id": record_id, "check": check, "detail": detail, "severity": severity}


def check_record(root: Path, record: Record, *, verify_regeneration: bool = False,
                 verify_hash: bool = True) -> list[dict[str, str]]:
    problems = [_problem(record.record_id, "schema", p) for p in record.validate()]
    path = root / record.file_path
    if not path.exists():
        problems.append(_problem(record.record_id, "file", f"missing {record.file_path}"))
        return problems
    if verify_hash and record.file_sha256 != "unknown" and sha256_of_file(path) != record.file_sha256:
        problems.append(_problem(record.record_id, "hash", "file_sha256 does not match file"))
    try:
        arrays = load_arrays(root, record)
    except Exception as exc:  # noqa: BLE001 - any loading failure is a data problem
        problems.append(_problem(record.record_id, "file", f"cannot load: {exc}"))
        return problems
    mix = arrays.get("mix")
    if mix is None:
        problems.append(_problem(record.record_id, "file", "no 'mix' array"))
        return problems
    if mix.ndim != 2:
        problems.append(_problem(record.record_id, "shape", f"expected [samples, channels], got {mix.shape}"))
        return problems
    if mix.shape[0] != record.n_samples:
        problems.append(_problem(record.record_id, "shape", f"n_samples {record.n_samples} != array {mix.shape[0]}"))
    if mix.shape[1] != len(record.channel_names):
        problems.append(_problem(record.record_id, "shape", f"{len(record.channel_names)} channel names for {mix.shape[1]} columns"))
    if not np.isfinite(mix).all():
        problems.append(_problem(record.record_id, "finite", "NaN or inf in mix"))
        return problems
    for ch in range(mix.shape[1]):
        column = mix[:, ch]
        name = record.channel_names[ch] if ch < len(record.channel_names) else f"ch_{ch}"
        if np.ptp(column) == 0:
            problems.append(_problem(record.record_id, "constant", f"channel {name} is constant", "warning"))
            continue
        tail = max(1, int(TRUNCATION_TAIL_FRACTION * len(column)))
        if not record.is_synthetic and np.all(column[-tail:] == 0) and not np.all(column == 0):
            problems.append(_problem(record.record_id, "truncated", f"channel {name}: trailing {tail} samples are exactly zero", "warning"))
        peak = np.max(np.abs(column))
        at_peak = float(np.mean(np.abs(column) >= peak * (1 - 1e-6)))
        if record.is_synthetic:
            # The generator knows whether clipping was applied; the heuristic below is for real data
            # (sparse rectangular impulses would otherwise look saturated).
            declared = record.generation.get("saturation_limit")
            if declared is not None and record.parent_record_id is None and at_peak == 0:
                problems.append(_problem(record.record_id, "saturated", f"channel {name}: saturation_limit {declared} declared but no sample reaches it", "warning"))
        elif at_peak > SATURATION_FRACTION_WARN:
            problems.append(_problem(record.record_id, "saturated", f"channel {name}: {at_peak:.1%} of samples at the absolute peak", "warning"))
    if record.is_synthetic:
        problems.extend(_check_synthetic(record, arrays, verify_regeneration))
    return problems


def _check_synthetic(record: Record, arrays: dict[str, np.ndarray], verify_regeneration: bool) -> list[dict[str, str]]:
    problems = []
    clean, noise = arrays.get("clean"), arrays.get("noise")
    if clean is None or noise is None:
        return [_problem(record.record_id, "synthetic", "clean/noise arrays missing")]
    if record.snr_db_target is not None:
        p_s, p_n = float(np.mean(clean ** 2)), float(np.mean(noise ** 2))
        if p_s > 0 and p_n > 0:
            measured = 10 * math.log10(p_s / p_n)
            reference = record.generation.get("measured", {}).get("snr_reference", "global")
            is_parent = record.parent_record_id is None
            # A window's local SNR legitimately differs from its parent's global target.
            if is_parent and reference == "global" and abs(measured - record.snr_db_target) > SNR_TOLERANCE_DB:
                problems.append(_problem(record.record_id, "snr", f"measured {measured:.2f} dB vs target {record.snr_db_target:.2f} dB"))
            if record.snr_db_measured is not None and abs(measured - record.snr_db_measured) > SNR_TOLERANCE_DB:
                problems.append(_problem(record.record_id, "snr", f"float32 file SNR {measured:.2f} dB differs from manifest {record.snr_db_measured:.2f} dB"))
    components = record.generation.get("components", [])
    sines = [c for c in components if c["type"] == "sine"]
    if len(components) == 1 and sines:
        m = analyze(Signal(clean[:, 0], record.sample_rate_hz))["measurements"][0]
        bin_hz = record.sample_rate_hz / record.n_samples
        expected = sines[0]["frequency_hz"]
        if m["dominant_frequency_hz"] is None or abs(m["dominant_frequency_hz"] - expected) > FREQ_TOLERANCE_BINS * bin_hz:
            problems.append(_problem(record.record_id, "frequency", f"dominant {m['dominant_frequency_hz']} Hz vs generated {expected:.2f} Hz"))
        # Off-bin sines lose up to ~15 % through the Hann window (SPEC_V1): warn above 20 %.
        if abs(m["dominant_amplitude"] - sines[0].get("amplitude", 1.0)) > 0.2 * sines[0].get("amplitude", 1.0) + 1e-6:
            problems.append(_problem(record.record_id, "amplitude", f"dominant amplitude {m['dominant_amplitude']:.3f} vs generated {sines[0].get('amplitude', 1.0):.3f}", "warning"))
    if verify_regeneration:
        regenerated = regenerate_record(record)
        for key in ("clean", "noise", "mix"):
            stored = arrays[key]
            fresh = regenerated[key].astype(np.float32).astype(np.float64)
            if stored.shape != fresh.shape or not np.allclose(stored, fresh, rtol=0, atol=0):
                problems.append(_problem(record.record_id, "regeneration", f"{key} differs from regeneration"))
    return problems


def class_balance(records: list[Record]) -> dict[str, Any]:
    """Label counts per (specialist, task_type, split) and an imbalance flag."""
    table: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        key = f"{r.specialist}/{r.task_type}"
        for label in r.labels or ["<no_label>"]:
            table[key][r.split][label] += 1
    summary: dict[str, Any] = {}
    for key, splits in table.items():
        summary[key] = {}
        for split, counter in splits.items():
            counts = dict(counter)
            ratio = (max(counts.values()) / min(counts.values())) if len(counts) > 1 else 1.0
            summary[key][split] = {"labels": counts, "max_min_ratio": round(ratio, 2),
                                   "imbalanced": ratio > IMBALANCE_RATIO_WARN}
    return summary


def dataset_report(root: str | Path, records: list[Record], problems: list[dict[str, str]]) -> dict[str, Any]:
    root = Path(root)
    parents = [r for r in records if r.parent_record_id is None]
    by_split = Counter(r.split for r in records)
    by_specialist = Counter(r.specialist for r in records)
    by_task = Counter(f"{r.specialist}/{r.task_type}" for r in records)
    durations = [r.n_samples / r.sample_rate_hz for r in records]
    rates = Counter(f"{r.sample_rate_hz:g}" for r in records)
    snr = defaultdict(list)
    for r in records:
        if r.snr_db_measured is not None:
            snr[str(r.snr_db_target)].append(r.snr_db_measured)
    groups_per_split = defaultdict(set)
    for r in parents:
        groups_per_split[r.split].add(r.group_id)
    return {
        "root": str(root),
        "n_records": len(records), "n_parent_records": len(parents),
        "n_windows": len(records) - len(parents),
        "records_by_split": dict(by_split),
        "groups_by_split": {k: len(v) for k, v in groups_per_split.items()},
        "records_by_specialist": dict(by_specialist),
        "records_by_task": dict(by_task),
        "sample_rates_hz": dict(rates),
        "duration_s": {"min": min(durations) if durations else None,
                       "max": max(durations) if durations else None,
                       "total": sum(durations)},
        "n_synthetic": sum(r.is_synthetic for r in records),
        "snr_db_measured_by_target": {k: {"n": len(v), "mean": float(np.mean(v)),
                                           "min": float(min(v)), "max": float(max(v))}
                                       for k, v in snr.items()},
        "class_balance": class_balance(records),
        "problems": {"n_errors": sum(p["severity"] == "error" for p in problems),
                     "n_warnings": sum(p["severity"] == "warning" for p in problems),
                     "by_check": dict(Counter(p["check"] for p in problems)),
                     "items": problems[:200]},
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Rapport de qualité — `{report['root']}`", "",
             f"- Enregistrements : {report['n_records']} (parents {report['n_parent_records']}, fenêtres {report['n_windows']})",
             f"- Synthétiques : {report['n_synthetic']}",
             f"- Fréquences d'échantillonnage (Hz) : {report['sample_rates_hz']}",
             f"- Durée (s) : min {report['duration_s']['min']}, max {report['duration_s']['max']}, total {report['duration_s']['total']:.1f}",
             f"- Partitions (enregistrements) : {report['records_by_split']}",
             f"- Partitions (groupes parents) : {report['groups_by_split']}",
             f"- Par spécialité : {report['records_by_specialist']}",
             f"- Par tâche : {report['records_by_task']}", "", "## SNR mesuré par cible (dB)", ""]
    for target, stats in sorted(report["snr_db_measured_by_target"].items(), key=lambda kv: (kv[0] == "None", kv[0])):
        lines.append(f"- cible {target} : n={stats['n']}, moyenne {stats['mean']:.2f}, min {stats['min']:.2f}, max {stats['max']:.2f}")
    lines += ["", "## Équilibre des classes", ""]
    for task, splits in report["class_balance"].items():
        for split, info in splits.items():
            flag = " **(déséquilibré)**" if info["imbalanced"] else ""
            lines.append(f"- {task} / {split} : {info['labels']} (ratio max/min {info['max_min_ratio']}){flag}")
    p = report["problems"]
    lines += ["", "## Contrôles", "", f"- Erreurs : {p['n_errors']} ; avertissements : {p['n_warnings']}",
              f"- Par contrôle : {p['by_check']}"]
    for item in p["items"][:50]:
        lines.append(f"  - [{item['severity']}] {item['check']} — {item['record_id']} : {item['detail']}")
    lines += ["", "Ce rapport décrit des données synthétiques ou importées ; il ne constitue pas une",
              "validation de performance industrielle (voir docs/DATASET_PLAN.md)."]
    return "\n".join(lines) + "\n"


def run_quality(root: str | Path, records: list[Record], *, verify_regeneration: bool = False,
                verify_hash: bool = True) -> tuple[list[dict[str, str]], dict[str, Any]]:
    root = Path(root)
    problems: list[dict[str, str]] = []
    for record in records:
        problems.extend(check_record(root, record, verify_regeneration=verify_regeneration,
                                     verify_hash=verify_hash))
    # Constant (e.g. all-zero) windows are byte-identical across groups without being leaks.
    constant_ids = {p["record_id"] for p in problems if p["check"] == "constant"}
    problems.extend(_problem("<dataset>", "leakage", d)
                    for d in check_leakage(records, ignore_hash_for=constant_ids))
    ids = Counter(r.record_id for r in records)
    problems.extend(_problem(rid, "duplicate_id", f"record_id appears {n} times") for rid, n in ids.items() if n > 1)
    report = dataset_report(root, records, problems)
    return problems, report


def write_report(root: str | Path, report: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(root)
    json_path = root / "quality_report.json"
    md_path = root / "quality_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    md_path.write_text(report_markdown(report), encoding="utf-8")
    return json_path, md_path
