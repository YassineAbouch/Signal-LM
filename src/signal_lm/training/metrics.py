"""Inspectable count-based metrics, with explicit sample-level detection scope."""
import numpy as np


def metrics(rows, task_type, labels):
    if not rows:
        return {"n_records": 0}
    if task_type == "event_classification":
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for row in rows:
            matrix += row["confusion"]
        tp = np.diag(matrix)
        fp, fn = matrix.sum(axis=0) - tp, matrix.sum(axis=1) - tp
    else:
        tp = np.sum([r["tp"] for r in rows], axis=0)
        fp = np.sum([r["fp"] for r in rows], axis=0)
        fn = np.sum([r["fn"] for r in rows], axis=0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    report = {"n_records": len(rows), "n_groups": len({r["record"].group_id for r in rows}),
        "scope": "sample_level" if task_type == "event_detection" else "record_level",
        "macro_f1": float(f1.mean()),
        "micro_f1": float(2*tp.sum() / (2*tp.sum()+fp.sum()+fn.sum())) if (2*tp.sum()+fp.sum()+fn.sum()) else 0,
        "per_label": {label: {"tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
            "precision": float(tp[i]/(tp[i]+fp[i])) if tp[i]+fp[i] else 0,
            "recall": float(tp[i]/(tp[i]+fn[i])) if tp[i]+fn[i] else 0, "f1": float(f1[i])}
            for i, label in enumerate(labels)}}
    if task_type == "event_classification":
        report["confusion_matrix"] = matrix.tolist()
        report["accuracy"] = float(np.trace(matrix)/matrix.sum())
    return report


def grouped_metrics(rows, task_type, labels, split_summary):
    def bucket(record):
        if record.snr_db_target is None:
            return "clean" if record.is_synthetic else "unknown"
        return f"{record.snr_db_target:g} dB"
    holdouts = split_summary.get("holdout_groups", {})
    reasons = split_summary.get("reasons", {})
    report_labels = ["deviation"] if task_type == "anomaly_detection" else labels
    report = {"overall": metrics(rows, task_type, report_labels)}
    for title, key in [
        ("by_snr_db_target", lambda r: bucket(r)),
        ("by_holdout_rule", lambda r: reasons.get(r.group_id, "holdout") if r.group_id in holdouts else "not_holdout"),
        ("by_machine_id", lambda r: r.machine_id),
    ]:
        groups = {}
        for row in rows:
            groups.setdefault(key(row["record"]), []).append(row)
        report[title] = {name: metrics(members, task_type, report_labels) for name, members in groups.items()}
    return report
