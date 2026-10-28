"""Data preparation CLI: ``python -m signal_lm.data.cli <command>``.

Commands
    generate   render a synthetic dataset from a config (config.json + manifest)
    split      assign train/validation/test by group (in place in the manifest)
    window     cut fixed windows from parent records (after split)
    validate   run quality checks and write quality_report.{json,md}
    import-wav / import-csv   add one real recording to a dataset root
    show       print a short summary of a manifest

No command trains anything or touches model weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .importers import import_csv, import_wav
from .quality import run_quality, write_report
from .schema import read_manifest, write_manifest
from .split import assign_splits, check_leakage
from .store import generate_dataset
from .windows import window_record


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_generate(args: argparse.Namespace) -> dict:
    config = _load_json(args.config)
    records = generate_dataset(config, args.root, overwrite=args.overwrite)
    return {"dataset_id": config["dataset_id"], "root": str(args.root), "n_records": len(records)}


def cmd_split(args: argparse.Namespace) -> dict:
    records = read_manifest(args.root / "manifest.jsonl")
    split_config = _load_json(args.config) if args.config else {}
    summary = assign_splits(records, seed=split_config.get("seed", args.seed),
                            fractions=split_config.get("fractions"),
                            holdout=split_config.get("holdout"),
                            stratify_by=split_config.get("stratify_by", "session_id"))
    leaks = check_leakage(records)
    if leaks:
        raise SystemExit("Leakage detected after split: " + "; ".join(leaks[:5]))
    write_manifest(records, args.root / "manifest.jsonl")
    (args.root / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary.pop("reasons", None)
    return summary


def cmd_window(args: argparse.Namespace) -> dict:
    records = read_manifest(args.root / "manifest.jsonl")
    parents = [r for r in records if r.parent_record_id is None]
    if len(parents) != len(records):
        raise SystemExit("manifest already contains windows; regenerate or split again before re-windowing")
    windows = []
    for record in parents:
        windows.extend(window_record(args.root, record, window_samples=args.window_samples,
                                     hop_samples=args.hop_samples))
    write_manifest(parents + windows, args.root / "manifest.jsonl")
    return {"n_parents": len(parents), "n_windows": len(windows)}


def cmd_validate(args: argparse.Namespace) -> dict:
    records = read_manifest(args.root / "manifest.jsonl")
    problems, report = run_quality(args.root, records, verify_regeneration=args.regenerate,
                                   verify_hash=not args.skip_hash)
    json_path, md_path = write_report(args.root, report)
    errors = [p for p in problems if p["severity"] == "error"]
    result = {"n_records": len(records), "n_errors": len(errors),
              "n_warnings": len(problems) - len(errors), "report": str(md_path), "json": str(json_path)}
    if errors and args.strict:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    return result


def _import_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--specialist", required=True)
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--group-id", required=True, help="machine/session/acquisition unit kept in one split")
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--license", default="unknown")
    parser.add_argument("--machine-id", default="unknown")
    parser.add_argument("--session-id", default="unknown")
    parser.add_argument("--acquisition-id", default="unknown")
    parser.add_argument("--notes", default="")


def _append_record(root: Path, record) -> dict:
    manifest = root / "manifest.jsonl"
    records = read_manifest(manifest) if manifest.exists() else []
    if any(r.record_id == record.record_id for r in records):
        raise SystemExit(f"record_id {record.record_id} already in manifest")
    records.append(record)
    write_manifest(records, manifest)
    return {"record_id": record.record_id, "n_samples": record.n_samples,
            "sample_rate_hz": record.sample_rate_hz, "units": record.units, "n_records": len(records)}


def cmd_import_wav(args: argparse.Namespace) -> dict:
    args.root.mkdir(parents=True, exist_ok=True)
    record = import_wav(args.root, args.path, record_id=args.record_id, source_id=args.source_id,
                        specialist=args.specialist, task_type=args.task_type, group_id=args.group_id,
                        labels=args.labels, license=args.license, machine_id=args.machine_id,
                        session_id=args.session_id, acquisition_id=args.acquisition_id, notes=args.notes)
    return _append_record(args.root, record)


def cmd_import_csv(args: argparse.Namespace) -> dict:
    args.root.mkdir(parents=True, exist_ok=True)
    record = import_csv(args.root, args.path, record_id=args.record_id, source_id=args.source_id,
                        specialist=args.specialist, task_type=args.task_type, group_id=args.group_id,
                        sample_rate_hz=args.sample_rate, columns=args.columns, units=args.units,
                        delimiter=args.delimiter, labels=args.labels, license=args.license,
                        machine_id=args.machine_id, session_id=args.session_id,
                        acquisition_id=args.acquisition_id, notes=args.notes)
    return _append_record(args.root, record)


def cmd_show(args: argparse.Namespace) -> dict:
    from collections import Counter
    records = read_manifest(args.root / "manifest.jsonl")
    return {"n_records": len(records),
            "by_split": dict(Counter(r.split for r in records)),
            "by_task": dict(Counter(f"{r.specialist}/{r.task_type}" for r in records)),
            "labels": dict(Counter(l for r in records for l in r.labels))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal_lm.data", description="Signal-LM data preparation (no training)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="render a synthetic dataset")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("split", help="assign splits by group")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--config", type=Path, help="JSON with seed, fractions, holdout rules")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("window", help="cut fixed windows after split")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--window-samples", type=int, required=True)
    p.add_argument("--hop-samples", type=int, required=True)
    p.set_defaults(func=cmd_window)

    p = sub.add_parser("validate", help="quality checks + report")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--regenerate", action="store_true", help="re-render synthetic records and compare")
    p.add_argument("--skip-hash", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 1 on errors")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("import-wav")
    _import_common(p)
    p.set_defaults(func=cmd_import_wav)

    p = sub.add_parser("import-csv")
    _import_common(p)
    p.add_argument("--sample-rate", type=float, required=True)
    p.add_argument("--columns", nargs="+", required=True)
    p.add_argument("--units", nargs="+")
    p.add_argument("--delimiter", default=",")
    p.set_defaults(func=cmd_import_csv)

    p = sub.add_parser("show")
    p.add_argument("--root", type=Path, required=True)
    p.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main(sys.argv[1:])
