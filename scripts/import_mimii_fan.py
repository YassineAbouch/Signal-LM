"""Import the verified official archive through the existing WAV importer, no training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from signal_lm.data.importers import import_wav
from signal_lm.data.schema import read_manifest, sha256_of_file, write_manifest
from signal_lm.data.split import assign_splits, check_leakage

from download_mimii import MD5, SIZE, checksum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/0_dB_fan.zip"))
    parser.add_argument("--root", type=Path, default=Path("data/mimii_fan_0db"))
    parser.add_argument("--max-per-state", type=int, default=25,
                        help="first N WAV per machine/state; 0 imports all files")
    args = parser.parse_args()
    if args.max_per_state < 0:
        parser.error("max-per-state must be nonnegative")
    if args.archive.stat().st_size != SIZE or checksum(args.archive) != MD5:
        raise ValueError("Archive differs from official size/MD5")
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    config = {"dataset_id": "mimii_fan_0db", "seed": 7, "source": "zenodo.3384388",
              "version": "public 1.0", "archive_md5": MD5, "license": "CC BY-SA 4.0",
              "group_id": "machine_id", "snr_db_target": 0,
              "reference_labels": ["normal"], "deviation_labels": ["abnormal"],
              "max_per_machine_state": args.max_per_state, "selection": "sorted_archive_member_names_before_evaluation"}
    config_path = root/"config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing import has a different config")
    else:
        with config_path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
    manifest = root/"manifest.jsonl"
    records = read_manifest(manifest) if manifest.exists() else []
    existing = {r.record_id: r for r in records}
    try:
        with zipfile.ZipFile(args.archive) as archive, tempfile.TemporaryDirectory(prefix="wav-import-", dir=root) as scratch:
            entries = sorted((e for e in archive.infolist() if e.filename.lower().endswith(".wav")), key=lambda e: e.filename)
            if args.max_per_state:
                counts, selected = defaultdict(int), []
                for entry in entries:
                    parts = Path(entry.filename).parts
                    key = (next((p for p in parts if p.startswith("id_")), None),
                           next((p for p in parts if p in {"normal", "abnormal"}), None))
                    if counts[key] < args.max_per_state:
                        counts[key] += 1
                        selected.append(entry)
                entries = selected
            for index, entry in enumerate(entries, 1):
                parts = Path(entry.filename).parts
                machine_part = next((p for p in parts if p.startswith("id_")), None)
                state = next((p for p in parts if p in {"normal", "abnormal"}), None)
                if machine_part is None or state is None or "fan" not in parts:
                    raise ValueError(f"Unexpected archive member: {entry.filename}")
                machine_id = "fan_" + machine_part
                record_id = f"mimii_fan_0db__{machine_id}__{state}__{Path(entry.filename).stem}"
                if record_id in existing:
                    if sha256_of_file(root/existing[record_id].file_path) != existing[record_id].file_sha256:
                        raise ValueError(f"Existing imported file changed: {record_id}")
                    continue
                if (root/"records"/f"{record_id}.npz").exists():
                    raise FileExistsError(f"Unmanifested NPZ preserved: {record_id}")
                wav = Path(scratch)/"record.wav"
                with archive.open(entry) as source, wav.open("wb") as target:
                    shutil.copyfileobj(source, target)
                record = import_wav(root, wav, record_id=record_id, source_id="mimii_fan_0db",
                    specialist="acoustic", task_type="anomaly_detection", group_id=machine_id,
                    machine_id=machine_id, acquisition_id=record_id, labels=[state],
                    license="CC BY-SA 4.0 (https://zenodo.org/records/3384388)",
                    extra_provenance={"original_path": f"0_dB_fan.zip!{entry.filename}",
                        "source_url": "https://zenodo.org/records/3384388", "version": "public 1.0",
                        "archive_md5": MD5, "snr_db_target": 0},
                    notes="Source normal/abnormal labels only; no mechanical diagnosis inferred. Acquisition date unknown.")
                record.snr_db_target = 0
                records.append(record)
                existing[record_id] = record
                if index % 100 == 0:
                    write_manifest(records, manifest)
                    print(f"Imported {index}/{len(entries)} WAV files", flush=True)
        machines = {r.machine_id for r in records}
        if machines != {"fan_id_00", "fan_id_02", "fan_id_04", "fan_id_06"}:
            raise ValueError(f"Unexpected machine IDs: {sorted(machines)}")
        summary = assign_splits(records, seed=7, fractions={"train": 1., "validation": 0., "test": 0.},
            stratify_by=None, holdout=[
                {"field": "machine_id", "equals": "fan_id_04", "split": "validation", "reason": "held-out validation machine fan_id_04"},
                {"field": "machine_id", "equals": "fan_id_06", "split": "test", "reason": "held-out test machine fan_id_06"}])
        problems = check_leakage(records)
        if problems:
            raise ValueError("; ".join(problems[:5]))
        summary["strategy"] = "whole-machine holdout; no within-machine stratification (group_id=machine_id)"
        (root/"split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"n_records": len(records), "counts": summary["counts"], "root": str(root)}, indent=2), flush=True)
    finally:
        write_manifest(records, manifest)


if __name__ == "__main__":
    main()
