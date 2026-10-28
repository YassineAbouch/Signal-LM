"""Leak-free train / validation / test assignment.

Rules (docs/DATASET_PLAN.md, section "fuites"):

* The unit of assignment is ``group_id`` (machine, session, acquisition, or a
  synthetic parent realisation). Every record of a group goes to one split.
  Windows and augmentations inherit the split of their parent, so splitting
  is done on *parent* records, before windowing.
* Optional ``holdout`` rules force whole groups into one split, e.g. parents
  whose generation parameter falls in an unseen range, or a named machine.
  This is how "generalisation to unseen conditions" is measured.
* Assignment is deterministic given ``seed``.
* ``check_leakage`` verifies that no group, parent or file hash crosses splits.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

from .schema import Record

DEFAULT_FRACTIONS = {"train": 0.7, "validation": 0.15, "test": 0.15}


def _group_rank(group_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _lookup(record: Record, path: str) -> Any:
    """Dotted lookup into a record, e.g. 'generation.components.0.frequency_hz'."""
    value: Any = record
    for part in path.split("."):
        if isinstance(value, Record):
            value = getattr(value, part, None)
        elif isinstance(value, dict):
            value = value.get(part)
        elif isinstance(value, list) and part.isdigit():
            value = value[int(part)] if int(part) < len(value) else None
        else:
            return None
        if value is None:
            return None
    return value


def _rule_matches(record: Record, rule: dict[str, Any]) -> bool:
    value = _lookup(record, rule["field"])
    if value is None:
        return False
    if "in" in rule:
        return value in rule["in"]
    if "range" in rule:
        lo, hi = rule["range"]
        return isinstance(value, (int, float)) and lo <= value <= hi
    if "equals" in rule:
        return value == rule["equals"]
    raise ValueError(f"Unsupported holdout rule: {rule}")


def assign_splits(records: list[Record], *, seed: int = 0,
                  fractions: dict[str, float] | None = None,
                  holdout: list[dict[str, Any]] | None = None,
                  stratify_by: str | None = "session_id") -> dict[str, Any]:
    """Assign ``record.split`` in place by group; returns a summary.

    ``holdout`` entries look like ``{"field": "generation.components.0.frequency_hz",
    "range": [1500, 2000], "split": "test", "reason": "unseen frequency band"}``.
    A group is held out when *any* of its records matches the rule.

    ``stratify_by`` names a record field (default ``session_id``, which is the
    recipe for synthetic data) whose values are each split with the same
    fractions, so every recipe / machine family appears in every split.
    Pass ``None`` to split the whole pool at once.
    """
    fractions = dict(fractions or DEFAULT_FRACTIONS)
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or any(v < 0 for v in fractions.values()):
        raise ValueError("fractions must be non-negative and sum to 1.")
    parents = [r for r in records if r.parent_record_id is None]
    children = [r for r in records if r.parent_record_id is not None]
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in parents:
        groups[record.group_id].append(record)

    reasons: dict[str, str] = {}
    forced: dict[str, str] = {}
    for rule in holdout or []:
        for gid, members in groups.items():
            if gid in forced:
                continue
            if any(_rule_matches(r, rule) for r in members):
                forced[gid] = rule["split"]
                reasons[gid] = rule.get("reason", f"holdout rule on {rule['field']}")

    assignment = dict(forced)
    strata: dict[str, list[str]] = defaultdict(list)
    for gid, members in groups.items():
        if gid in forced:
            continue
        key = str(getattr(members[0], stratify_by)) if stratify_by else "all"
        strata[key].append(gid)
    names = list(fractions)
    for key in sorted(strata):
        remaining = sorted(strata[key], key=lambda g: _group_rank(g, seed))
        n = len(remaining)
        cumulative = 0.0
        start = 0
        for name in names:
            cumulative += fractions[name]
            stop = int(math.floor(cumulative * n + 1e-9))
            for gid in remaining[start:stop]:
                assignment[gid] = name
                reasons[gid] = f"random by group within stratum {key!r} (seed={seed})"
            start = stop
        for gid in remaining[start:]:  # rounding remainder goes to the last split
            assignment[gid] = names[-1]
            reasons[gid] = f"random by group within stratum {key!r} (seed={seed})"

    for record in parents:
        record.split = assignment[record.group_id]
    by_id = {r.record_id: r for r in parents}
    for record in children:
        parent = by_id.get(record.parent_record_id)
        if parent is None:
            raise ValueError(f"{record.record_id}: parent {record.parent_record_id} not found.")
        record.split = parent.split
        record.group_id = parent.group_id

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"groups": 0, "records": 0})
    for gid, split in assignment.items():
        counts[split]["groups"] += 1
    for record in records:
        counts[record.split]["records"] += 1
    return {"seed": seed, "fractions": fractions, "n_groups": len(groups),
            "holdout_groups": {g: s for g, s in forced.items()},
            "reasons": reasons, "counts": dict(counts)}


def check_leakage(records: list[Record], *, ignore_hash_for: set[str] | None = None) -> list[str]:
    """Return problems: a group, parent or identical file present in several splits.

    ``ignore_hash_for`` lists record ids exempt from the identical-file check
    (degenerate constant windows that collide by content, not by origin).
    """
    problems = []
    ignore_hash_for = ignore_hash_for or set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    parent_splits: dict[str, set[str]] = defaultdict(set)
    by_id = {r.record_id: r for r in records}
    for r in records:
        if r.split == "unassigned":
            problems.append(f"{r.record_id}: split unassigned")
        group_splits[r.group_id].add(r.split)
        if r.file_sha256 and r.file_sha256 != "unknown" and r.record_id not in ignore_hash_for:
            hash_splits[r.file_sha256].add(r.split)
        if r.parent_record_id:
            parent = by_id.get(r.parent_record_id)
            if parent is None:
                problems.append(f"{r.record_id}: parent {r.parent_record_id} missing from manifest")
            else:
                parent_splits[r.parent_record_id].add(r.split)
                if parent.split != r.split:
                    problems.append(f"{r.record_id}: split {r.split} != parent split {parent.split}")
    for gid, splits in group_splits.items():
        if len(splits) > 1:
            problems.append(f"group {gid} spans splits {sorted(splits)}")
    for digest, splits in hash_splits.items():
        if len(splits) > 1:
            problems.append(f"identical file {digest[:12]} present in splits {sorted(splits)}")
    return problems
