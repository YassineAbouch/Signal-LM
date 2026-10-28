"""Data preparation for Signal-LM: synthetic generation, import, split, quality.

This package produces *traceable* datasets (NPZ + JSONL manifest). It does
not train, normalise or interpret anything; see docs/DATASET_PLAN.md.
"""
from .schema import EventAnnotation, Record, read_manifest, write_manifest
from .store import generate_dataset, load_signal
from .split import assign_splits, check_leakage
from .quality import run_quality

__all__ = ["EventAnnotation", "Record", "read_manifest", "write_manifest", "generate_dataset",
           "load_signal", "assign_splits", "check_leakage", "run_quality"]
