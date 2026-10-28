"""Explicit fit and frozen evaluation commands. No auto-training on import."""
import argparse
import json
from pathlib import Path

import torch

from .engine import evaluate_checkpoint, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a specialist from a split manifest")
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--specialist", required=True)
    fit.add_argument("--task-type", required=True, choices=["event_classification", "event_detection", "multilabel_classification", "anomaly_detection"])
    fit.add_argument("--level", choices=["parents", "windows"], default="windows")
    fit.add_argument("--epochs", type=int, default=1)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--learning-rate", type=float, default=.001)
    fit.add_argument("--reference-quantile", type=float, default=.95)
    fit.add_argument("--allow-real-training", action="store_true")
    final = sub.add_parser("evaluate")
    final.add_argument("--checkpoint", type=Path, required=True)
    final.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["test"])
    for command in [fit, final]:
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--threads", type=int, default=1)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    threads = args.pop("threads")
    if threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(threads)
    try:
        payload = train(**args) if command == "fit" else evaluate_checkpoint(**args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": payload["status"], "specialist": payload["specialist"],
        "task_type": payload["task_type"], "training_steps": payload["training_steps"],
        "selected_epoch": payload["selected_epoch"], "provenance": payload["provenance"],
        "evaluations": payload["evaluations"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
