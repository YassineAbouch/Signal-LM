import argparse
import json
from pathlib import Path

import numpy as np

from .analysis import analyze
from .signal import Signal, load_csv


def main():
    parser = argparse.ArgumentParser(description="Signal-LM: virgin model and physical measurements")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Save fresh, untrained weights")
    init.add_argument("--config", type=Path, default=Path("configs/base.json"))
    init.add_argument("--output", type=Path, required=True)
    commands.add_parser("doctor", help="Inspect Python/PyTorch/GPU")
    demo = commands.add_parser("demo", help="Measure a synthetic 120 Hz sine; no training")
    demo.add_argument("--spectrum", action="store_true")
    csv_cmd = commands.add_parser("analyze-csv")
    csv_cmd.add_argument("path", type=Path)
    csv_cmd.add_argument("--sample-rate", type=float, required=True)
    csv_cmd.add_argument("--columns", nargs="+", required=True)
    csv_cmd.add_argument("--units", nargs="+")
    csv_cmd.add_argument("--delimiter", default=",")
    csv_cmd.add_argument("--spectrum", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "init":
            from .model import ModelConfig, save_virgin
            result = save_virgin(ModelConfig(**json.loads(args.config.read_text(encoding="utf-8"))), args.output)
        elif args.command == "doctor":
            import platform
            import torch
            result = {"python": platform.python_version(), "platform": platform.platform(),
                      "torch": torch.__version__, "cuda_build": torch.version.cuda,
                      "cuda_available": torch.cuda.is_available(),
                      "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
        elif args.command == "demo":
            t = np.arange(4096) / 4096
            result = analyze(Signal(np.sin(2 * np.pi * 120 * t), 4096, ("synthetic",), ("a.u.",)),
                             include_spectrum=args.spectrum)
        else:
            signal = load_csv(args.path, sample_rate_hz=args.sample_rate, columns=args.columns,
                              units=args.units, delimiter=args.delimiter)
            result = analyze(signal, include_spectrum=args.spectrum)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
