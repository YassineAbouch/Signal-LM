"""Verify the default virgin checkpoint and CUDA without optimizer steps."""
import json
from pathlib import Path

import torch

from signal_lm.model import ModelConfig, build_virgin, load_virgin, save_virgin


def main():
    path = Path("artifacts/virgin.pt")
    config = ModelConfig()
    if not path.exists():
        save_virgin(config, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["status"] == "untrained" and payload["training_steps"] == 0
    original = build_virgin(config).eval()
    restored = load_virgin(path)
    x = torch.linspace(-1, 1, 4096).reshape(1, 4096, 1)
    with torch.no_grad():
        expected = original(x)["embedding"]
        actual = restored(x)["embedding"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    before = path.read_bytes()
    try:
        save_virgin(config, path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing checkpoint was overwritten")
    assert path.read_bytes() == before
    report = {
        "checkpoint": str(path), "status": payload["status"],
        "training_steps": payload["training_steps"],
        "parameters": sum(p.numel() for p in restored.parameters()),
        "cpu_roundtrip_exact": True, "overwrite_refused": True,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        restored = restored.cuda()
        before_weights = {k: v.clone() for k, v in restored.state_dict().items()}
        gpu_x = x.cuda().requires_grad_()
        output = restored(gpu_x)["embedding"]
        assert output.is_cuda and torch.isfinite(output).all()
        torch.testing.assert_close(output.detach().cpu(), expected, rtol=1e-4, atol=1e-5)
        output.square().mean().backward()
        assert gpu_x.grad is not None and torch.isfinite(gpu_x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in restored.parameters())
        assert all(torch.equal(v, before_weights[k]) for k, v in restored.state_dict().items())
        torch.cuda.synchronize()
        report.update(gpu=torch.cuda.get_device_name(), cuda_forward_backward_finite=True,
                      cuda_matches_cpu=True, weights_unchanged=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
