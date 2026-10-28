"""Randomly initialized CNN. Embeddings/logits are not expert predictions."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 1
    widths: tuple[int, ...] = (16, 32, 64)
    kernel_size: int = 7
    embedding_dim: int = 64
    num_classes: int = 0
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "widths", tuple(self.widths))
        positive = (self.input_channels, *self.widths, self.kernel_size, self.embedding_dim)
        if not self.widths or any(type(v) is not int or v <= 0 for v in positive):
            raise ValueError("Model dimensions must be positive integers.")
        if self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd.")
        if type(self.num_classes) is not int or self.num_classes < 0:
            raise ValueError("num_classes must be a nonnegative integer.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")


class SignalEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        layers = []
        channels = config.input_channels
        for width in config.widths:
            layers.extend([nn.Conv1d(channels, width, config.kernel_size,
                                     padding=config.kernel_size // 2), nn.GELU()])
            channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(channels, config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.num_classes) if config.num_classes else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Public tensor convention also uses samples x channels.
        if x.ndim != 3 or x.shape[2] != self.config.input_channels or x.shape[1] < 4:
            raise ValueError("Expected [batch, samples >= 4, configured channels].")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("Input tensor must contain finite floating-point values.")
        embedding = self.project(self.pool(self.features(x.transpose(1, 2))).squeeze(-1))
        result = {"embedding": embedding}
        if self.head is not None:
            result["logits"] = self.head(embedding)
        return result


def build_virgin(config: ModelConfig) -> SignalEncoder:
    # Initialize on CPU reproducibly without altering the caller's RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(config.seed)
        return SignalEncoder(config)


def save_virgin(config: ModelConfig, path: str | Path) -> dict:
    """Always build fresh weights: cannot relabel a trained instance as virgin."""
    model = build_virgin(config)
    payload = {"format_version": 1, "status": "untrained", "training_steps": 0,
               "config": asdict(config), "state_dict": model.state_dict()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return {"status": "untrained", "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint": str(path)}


def load_virgin(path: str | Path) -> SignalEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format_version") != 1 or payload.get("status") != "untrained"
            or payload.get("training_steps") != 0):
        raise ValueError("Expected a version-1 untrained checkpoint.")
    model = build_virgin(ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()
