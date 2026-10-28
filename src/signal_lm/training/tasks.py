"""Task-specific heads; the original virgin encoder remains unchanged."""
import torch
from torch import nn

from signal_lm.model import ModelConfig, build_virgin


class TemporalDetector(nn.Module):
    def __init__(self, config, n_labels):
        super().__init__()
        self.config = config
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.features = build_virgin(config).features
            self.temporal_head = nn.Conv1d(config.widths[-1], n_labels, 1)

    def forward(self, x):
        if (x.ndim != 3 or x.shape[1] < 4 or x.shape[2] != self.config.input_channels
                or not x.is_floating_point() or not torch.isfinite(x).all()):
            raise ValueError("Expected finite [batch, samples >= 4, configured channels]")
        return {"logits": self.temporal_head(self.features(x.transpose(1, 2))).transpose(1, 2)}


def build_task_model(config, task_type, n_labels):
    if task_type == "event_detection":
        return TemporalDetector(config, n_labels)
    return build_virgin(config)


def task_config(channels, task_type, n_labels, seed):
    return ModelConfig(input_channels=channels, seed=seed,
                       num_classes=n_labels if task_type in {"event_classification", "multilabel_classification"} else 0)


def task_loss(output, target, task_type, center=None):
    if task_type == "event_classification":
        return nn.functional.cross_entropy(output["logits"], target)
    if task_type in {"multilabel_classification", "event_detection"}:
        return nn.functional.binary_cross_entropy_with_logits(output["logits"], target)
    if task_type == "anomaly_detection":
        if torch.any(target != 0):
            raise ValueError("Anomaly fitting permits reference examples only")
        return ((output["embedding"] - center) ** 2).sum(dim=1).mean()
    raise ValueError("Unsupported task")
