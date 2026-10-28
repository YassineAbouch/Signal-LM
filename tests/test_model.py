import pytest
import torch

from signal_lm.model import ModelConfig, build_virgin, load_virgin, save_virgin


def test_checkpoint_roundtrip_and_no_overwrite(tmp_path):
    cfg = ModelConfig(input_channels=2, num_classes=3)
    path = tmp_path / "virgin.pt"
    report = save_virgin(cfg, path)
    assert report["status"] == "untrained"
    assert 0 < report["parameters"] < 100_000
    original = build_virgin(cfg).eval()
    restored = load_virgin(path)
    x = torch.randn(2, 128, 2)
    with torch.no_grad():
        for key in original(x):
            torch.testing.assert_close(original(x)[key], restored(x)[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_virgin(cfg, path)


def test_reproducible_initialization_and_rng_preserved():
    before = torch.get_rng_state().clone()
    a = build_virgin(ModelConfig(seed=7))
    assert torch.equal(before, torch.get_rng_state())
    b = build_virgin(ModelConfig(seed=7))
    c = build_virgin(ModelConfig(seed=8))
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), c.parameters()))


def test_backward_support_and_variable_lengths():
    model = build_virgin(ModelConfig(num_classes=3))
    for length in (64, 129):
        output = model(torch.randn(2, length, 1))
        assert output["embedding"].shape == (2, 64)
        assert output["logits"].shape == (2, 3)
        torch.nn.functional.cross_entropy(output["logits"], torch.tensor([0, 2])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_default_encoder_has_no_classification_head():
    model = build_virgin(ModelConfig())
    assert set(model(torch.zeros(1, 32, 1))) == {"embedding"}
    with pytest.raises(ValueError):
        model(torch.zeros(1, 32, 2))


@pytest.mark.parametrize("kwargs", [{"widths": []}, {"input_channels": 0},
    {"kernel_size": 2}, {"num_classes": -1}, {"embedding_dim": 1.5}])
def test_invalid_model_config(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_reject_trained_checkpoint(tmp_path):
    path = tmp_path / "trained.pt"
    torch.save({"format_version": 1, "status": "trained", "training_steps": 10}, path)
    with pytest.raises(ValueError):
        load_virgin(path)
