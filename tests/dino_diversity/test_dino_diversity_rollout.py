"""Tests for Krea2DinoDiversityNetworkTrainer._rollout: a full-gradient Euler rollout, unlike
TDM's _student_rollout there is no grad_from_step split -- every step must stay in the autograd
graph since the whole trajectory feeds the diversity loss."""

import torch

from boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity import Krea2DinoDiversityNetworkTrainer
from tests.dino_diversity.conftest import FakeAccelerator
from tests.self_flow.conftest_k2_self_flow import make_k2_batch

from .test_dino_diversity_arg_validation import make_args


class DummyArgs:
    gradient_checkpointing = False
    dino_diversity = False
    blocks_to_swap = 0


def test_rollout_returns_correct_lengths_and_finite(tiny_k2_model):
    torch.manual_seed(0)
    trainer = Krea2DinoDiversityNetworkTrainer()
    acc = FakeAccelerator()
    batch, _, _ = make_k2_batch(B=1, H=8, W=8, n_txt=3)

    trajectory, timesteps = trainer._rollout(
        DummyArgs(),
        acc,
        tiny_k2_model,
        batch,
        num_steps=4,
        device="cpu",
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    assert len(trajectory) == 5
    assert len(timesteps) == 5
    for x in trajectory:
        assert torch.isfinite(x).all()


def test_rollout_uses_batch_latent_resolution(tiny_k2_model):
    """Regression: rollout must not hardcode a fixed latent resolution."""
    torch.manual_seed(0)
    trainer = Krea2DinoDiversityNetworkTrainer()
    acc = FakeAccelerator()
    batch, _, _ = make_k2_batch(B=1, H=12, W=16, n_txt=3)

    trajectory, _ = trainer._rollout(
        DummyArgs(),
        acc,
        tiny_k2_model,
        batch,
        num_steps=2,
        device="cpu",
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    for x in trajectory:
        assert x.shape[-2:] == (12, 16)


def test_rollout_is_fully_grad_enabled_at_every_step(tiny_k2_model):
    """Unlike TDM's _student_rollout, there's no grad_from_step gate here -- every rollout step
    must carry gradients back to the transformer's params, including the very first one."""
    torch.manual_seed(0)
    trainer = Krea2DinoDiversityNetworkTrainer()
    acc = FakeAccelerator()
    batch, _, _ = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)

    trajectory, _ = trainer._rollout(
        DummyArgs(),
        acc,
        tiny_k2_model,
        batch,
        num_steps=4,
        device="cpu",
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    # trajectory[0] is the raw noise draw (never touches the model), so it has no grad_fn; every
    # subsequent step is the result of a differentiable Euler update through the model.
    assert trajectory[0].requires_grad is False
    for x in trajectory[1:]:
        assert x.requires_grad is True


def test_rollout_uses_supplied_noise_instead_of_drawing_fresh(tiny_k2_model):
    torch.manual_seed(0)
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args()
    trainer.handle_model_specific_args(args)
    acc = FakeAccelerator()
    batch, _latents, _noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)

    lat_h, lat_w = batch["latents"].shape[-2], batch["latents"].shape[-1]
    fixed_noise = torch.zeros(1, tiny_k2_model.config.channels, 1, lat_h, lat_w)
    trajectory, _ts = trainer._rollout(
        args,
        acc,
        tiny_k2_model,
        batch,
        num_steps=2,
        device=torch.device("cpu"),
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
        noise=fixed_noise,
    )
    assert torch.equal(trajectory[0], fixed_noise)


def test_rollout_draws_fresh_noise_when_not_supplied(tiny_k2_model):
    torch.manual_seed(0)
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args()
    trainer.handle_model_specific_args(args)
    acc = FakeAccelerator()
    batch, _latents, _noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)

    trajectory, _ts = trainer._rollout(
        args,
        acc,
        tiny_k2_model,
        batch,
        num_steps=2,
        device=torch.device("cpu"),
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    assert not torch.allclose(trajectory[0], torch.zeros_like(trajectory[0]))


def test_call_dit_resets_block_swap_when_enabled(monkeypatch):
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(blocks_to_swap=2)

    calls = []

    class DummyTransformer:
        def prepare_block_swap_before_forward(self):
            calls.append("reset")

    acc = FakeAccelerator()

    import boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity as dino_module

    monkeypatch.setattr(
        dino_module.Krea2NetworkTrainer,
        "call_dit",
        lambda self, *a, **kw: "super-called",
    )

    result = trainer.call_dit(args, acc, DummyTransformer(), None, {}, None, None, None, torch.float32)
    assert calls == ["reset"]
    assert result == "super-called"


def test_call_dit_skips_block_swap_reset_when_disabled(monkeypatch):
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(blocks_to_swap=None)

    calls = []

    class DummyTransformer:
        def prepare_block_swap_before_forward(self):
            calls.append("reset")

    acc = FakeAccelerator()

    import boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity as dino_module

    monkeypatch.setattr(
        dino_module.Krea2NetworkTrainer,
        "call_dit",
        lambda self, *a, **kw: "super-called",
    )

    result = trainer.call_dit(args, acc, DummyTransformer(), None, {}, None, None, None, torch.float32)
    assert calls == []
    assert result == "super-called"


def test_call_dit_vanilla_fallthrough_when_dino_diversity_off(monkeypatch):
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity=False, blocks_to_swap=2)

    calls = []

    class DummyTransformer:
        def prepare_block_swap_before_forward(self):
            calls.append("reset")

    acc = FakeAccelerator()

    import boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity as dino_module

    monkeypatch.setattr(
        dino_module.Krea2NetworkTrainer,
        "call_dit",
        lambda self, *a, **kw: "super-called",
    )

    result = trainer.call_dit(args, acc, DummyTransformer(), None, {}, None, None, None, torch.float32)
    assert calls == []
    assert result == "super-called"
