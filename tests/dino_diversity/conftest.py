"""Shared fixtures for dino_diversity extension tests. Mirrors tests/tdm_distill/conftest.py's
tiny-model pattern (self-contained per extension, no shared root conftest)."""

import pytest
import torch
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT


@pytest.fixture
def tiny_k2_config():
    """Minimal K2 config for fast CPU tests — not real weights. features=32, heads=2 ->
    headdim=16 -> axes=[4,6,6] (sum=16, all even), the constraint SingleStreamDiT asserts on."""
    return SingleMMDiTConfig(
        features=32,
        tdim=32,
        txtdim=32,
        heads=2,
        multiplier=1,
        layers=2,
        patch=2,
        channels=4,
        bias=False,
        theta=1e3,
        kvheads=None,
        txtlayers=1,
        txtheads=2,
        txtkvheads=2,
    )


@pytest.fixture
def tiny_k2_model(tiny_k2_config):
    torch.manual_seed(0)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    return model


class FakeAccelerator:
    device = torch.device("cpu")

    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def autocast(self):
        return self._NullCtx()

    def unwrap_model(self, m):
        return m

    def prepare(self, *objs):
        return objs[0] if len(objs) == 1 else objs

    def print(self, *args, **kwargs):
        pass

    def backward(self, loss, gradient=None):
        loss.backward(gradient=gradient)


class StubLoraNetwork(torch.nn.Module):
    """Minimal stand-in for musubi_tuner.networks.lora.LoRANetwork: a single trainable
    parameter plus a multiplier, enough to exercise basic LoRA logic without a real LoRA module."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.lora_w = torch.nn.Parameter(torch.full((4,), init_value))
        self.multiplier = 1.0

    def set_multiplier(self, multiplier: float) -> None:
        self.multiplier = multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.multiplier * self.lora_w
