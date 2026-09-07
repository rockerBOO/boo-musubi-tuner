"""Tests for Krea2DinoDiversityNetworkTrainer._build_network: the dino_diversity LoRA is built
with per-module dim/alpha inferred directly from --dino_diversity_lora_init's own weights."""

import torch

import boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity as dino_module
from boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity import Krea2DinoDiversityNetworkTrainer
from tests.dino_diversity.conftest import FakeAccelerator, StubLoraNetwork

from .test_dino_diversity_arg_validation import make_args


class _FakeNetworkModule:
    def __init__(self, network):
        self._network = network
        self.create_arch_network_from_weights_calls = []
        self.apply_to_calls = []
        self.load_weights_calls = []

    def create_arch_network_from_weights(self, multiplier, weights_sd, unet=None):
        self.create_arch_network_from_weights_calls.append((multiplier, weights_sd, unet))
        return self._network


def test_build_network_infers_shape_and_weights_from_dino_diversity_lora(monkeypatch):
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(network_module="fake.module")
    trainer.handle_model_specific_args(args)

    net = StubLoraNetwork(init_value=0.0)
    apply_to_calls = []
    net.apply_to = lambda *a, **kw: apply_to_calls.append((a, kw))

    def load_weights(path):
        net.lora_w.data.fill_(3.0)
        return f"loaded {path}"

    net.load_weights = load_weights

    fake_module = _FakeNetworkModule(net)
    monkeypatch.setattr(dino_module.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setattr(dino_module, "load_file", lambda path: {"fake_weights_sd": path})

    acc = FakeAccelerator()
    result = trainer._build_network(args, acc, transformer=object(), vae=None, weight_dtype=torch.float32)

    assert result is net
    assert torch.equal(net.lora_w, torch.full((4,), 3.0))
    assert len(fake_module.create_arch_network_from_weights_calls) == 1
    multiplier, weights_sd, _unet = fake_module.create_arch_network_from_weights_calls[0]
    assert multiplier == 1.0
    assert weights_sd == {"fake_weights_sd": args.dino_diversity_lora_init}
    assert len(apply_to_calls) == 1


def test_build_network_vanilla_fallthrough_when_dino_diversity_off(monkeypatch):
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity=False)
    trainer.handle_model_specific_args(args)

    calls = []

    def fake_super_build_network(self, args, accelerator, transformer, vae, weight_dtype):
        calls.append((transformer, vae, weight_dtype))
        return "vanilla-network"

    monkeypatch.setattr(dino_module.Krea2NetworkTrainer, "_build_network", fake_super_build_network)

    acc = FakeAccelerator()
    result = trainer._build_network(args, acc, transformer="tf", vae="vae", weight_dtype=torch.float32)

    assert result == "vanilla-network"
    assert calls == [("tf", "vae", torch.float32)]
