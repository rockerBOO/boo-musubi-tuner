"""Tests for Krea2DinoDiversityNetworkTrainer.on_train_start and on_post_optimizer_step."""

from boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity import Krea2DinoDiversityNetworkTrainer
from tests.dino_diversity.conftest import FakeAccelerator, StubLoraNetwork

from .test_dino_diversity_arg_validation import make_args


def test_on_train_start_resets_dinov3_embedder():
    trainer = Krea2DinoDiversityNetworkTrainer()
    trainer._dinov3_embedder = "stale"
    args = make_args()
    net = StubLoraNetwork()
    acc = FakeAccelerator()

    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)

    assert trainer._dinov3_embedder is None


def test_on_train_start_noop_without_dino_diversity():
    trainer = Krea2DinoDiversityNetworkTrainer()
    trainer._dinov3_embedder = "stale"
    args = make_args(dino_diversity=False)
    net = StubLoraNetwork()
    acc = FakeAccelerator()

    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)

    assert trainer._dinov3_embedder == "stale"


def test_on_post_optimizer_step_returns_vae_to_cpu_when_flagged():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args()
    acc = FakeAccelerator()

    class DummyVae:
        def __init__(self):
            self.device_calls = []

        def to(self, device):
            self.device_calls.append(device)

    vae = DummyVae()
    trainer._vae_ref = vae
    trainer._vae_needs_cpu_return = True

    trainer.handle_model_specific_args(args)
    trainer.on_post_optimizer_step(args, acc, None, None, True, 1)

    assert vae.device_calls == ["cpu"]
    assert trainer._vae_ref is None
    assert trainer._vae_needs_cpu_return is False


def test_on_post_optimizer_step_noop_when_not_flagged():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args()
    acc = FakeAccelerator()

    class DummyVae:
        def __init__(self):
            self.device_calls = []

        def to(self, device):
            self.device_calls.append(device)

    vae = DummyVae()
    trainer._vae_ref = vae
    trainer._vae_needs_cpu_return = False

    trainer.handle_model_specific_args(args)
    trainer.on_post_optimizer_step(args, acc, None, None, True, 1)

    assert vae.device_calls == []
    assert trainer._vae_ref is vae


def test_on_post_optimizer_step_noop_without_dino_diversity():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity=False)
    acc = FakeAccelerator()

    class DummyVae:
        def to(self, device):
            raise AssertionError("should not be called when dino_diversity is off")

    trainer._vae_ref = DummyVae()
    trainer._vae_needs_cpu_return = True

    trainer.handle_model_specific_args(args)
    trainer.on_post_optimizer_step(args, acc, None, None, True, 1)

    assert trainer._vae_needs_cpu_return is True
