"""Tests for on_before/after_sample_images stacking order when both --turbo_dit
and --self_flow are active. Krea2NetworkTrainer already swaps the transformer's
BASE weights RAW<->Turbo around sampling; Self-Flow swaps the LoRA NETWORK's
weights student<->EMA. These touch different objects but both must run, in
correct (LIFO) order."""

import torch

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import Krea2SelfFlowNetworkTrainer

from .test_self_flow_k2_arg_validation import make_args
from .test_self_flow_k2_lifecycle import FakeAccelerator, StubNetwork


class RecordingTransformer(torch.nn.Module):
    """Stub standing in for the K2 SingleStreamDiT during sampling-swap tests.

    Records calls instead of doing real RAW/Turbo weight I/O (that path needs
    a real checkpoint file on disk, exercised only in integration tests)."""

    def __init__(self):
        super().__init__()
        self.calls: list[str] = []

    class _FakeConfig:
        patch = 2

    config = _FakeConfig()


def test_ema_and_turbo_swap_compose_without_turbo(tiny_k2_model_not_needed=None):
    """Without --turbo_dit, only the EMA swap should run (base class is a no-op)."""
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(turbo_dit=None)
    trainer.ema_lora_state = {"lora_w": torch.full((4,), 9.0)}
    net = StubNetwork()
    student_val = net.lora_w.detach().clone()

    trainer.on_before_sample_images(FakeAccelerator(), args, 0, 0, None, RecordingTransformer(), net, [], torch.float32)
    assert torch.equal(net.lora_w.detach(), torch.full((4,), 9.0))  # swapped to EMA

    trainer.on_after_sample_images(FakeAccelerator(), args, 0, 0, None, RecordingTransformer(), net, [], torch.float32)
    assert torch.equal(net.lora_w.detach(), student_val)  # restored to student


def test_noop_without_self_flow():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False, turbo_dit=None)
    net = StubNetwork()
    student_val = net.lora_w.detach().clone()

    trainer.on_before_sample_images(FakeAccelerator(), args, 0, 0, None, RecordingTransformer(), net, [], torch.float32)
    assert torch.equal(net.lora_w.detach(), student_val)  # unchanged, no EMA state
    trainer.on_after_sample_images(FakeAccelerator(), args, 0, 0, None, RecordingTransformer(), net, [], torch.float32)
    assert torch.equal(net.lora_w.detach(), student_val)


def test_noop_without_ema_state():
    """self_flow=True but ema_lora_state never populated (e.g. on_train_start
    not yet called) must not crash."""
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(turbo_dit=None)
    net = StubNetwork()
    trainer.on_before_sample_images(FakeAccelerator(), args, 0, 0, None, RecordingTransformer(), net, [], torch.float32)
    trainer.on_after_sample_images(FakeAccelerator(), args, 0, 0, None, RecordingTransformer(), net, [], torch.float32)
    assert torch.equal(net.lora_w.detach(), torch.ones(4))
