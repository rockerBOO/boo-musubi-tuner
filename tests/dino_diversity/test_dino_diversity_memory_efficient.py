"""The memory-efficient two-pass diversity path must produce the same gradient as the
batched path -- it's a VRAM optimization, not an approximation."""

import pytest
import torch
from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity import Krea2DinoDiversityNetworkTrainer
from tests.dino_diversity.conftest import FakeAccelerator, StubLoraNetwork
from tests.self_flow.conftest_k2_self_flow import make_k2_batch

from .test_dino_diversity_arg_validation import make_args
from .test_dino_diversity_process_batch import _attach_stub_lora, _StubDinov3Embedder, _StubVae


def make_noise_scheduler(args):
    return FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")


def _run(tiny_k2_config, seed, memory_efficient, **arg_overrides):
    torch.manual_seed(seed)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(
        dino_diversity_step_count=2,
        dino_diversity_group_size=3,
        dino_diversity_memory_efficient=memory_efficient,
        **arg_overrides,
    )
    acc = FakeAccelerator()
    net = StubLoraNetwork(init_value=1.0)
    net.load_weights = lambda path: "ok"
    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, model, None)
    trainer._dinov3_embedder = _StubDinov3Embedder()
    handle = _attach_stub_lora(model, net)

    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    loss, metrics = trainer.process_batch(
        args, acc, model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()
    return loss, metrics, net


def test_memory_efficient_matches_batched_gradient(tiny_k2_config):
    loss_full, metrics_full, net_full = _run(tiny_k2_config, seed=11, memory_efficient=False)
    loss_full.backward()
    grad_full = net_full.lora_w.grad.clone()

    _loss_eff, metrics_eff, net_eff = _run(tiny_k2_config, seed=11, memory_efficient=True)
    # loss_eff already carries the backward from inside process_batch for the memory-efficient
    # path; calling .backward() again here would double-accumulate, so just compare grads as-is
    # after process_batch returns for this path, and the batched path's fresh backward above.
    grad_eff = net_eff.lora_w.grad.clone()

    assert torch.allclose(grad_full, grad_eff, atol=1e-5)
    assert metrics_full["loss/diversity"] == pytest.approx(metrics_eff["loss/diversity"], abs=1e-5)


def test_memory_efficient_backprops_inside_process_batch(tiny_k2_config):
    """Regression guard: the memory-efficient path's backward happens inside process_batch
    itself (per-sample), so net.lora_w.grad must already be populated before the caller ever
    calls loss.backward() on the returned value."""
    _loss, _metrics, net = _run(tiny_k2_config, seed=12, memory_efficient=True)
    assert net.lora_w.grad is not None
