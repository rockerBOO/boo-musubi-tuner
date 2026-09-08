"""--dino_diversity_pass1_chunk_size lets pass 1 of the memory-efficient diversity path process
the group in sub-batches instead of one batched forward over the whole group_size -- pass 1's
no_grad forward still costs peak activation memory proportional to its batch size even though no
backward graph is retained, so group_size (diversity quality) and peak VRAM (chunk size) need to
be independent knobs."""

import pytest
import torch
from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT

from tests.self_flow.conftest_k2_self_flow import make_k2_batch

from .test_dino_diversity_process_batch import (
    _prepared_trainer,
    _StubDinov3Embedder,
    _StubVae,
    make_noise_scheduler,
)


def test_pass1_runs_in_chunks_not_one_batched_forward(tiny_k2_config):
    """With group_size=4 and pass1_chunk_size=2, pass 1's rollout must be called twice with
    batch size 2 each, not once with batch size 4."""
    torch.manual_seed(4)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    trainer, args, acc, net, handle = _prepared_trainer(
        model,
        dino_diversity_group_size=4,
        dino_diversity_memory_efficient=True,
        dino_diversity_pass1_chunk_size=2,
    )
    for p in model.parameters():
        p.requires_grad_(True)
    trainer._dinov3_embedder = _StubDinov3Embedder()

    rollout_batch_sizes = []
    orig_rollout = trainer._rollout

    def spy_rollout(*a, **kw):
        batch = a[3]
        rollout_batch_sizes.append(batch["latents"].shape[0])
        return orig_rollout(*a, **kw)

    trainer._rollout = spy_rollout

    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    trainer.process_batch(args, acc, model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0)
    handle.remove()

    # Pass 1: two chunked calls of batch size 2. Pass 2: four per-sample calls of batch size 1.
    assert rollout_batch_sizes == [2, 2, 1, 1, 1, 1]


def test_pass1_no_chunk_size_runs_group_in_one_forward(tiny_k2_config):
    """Default (no --dino_diversity_pass1_chunk_size) behavior is unchanged: pass 1 still runs
    the whole group in one batched forward."""
    torch.manual_seed(5)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    trainer, args, acc, net, handle = _prepared_trainer(
        model,
        dino_diversity_group_size=4,
        dino_diversity_memory_efficient=True,
    )
    for p in model.parameters():
        p.requires_grad_(True)
    trainer._dinov3_embedder = _StubDinov3Embedder()

    rollout_batch_sizes = []
    orig_rollout = trainer._rollout

    def spy_rollout(*a, **kw):
        batch = a[3]
        rollout_batch_sizes.append(batch["latents"].shape[0])
        return orig_rollout(*a, **kw)

    trainer._rollout = spy_rollout

    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    trainer.process_batch(args, acc, model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0)
    handle.remove()

    assert rollout_batch_sizes == [4, 1, 1, 1, 1]


def _attach_and_run(tiny_k2_config, seed, chunk_size):
    torch.manual_seed(seed)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    trainer, args, acc, net, handle = _prepared_trainer(
        model,
        dino_diversity_group_size=4,
        dino_diversity_memory_efficient=True,
        dino_diversity_pass1_chunk_size=chunk_size,
    )
    trainer._dinov3_embedder = _StubDinov3Embedder()
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    loss, metrics = trainer.process_batch(
        args, acc, model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()
    return loss, metrics, net


def test_chunked_pass1_matches_unchunked_gradient(tiny_k2_config):
    """Chunking pass 1 is a VRAM optimization, not an approximation -- same noise draw, same
    per-sample trajectories, just computed in smaller sub-batches, so the resulting gradient and
    diversity loss value must match the unchunked (chunk_size=None) path exactly for the same
    seed."""
    loss_unchunked, metrics_unchunked, net_unchunked = _attach_and_run(tiny_k2_config, seed=7, chunk_size=None)
    loss_unchunked.backward()
    grad_unchunked = net_unchunked.lora_w.grad.clone()

    loss_chunked, metrics_chunked, net_chunked = _attach_and_run(tiny_k2_config, seed=7, chunk_size=2)
    loss_chunked.backward()
    grad_chunked = net_chunked.lora_w.grad.clone()

    assert torch.allclose(grad_unchunked, grad_chunked, atol=1e-5, rtol=1e-4)
    assert metrics_unchunked["loss/diversity"] == pytest.approx(metrics_chunked["loss/diversity"], abs=1e-5)
