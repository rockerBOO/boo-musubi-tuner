import pytest
import torch
from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT

from tests.self_flow.conftest_k2_self_flow import make_k2_batch
from tests.tdm_distill.test_tdm_process_batch import (
    _prepared_trainer,
    _StubDinov3Embedder,
    _StubVae,
    make_noise_scheduler,
)


def _run(tiny_k2_config, seed, memory_efficient, **arg_overrides):
    torch.manual_seed(seed)
    tiny_k2_model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    tiny_k2_model.eval()
    trainer, args, acc, net, handle = _prepared_trainer(
        tiny_k2_model,
        tdm_diversity_group_size=3,
        tdm_diversity_memory_efficient=memory_efficient,
        **arg_overrides,
    )
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    trainer._dinov3_embedder = _StubDinov3Embedder()
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    loss, metrics = trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()
    return loss, metrics, net


def test_memory_efficient_path_matches_full_path_gradient(tiny_k2_config):
    # tdm_diversity_weight is set far above the default (0.1) so the diversity term's contribution
    # to the gradient dominates the comparison -- at the default weight the diversity term's
    # contribution is smaller than the tolerances below, so the assertion would pass even if the
    # two paths' diversity gradients disagreed completely.
    loss_full, metrics_full, net_full = _run(tiny_k2_config, seed=11, memory_efficient=False, tdm_diversity_weight=1000)
    loss_full.backward()
    grad_full = net_full.lora_w.grad.clone()

    loss_eff, metrics_eff, net_eff = _run(tiny_k2_config, seed=11, memory_efficient=True, tdm_diversity_weight=1000)
    # loss_eff already carries the diversity contribution's backward from inside process_batch;
    # backward() on the returned loss adds only the main TDM student loss's contribution, same
    # as the full path's second backward would.
    loss_eff.backward()
    grad_eff = net_eff.lora_w.grad.clone()

    assert torch.allclose(grad_full, grad_eff, atol=1e-4, rtol=1e-3)
    assert metrics_full["loss/diversity"] == pytest.approx(metrics_eff["loss/diversity"], abs=1e-5)


def test_memory_efficient_path_reuses_pass_one_noise_in_pass_two(tiny_k2_config, monkeypatch):
    torch.manual_seed(13)
    tiny_k2_model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    tiny_k2_model.eval()
    trainer, args, acc, net, handle = _prepared_trainer(
        tiny_k2_model,
        tdm_diversity_group_size=3,
        tdm_diversity_memory_efficient=True,
    )
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    trainer._dinov3_embedder = _StubDinov3Embedder()
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    noise_args_seen = []
    orig_rollout = trainer._student_rollout

    def spy_rollout(*a, **kw):
        noise_args_seen.append(kw.get("noise"))
        return orig_rollout(*a, **kw)

    trainer._student_rollout = spy_rollout

    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()

    # Calls: [0] main student rollout (noise=None), [1] pass 1 batched (noise=noise_batch),
    # [2..4] pass 2 per-sample (noise=noise_batch[i:i+1] for i in 0,1,2).
    pass1_noise = noise_args_seen[1]
    for i in range(3):
        assert torch.equal(noise_args_seen[2 + i], pass1_noise[i : i + 1])


def test_memory_efficient_path_backward_call_count(tiny_k2_config):
    torch.manual_seed(17)
    tiny_k2_model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    tiny_k2_model.eval()
    trainer, args, acc, net, handle = _prepared_trainer(
        tiny_k2_model,
        tdm_diversity_group_size=3,
        tdm_diversity_memory_efficient=True,
    )
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    trainer._dinov3_embedder = _StubDinov3Embedder()
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    backward_calls = []
    orig_backward = acc.backward

    def spy_backward(loss, gradient=None):
        backward_calls.append(gradient is not None)
        return orig_backward(loss, gradient=gradient)

    acc.backward = spy_backward

    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()

    # fake-score loss (step 3, gradient=None) + 3 diversity pass-2 per-sample backprops
    # (gradient=upstream_grad slice, not None). The diversity pass-1 leaf-tensor Jacobian query
    # uses torch.autograd.grad directly (not accelerator.backward), so it doesn't show up here --
    # going through accelerator.backward there would double-apply gradient-accumulation/GradScaler
    # scaling.
    assert backward_calls == [False, True, True, True]
