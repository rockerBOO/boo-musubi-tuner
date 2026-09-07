import torch
from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

import boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill as tdm_module
from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import Krea2TdmDistillNetworkTrainer
from boo_musubi_tuner.tdm_distill.tdm_distill import cfg_combine
from tests.self_flow.conftest_k2_self_flow import make_k2_batch
from tests.tdm_distill.conftest import FakeAccelerator, StubLoraNetwork

from .test_tdm_arg_validation import make_args


def make_noise_scheduler(args):
    return FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")


def _attach_stub_lora(model, net):
    """Wire StubLoraNetwork into the tiny model's forward so its parameter actually receives
    gradients, the way a real LoRA does. Respects `net.multiplier`, so the teacher role
    (multiplier 0) genuinely reduces to the raw base model.

    The contribution is deliberately *quadratic* in `lora_w` so that autograd has to save
    `lora_w` itself for the backward pass — matching a real LoRA's chained down->up Linears,
    where the up weight is saved to backprop into the down weight. That makes this stub
    sensitive to the same hazard real LoRAs face: any in-place role swap
    (`load_state_dict`) between a grad-enabled forward and its deferred `backward()` bumps
    the parameter's version counter and makes autograd raise."""

    def hook(_module, _inputs, output):
        return output + net.multiplier * 0.01 * (net.lora_w * net.lora_w).sum()

    return model.register_forward_hook(hook)


def _prepared_trainer(tiny_k2_model, **arg_overrides):
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_step_counts="2,4", learning_rate=1e-4, optimizer_type="AdamW", **arg_overrides)
    acc = FakeAccelerator()
    net = StubLoraNetwork(init_value=1.0)
    net.load_weights = lambda path: "ok"
    trainer.handle_model_specific_args(args)
    if args.tdm_guidance_scale > 1.0:
        # Mirrors the real trainer's call order: process_sample_prompts (which populates
        # _teacher_uncond_embed for CFG) runs before on_train_start.
        trainer.process_sample_prompts(args, acc, args.sample_prompts)
    trainer.on_train_start(args, acc, net, tiny_k2_model, None)
    handle = _attach_stub_lora(tiny_k2_model, net)
    return trainer, args, acc, net, handle


def test_process_batch_tdm_smoke(tiny_k2_model):
    torch.manual_seed(0)
    trainer, args, acc, net, _handle = _prepared_trainer(tiny_k2_model)
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    loss, metrics = trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "loss/tdm" in metrics and "loss/fake_score" in metrics and "tdm/k" in metrics

    loss.backward()
    assert net.lora_w.grad is not None


def test_process_batch_fake_score_optimizer_stepped(tiny_k2_model):
    torch.manual_seed(1)
    trainer, args, acc, net, _handle = _prepared_trainer(tiny_k2_model)
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    fake_score_before = {k: v.clone() for k, v in trainer._role_switcher.fake_score_state.items()}
    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    fake_score_after = trainer._role_switcher.fake_score_state
    assert not torch.equal(fake_score_before["lora_w"], fake_score_after["lora_w"])


def test_process_batch_leaves_student_role_active(tiny_k2_model):
    """process_batch must return with the student role live: the base trainer's outer loop
    backward()/optimizer.step() operates on whatever weights are currently on `net`."""
    torch.manual_seed(2)
    trainer, args, acc, net, _handle = _prepared_trainer(tiny_k2_model)
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    assert net.multiplier == 1.0
    assert torch.equal(net.lora_w.detach(), trainer._role_switcher.student_state["lora_w"])


def test_process_batch_resamples_step_count_across_calls(tiny_k2_model):
    """K must be redrawn per iteration (TDM Eq. 8). Regression guard: a freshly constructed
    torch.Generator carries a fixed default seed, so per-call generator construction would
    silently pin K (and the interval) to the same value on every training step."""
    torch.manual_seed(4)
    trainer, args, acc, net, _handle = _prepared_trainer(tiny_k2_model)
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    seen = set()
    for _ in range(8):
        _, metrics = trainer.process_batch(
            args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
        )
        seen.add(metrics["tdm/k"])
    assert seen == {2.0, 4.0}


class _StubVae:
    """Minimal VAE stand-in. decode_to_pixels is a cheap but *differentiable* function of the
    latents (channel-mean -> per-channel scaling -> sigmoid into [0, 1]), so gradient can flow
    from the decoded pixels back into the rollout that produced the latents — exercising the
    real decode's graph-carrying behaviour without a real autoencoder."""

    dtype = torch.float32

    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: (N, C, 1, 8, 8) -> (N, 3, 8, 8) in [0, 1], same layout as the real VAE's.
        n = latents.shape[0]
        base = latents.reshape(n, -1, 8, 8).mean(dim=1, keepdim=True)
        return torch.sigmoid(torch.cat([base, base * 0.5, base * 2.0], dim=1))


class _StubDinov3Embedder:
    """Minimal DINOv3 embedder stand-in. Deliberately exposes only embed_differentiable (no
    `embed`), so a regression back to the numpy/no_grad `embed()` path fails loudly here. The
    embedding is a plain differentiable reduction of the pixel tensor — no model download."""

    def embed_differentiable(self, pixel_values: torch.Tensor) -> torch.Tensor:
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        return flat[:, :8] + flat.mean(dim=1, keepdim=True)


def _run_diversity_process_batch(tiny_k2_config, seed, **arg_overrides):
    """One seeded process_batch call through the diversity block, returning
    (loss, metrics, net, rollout_calls).

    Builds its OWN model from the config rather than taking the shared `tiny_k2_model` fixture:
    callers compare two runs at the same seed, and reusing one model instance would stack a
    second `_attach_stub_lora` forward hook on the second run — the runs would then differ
    because of hook contamination rather than the effect under test. Seeding before
    construction means both models start from identical weights. The hook is removed after the
    call for good measure.
    """
    torch.manual_seed(seed)
    tiny_k2_model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    tiny_k2_model.eval()
    trainer, args, acc, net, handle = _prepared_trainer(tiny_k2_model, tdm_diversity_group_size=3, **arg_overrides)
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    trainer._dinov3_embedder = _StubDinov3Embedder()
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    rollout_calls = []
    orig_rollout = trainer._student_rollout

    def spy_rollout(*a, **kw):
        trajectory, ts = orig_rollout(*a, **kw)
        rollout_calls.append({"grad_from_step": kw.get("grad_from_step"), "trajectory": trajectory})
        return trajectory, ts

    trainer._student_rollout = spy_rollout

    loss, metrics = trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()
    return loss, metrics, net, rollout_calls


def test_process_batch_diversity_term_gradient_flows_to_student(tiny_k2_config):
    """Exercises the diversity block (vae is not None) end-to-end with stub VAE/embedder.

    Confirms: no KeyError/IndexError from single_prompt_batch's missing/mis-sized "latents";
    the diversity metrics land in the returned dict as plain floats (not graph-carrying
    tensors); and — the point of the term — the whole rollout -> VAE decode -> embed ->
    diversity-loss chain is differentiable, so gradient reaches the student LoRA parameter."""
    loss, metrics, net, rollout_calls = _run_diversity_process_batch(tiny_k2_config, seed=5)

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert isinstance(metrics["loss/diversity"], float)
    assert isinstance(metrics["tdm/diversity_score"], float)
    assert metrics["tdm/diversity_score"] == -metrics["loss/diversity"]

    # Two rollouts happen: the main student rollout (step 1) and the group-diversity rollout.
    assert len(rollout_calls) == 2
    _main_call, diversity_call = rollout_calls
    # The diversity rollout must be fully grad-enabled (grad_from_step=0) so the diversity loss
    # can backprop through every Euler step into the LoRA weights.
    assert diversity_call["grad_from_step"] == 0
    assert diversity_call["trajectory"][-1].requires_grad

    loss.backward()
    assert net.lora_w.grad is not None


def test_process_batch_diversity_weight_changes_student_gradient(tiny_k2_config):
    """Stronger guard that the diversity term is a live training signal and not dead weight:
    at the same seed, the student LoRA's gradient must differ between diversity_weight=0 (block
    skipped entirely) and a nonzero weight. A non-differentiable diversity path would leave the
    two gradients identical."""
    loss_off, _, net_off, _ = _run_diversity_process_batch(tiny_k2_config, seed=7, tdm_diversity_weight=0.0)
    loss_off.backward()
    grad_off = net_off.lora_w.grad.clone()

    loss_on, _, net_on, _ = _run_diversity_process_batch(tiny_k2_config, seed=7, tdm_diversity_weight=1.0)
    loss_on.backward()
    grad_on = net_on.lora_w.grad.clone()

    assert grad_off is not None and grad_on is not None
    assert not torch.allclose(grad_off, grad_on)


def test_process_batch_vanilla_fallthrough(tiny_k2_model):
    torch.manual_seed(3)
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False)
    acc = FakeAccelerator()
    net = StubLoraNetwork()
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    loss, metrics = trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics == {}


def test_process_batch_cfg_off_single_teacher_forward(tiny_k2_model):
    """tdm_guidance_scale <= 1.0 must cost exactly one teacher forward -- no perf regression."""
    torch.manual_seed(6)
    trainer, args, acc, net, _handle = _prepared_trainer(tiny_k2_model, tdm_guidance_scale=1.0)
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    teacher_calls = []
    orig_call_dit = trainer.call_dit

    def spy_call_dit(*a, **kw):
        if net.multiplier == 0.0:
            teacher_calls.append(1)
        return orig_call_dit(*a, **kw)

    trainer.call_dit = spy_call_dit
    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    assert len(teacher_calls) == 1


def test_process_batch_cfg_on_two_teacher_forwards_and_combines(tiny_k2_model, monkeypatch):
    """tdm_guidance_scale > 1.0 must run cond + uncond teacher forwards and combine via cfg_combine."""
    torch.manual_seed(6)
    # _prepared_trainer's process_sample_prompts runs the real CFG uncond-embed caching path when
    # tdm_guidance_scale > 1.0 and text_encoder is set -- stub the encoder load and prompt file read
    # so they don't try to touch real files. Shape matches tiny_k2_config's txtlayers=1, txtdim=32.
    monkeypatch.setattr(tdm_module.krea2_utils, "load_krea2_text_encoder", lambda path, dtype, device: object())
    monkeypatch.setattr(
        tdm_module.krea2_utils,
        "get_krea2_prompt_embeds",
        lambda encoder, prompts: (torch.randn(1, 2, 1, 32), torch.ones(1, 2, dtype=torch.bool)),
    )
    monkeypatch.setattr(tdm_module, "load_prompts", lambda path: [{"prompt": "a cat"}])
    trainer, args, acc, net, _handle = _prepared_trainer(
        tiny_k2_model, tdm_guidance_scale=3.5, text_encoder="/path/to/qwen3_vl.safetensors"
    )
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    teacher_preds = []
    orig_call_dit = trainer.call_dit

    def spy_call_dit(*a, **kw):
        output = orig_call_dit(*a, **kw)
        if net.multiplier == 0.0:
            teacher_preds.append(output.pred.detach().clone())
        return output

    trainer.call_dit = spy_call_dit

    revised_sample_calls = []
    orig_revised_sample = tdm_module.revised_sample

    def spy_revised_sample(x_ti, real_score, fake_score_updated, lambda_tau):
        revised_sample_calls.append(real_score.detach().clone())
        return orig_revised_sample(x_ti, real_score, fake_score_updated, lambda_tau)

    monkeypatch.setattr(tdm_module, "revised_sample", spy_revised_sample)

    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    # Two teacher forwards this step: cond then uncond.
    assert len(teacher_preds) == 2
    cond_score, uncond_score = teacher_preds
    assert not torch.allclose(cond_score, uncond_score)

    # The real_score actually consumed downstream (by revised_sample) must be the cfg_combine
    # output, not a discarded/bypassed cond_score. This catches an implementation that silently
    # drops the cfg_combine call and leaves real_score == cond_score.
    assert len(revised_sample_calls) == 1
    expected_real_score = cfg_combine(cond_score, uncond_score, args.tdm_guidance_scale)
    assert torch.allclose(revised_sample_calls[0], expected_real_score)
    assert not torch.allclose(revised_sample_calls[0], cond_score)


def test_process_batch_fake_score_loss_uses_min_snr_weight(tiny_k2_model, monkeypatch):
    """process_batch's fake-score loss call must pass a tau-dependent omega_tau, not the default 1.0."""
    torch.manual_seed(6)
    trainer, args, acc, net, _handle = _prepared_trainer(tiny_k2_model, tdm_guidance_scale=1.0)
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    calls = []
    orig_loss_fn = tdm_module.fake_score_denoising_loss

    def spy_loss_fn(fake_score_pred, target, omega_tau=1.0):
        calls.append(omega_tau)
        return orig_loss_fn(fake_score_pred, target, omega_tau=omega_tau)

    monkeypatch.setattr(tdm_module, "fake_score_denoising_loss", spy_loss_fn)
    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    assert len(calls) == 1
    assert calls[0] != 1.0
    assert 0.0 <= calls[0] <= 5.0
