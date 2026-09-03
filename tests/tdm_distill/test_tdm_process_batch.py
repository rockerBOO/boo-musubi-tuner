import torch
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import Krea2TdmDistillNetworkTrainer
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
    trainer.on_train_start(args, acc, net, tiny_k2_model, None)
    _attach_stub_lora(tiny_k2_model, net)
    return trainer, args, acc, net


def test_process_batch_tdm_smoke(tiny_k2_model):
    torch.manual_seed(0)
    trainer, args, acc, net = _prepared_trainer(tiny_k2_model)
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
    trainer, args, acc, net = _prepared_trainer(tiny_k2_model)
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
    trainer, args, acc, net = _prepared_trainer(tiny_k2_model)
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
    trainer, args, acc, net = _prepared_trainer(tiny_k2_model)
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
    """Minimal VAE stand-in: decode_to_pixels ignores its input shape/content and returns a
    fixed-size pixel tensor, enough to exercise the diversity block's decode->image->embed
    pipeline without a real autoencoder."""

    dtype = torch.float32

    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        n = latents.shape[0]
        return torch.rand(n, 3, 8, 8, dtype=torch.float32)


class _StubDinov3Embedder:
    """Minimal DINOv3 embedder stand-in: returns a fixed-shape embedding per image, avoiding
    any real model download."""

    def embed(self, images: list) -> torch.Tensor:
        return torch.randn(len(images), 8)


def test_process_batch_diversity_term_end_to_end(tiny_k2_model):
    """Exercises the diversity block (vae is not None) end-to-end with stub VAE/embedder.
    Confirms: no KeyError/IndexError from single_prompt_batch's missing/mis-sized "latents"
    (Bug 1), the diversity metrics land in the returned dict, and the group rollout used for
    the diversity term does not build an autograd graph (Bug 2: grad_from_step must match
    num_steps, not 0)."""
    torch.manual_seed(5)
    trainer, args, acc, net = _prepared_trainer(tiny_k2_model, tdm_diversity_group_size=3)
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

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "loss/diversity" in metrics
    assert "tdm/diversity_score" in metrics

    # Two rollouts happen: the main student rollout (step 1) and the group-diversity rollout.
    assert len(rollout_calls) == 2
    main_call, diversity_call = rollout_calls
    # Bug 2 fix: the diversity rollout must use the "no grad needed" convention
    # (grad_from_step == num_steps), matching the main rollout, not grad_from_step=0.
    assert diversity_call["grad_from_step"] == main_call["grad_from_step"]
    assert all(not t.requires_grad for t in diversity_call["trajectory"])

    loss.backward()
    assert net.lora_w.grad is not None


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
