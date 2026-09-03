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
