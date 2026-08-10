import torch
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import Krea2SelfFlowNetworkTrainer
from tests.self_flow.conftest_k2_self_flow import make_k2_batch

from .test_self_flow_k2_arg_validation import make_args
from .test_self_flow_k2_lifecycle import FakeAccelerator, StubNetwork


def make_noise_scheduler(args):
    return FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")


def _prepared_trainer(tiny_k2_model, **arg_overrides):
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(
        student_feature_layer=0,
        teacher_feature_layer=2,
        mask_ratio=0.25,
        self_flow_gamma=0.8,
        ema_decay=0.999,
        **arg_overrides,
    )
    acc = FakeAccelerator()
    trainer.handle_model_specific_args(args)
    trainer.on_transformer_loaded(args, acc, tiny_k2_model)
    trainer.extra_trainable_params(args, acc, None, tiny_k2_model, [])
    net = StubNetwork()
    trainer.on_train_start(args, acc, net, tiny_k2_model, None)
    return trainer, args, acc, net


def test_process_batch_self_flow_smoke(tiny_k2_model):
    torch.manual_seed(0)
    trainer, args, acc, net = _prepared_trainer(tiny_k2_model)
    batch, latents, noise = make_k2_batch(B=2, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    loss, metrics = trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=10
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert "loss/gen" in metrics and "loss/rep" in metrics
    assert trainer._self_flow_logs
    for key in (
        "self_flow/ema_weight_drift",
        "self_flow/mismatch_patch_frac",
        "self_flow/mismatch_patch_count",
        "self_flow/cleaner_fraction_mean",
        "self_flow/actual_mask_ratio",
    ):
        assert key in trainer._self_flow_logs, f"{key} must be logged unconditionally"

    loss.backward()
    grads = [p.grad for p in trainer.rep_proj.parameters()]
    assert all(g is not None for g in grads)


def test_process_batch_restores_student_weights_after_teacher_swap(tiny_k2_model):
    torch.manual_seed(2)
    trainer, args, acc, net = _prepared_trainer(tiny_k2_model)
    trainer.ema_lora_state["lora_w"] = torch.full((4,), 5.0)
    expected_student_val = net.lora_w.detach().clone()

    batch, latents, noise = make_k2_batch(B=2, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=5
    )

    assert torch.allclose(net.lora_w.detach(), expected_student_val), (
        "net.lora_w after process_batch should be the student value — teacher-swap restore likely skipped or misordered"
    )


def test_process_batch_vanilla_fallthrough(tiny_k2_model):
    torch.manual_seed(1)
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    acc = FakeAccelerator()
    net = StubNetwork()
    batch, latents, noise = make_k2_batch(B=2, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    loss, metrics = trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics == {}
