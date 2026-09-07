import torch
from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

from boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity import Krea2DinoDiversityNetworkTrainer
from tests.dino_diversity.conftest import FakeAccelerator, StubLoraNetwork
from tests.self_flow.conftest_k2_self_flow import make_k2_batch

from .test_dino_diversity_arg_validation import make_args


def make_noise_scheduler(args):
    return FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")


def _attach_stub_lora(model, net):
    def hook(_module, _inputs, output):
        return output + net.multiplier * 0.01 * (net.lora_w * net.lora_w).sum()

    return model.register_forward_hook(hook)


class _StubVae:
    dtype = torch.float32

    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def decode_to_pixels(self, latents: torch.Tensor) -> torch.Tensor:
        n = latents.shape[0]
        base = latents.reshape(n, -1, 8, 8).mean(dim=1, keepdim=True)
        return torch.sigmoid(torch.cat([base, base * 0.5, base * 2.0], dim=1))


class _StubDinov3Embedder:
    def embed_differentiable(self, pixel_values: torch.Tensor) -> torch.Tensor:
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        return flat[:, :8] + flat.mean(dim=1, keepdim=True)


def _prepared_trainer(tiny_k2_model, **arg_overrides):
    trainer = Krea2DinoDiversityNetworkTrainer()
    defaults = {"dino_diversity_step_count": 2, "dino_diversity_group_size": 3}
    defaults.update(arg_overrides)
    args = make_args(**defaults)
    acc = FakeAccelerator()
    net = StubLoraNetwork(init_value=1.0)
    net.load_weights = lambda path: "ok"
    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, tiny_k2_model, None)
    trainer._dinov3_embedder = _StubDinov3Embedder()
    handle = _attach_stub_lora(tiny_k2_model, net)
    return trainer, args, acc, net, handle


def test_process_batch_smoke(tiny_k2_config):
    torch.manual_seed(0)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    trainer, args, acc, net, handle = _prepared_trainer(model)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    loss, metrics = trainer.process_batch(
        args, acc, model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()

    assert loss.ndim == 0 and torch.isfinite(loss)
    assert isinstance(metrics["loss/diversity"], float)
    assert isinstance(metrics["diversity/score"], float)
    assert metrics["diversity/score"] == -metrics["loss/diversity"]


def test_process_batch_gradient_flows_to_lora(tiny_k2_config):
    torch.manual_seed(1)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    trainer, args, acc, net, handle = _prepared_trainer(model)
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    loss, _metrics = trainer.process_batch(
        args, acc, model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()

    loss.backward()
    assert net.lora_w.grad is not None


def test_process_batch_uses_group_size_samples(tiny_k2_config):
    """The diversity rollout must build a --dino_diversity_group_size-sized batch, not reuse
    the incoming batch's own size."""
    torch.manual_seed(2)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    trainer, args, acc, net, handle = _prepared_trainer(model, dino_diversity_group_size=5)

    rollout_calls = []
    orig_rollout = trainer._rollout

    def spy_rollout(*a, **kw):
        result = orig_rollout(*a, **kw)
        rollout_calls.append(a[3])  # batch positional arg
        return result

    trainer._rollout = spy_rollout

    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    trainer.process_batch(args, acc, model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0)
    handle.remove()

    assert len(rollout_calls) == 1
    assert len(rollout_calls[0]["krea2_vl_embed"]) == 5
    assert rollout_calls[0]["latents"].shape[0] == 5


def test_process_batch_vanilla_fallthrough(tiny_k2_config):
    torch.manual_seed(3)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity=False)
    acc = FakeAccelerator()
    net = StubLoraNetwork()
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)

    loss, metrics = trainer.process_batch(
        args, acc, model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, None, global_step=0
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert metrics == {}
