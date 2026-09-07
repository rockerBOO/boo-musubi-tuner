import torch

from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import Krea2TdmDistillNetworkTrainer
from tests.self_flow.conftest_k2_self_flow import make_k2_batch
from tests.tdm_distill.conftest import FakeAccelerator

from .test_tdm_arg_validation import make_args


def test_student_rollout_uses_supplied_noise_instead_of_drawing_fresh(tiny_k2_model):
    torch.manual_seed(0)
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args()
    trainer.handle_model_specific_args(args)
    acc = FakeAccelerator()
    batch, _latents, _noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)

    lat_h, lat_w = batch["latents"].shape[-2], batch["latents"].shape[-1]
    fixed_noise = torch.zeros(1, tiny_k2_model.config.channels, 1, lat_h, lat_w)
    trajectory, _ts = trainer._student_rollout(
        args,
        acc,
        tiny_k2_model,
        batch,
        num_steps=2,
        grad_from_step=2,
        device=torch.device("cpu"),
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
        noise=fixed_noise,
    )
    assert torch.equal(trajectory[0], fixed_noise)


def test_student_rollout_draws_fresh_noise_when_not_supplied(tiny_k2_model):
    torch.manual_seed(0)
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args()
    trainer.handle_model_specific_args(args)
    acc = FakeAccelerator()
    batch, _latents, _noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)

    trajectory, _ts = trainer._student_rollout(
        args,
        acc,
        tiny_k2_model,
        batch,
        num_steps=2,
        grad_from_step=2,
        device=torch.device("cpu"),
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    # No supplied noise -> starting point is a fresh torch.randn draw, essentially never
    # all-zeros.
    assert not torch.allclose(trajectory[0], torch.zeros_like(trajectory[0]))
