import torch

from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import Krea2TdmDistillNetworkTrainer
from tests.self_flow.conftest_k2_self_flow import make_k2_batch
from tests.tdm_distill.conftest import FakeAccelerator


def test_student_rollout_returns_correct_lengths_and_finite(tiny_k2_model):
    torch.manual_seed(0)
    trainer = Krea2TdmDistillNetworkTrainer()
    acc = FakeAccelerator()
    batch, _, _ = make_k2_batch(B=1, H=8, W=8, n_txt=3)

    class DummyArgs:
        gradient_checkpointing = False

    trajectory, timesteps = trainer._student_rollout(
        DummyArgs(),
        acc,
        tiny_k2_model,
        batch,
        num_steps=4,
        grad_from_step=4,
        device="cpu",
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    assert len(trajectory) == 5
    assert len(timesteps) == 5
    for x in trajectory:
        assert torch.isfinite(x).all()


def test_student_rollout_uses_batch_latent_resolution(tiny_k2_model):
    """Regression: the rollout used to hardcode 8x8 latents, ignoring the batch's real size."""
    torch.manual_seed(0)
    trainer = Krea2TdmDistillNetworkTrainer()
    acc = FakeAccelerator()
    batch, _, _ = make_k2_batch(B=1, H=12, W=16, n_txt=3)

    class DummyArgs:
        gradient_checkpointing = False

    trajectory, _ = trainer._student_rollout(
        DummyArgs(),
        acc,
        tiny_k2_model,
        batch,
        num_steps=2,
        grad_from_step=2,
        device="cpu",
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    for x in trajectory:
        assert x.shape[-2:] == (12, 16)


def test_student_rollout_no_grad_before_grad_from_step(tiny_k2_model):
    torch.manual_seed(0)
    trainer = Krea2TdmDistillNetworkTrainer()
    acc = FakeAccelerator()
    batch, _, _ = make_k2_batch(B=1, H=8, W=8, n_txt=3)

    class DummyArgs:
        gradient_checkpointing = False

    trajectory, _ = trainer._student_rollout(
        DummyArgs(),
        acc,
        tiny_k2_model,
        batch,
        num_steps=4,
        grad_from_step=3,
        device="cpu",
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
    )
    # steps 0..2 built under no_grad -> not part of any graph; step 3 (index 3, i.e. the 4th
    # entry) onward should carry grad since transformer params require grad by default in this test
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    assert trajectory[0].requires_grad is False
    assert trajectory[2].requires_grad is False
