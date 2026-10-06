import math

import torch

from boo_musubi_tuner.tdm_distill.tdm_distill import critic_importance_weight, forward_transition


def test_transition_matches_marginal_noise_level():
    # x_t = (1-t)*x0 + t*eps1; after the transition the noise share must be tau (unit variance noise).
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 16, 1, 32, 32, generator=g)
    eps1 = torch.randn_like(x0, generator=g)
    noise = torch.randn_like(x0, generator=g)
    t, tau = 0.3, 0.6
    x_t = (1 - t) * x0 + t * eps1
    x_tau, a = forward_transition(x_t, t, tau, noise)
    assert math.isclose(a, (1 - tau) / (1 - t))
    mixed = (x_tau - (1 - tau) * x0) / tau
    assert abs(mixed.std().item() - 1.0) < 0.05


def test_transition_from_clean_equals_plain_noising():
    x0 = torch.randn(2, 16, 1, 8, 8)
    noise = torch.randn_like(x0)
    x_tau, a = forward_transition(x0, 0.0, 0.5, noise)
    assert torch.allclose(x_tau, 0.5 * x0 + 0.5 * noise)
    assert a == 0.5


def test_importance_weight_is_one_for_identical_noise():
    n = torch.randn(3, 16, 1, 8, 8)
    assert torch.allclose(critic_importance_weight(n, n), torch.ones(3))


def test_mottle_loss_zero_for_identical_and_for_smoother_student():
    from boo_musubi_tuner.tdm_distill.tdm_distill import gaussian_blur_latent, mottle_excess_loss

    torch.manual_seed(0)
    ref = torch.randn(2, 16, 1, 32, 32)
    loss, ratio = mottle_excess_loss(ref.clone(), ref)
    assert loss.item() == 0.0 and abs(ratio.item() - 1.0) < 1e-4
    smooth = gaussian_blur_latent(ref, 2.0)
    loss_s, ratio_s = mottle_excess_loss(smooth, ref)
    assert loss_s.item() == 0.0 and ratio_s.item() < 1.0


def test_mottle_loss_positive_for_noisy_student_and_has_grad():
    from boo_musubi_tuner.tdm_distill.tdm_distill import gaussian_blur_latent, mottle_excess_loss

    torch.manual_seed(1)
    ref = gaussian_blur_latent(torch.randn(2, 16, 1, 32, 32), 3.0)
    student = (ref + 0.2 * torch.randn_like(ref)).requires_grad_(True)
    loss, ratio = mottle_excess_loss(student, ref)
    assert loss.item() > 0.0 and ratio.item() > 1.0
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
