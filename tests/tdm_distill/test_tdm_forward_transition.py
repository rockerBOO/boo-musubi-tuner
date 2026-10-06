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
