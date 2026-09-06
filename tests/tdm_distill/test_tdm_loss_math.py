"""Tests for TDM's pure-tensor loss math (Eq. 7, Eq. 11 of arXiv:2503.06674)."""

import math

import pytest
import torch

from boo_musubi_tuner.tdm_distill.tdm_distill import (
    cfg_combine,
    fake_score_denoising_loss,
    min_snr_weight,
    pseudo_huber_c,
    pseudo_huber_loss,
    revised_sample,
)


def test_pseudo_huber_c_matches_paper_constant():
    assert pseudo_huber_c(100) == pytest.approx(0.00054 * math.sqrt(100))


def test_pseudo_huber_loss_zero_when_identical():
    x = torch.randn(3, 4, 5)
    loss = pseudo_huber_loss(x, x.clone(), c=pseudo_huber_c(20))
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_pseudo_huber_loss_positive_when_different():
    student_x = torch.zeros(2, 4)
    revised_x = torch.ones(2, 4)
    loss = pseudo_huber_loss(student_x, revised_x, c=pseudo_huber_c(4))
    assert loss.item() > 0.0


def test_pseudo_huber_loss_scales_with_c():
    student_x = torch.zeros(2, 4)
    revised_x = torch.ones(2, 4)
    loss_small_c = pseudo_huber_loss(student_x, revised_x, c=0.01)
    loss_large_c = pseudo_huber_loss(student_x, revised_x, c=10.0)
    # larger c dampens the effective gradient magnitude near small residuals -> smaller loss value
    assert loss_large_c.item() < loss_small_c.item()


def test_revised_sample_matches_formula():
    x_ti = torch.tensor([[1.0, 2.0]])
    real_score = torch.tensor([[3.0, 3.0]])
    fake_score = torch.tensor([[1.0, 1.0]])
    lambda_tau = torch.tensor([[0.5]])
    result = revised_sample(x_ti, real_score, fake_score, lambda_tau)
    expected = x_ti + 0.5 * (real_score - fake_score)
    assert torch.allclose(result, expected)


def test_revised_sample_scalar_lambda():
    x_ti = torch.zeros(2, 3)
    real_score = torch.ones(2, 3)
    fake_score = torch.zeros(2, 3)
    result = revised_sample(x_ti, real_score, fake_score, lambda_tau=2.0)
    assert torch.allclose(result, torch.full((2, 3), 2.0))


def test_fake_score_denoising_loss_zero_when_matching():
    pred = torch.randn(4, 8)
    loss = fake_score_denoising_loss(pred, pred.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_fake_score_denoising_loss_weighted():
    pred = torch.zeros(2, 2)
    target = torch.ones(2, 2)
    unweighted = fake_score_denoising_loss(pred, target, omega_tau=1.0)
    weighted = fake_score_denoising_loss(pred, target, omega_tau=2.0)
    assert weighted.item() == pytest.approx(2.0 * unweighted.item())


def test_cfg_combine_matches_formula():
    cond = torch.tensor([[2.0, 4.0]])
    uncond = torch.tensor([[1.0, 1.0]])
    result = cfg_combine(cond, uncond, guidance_scale=3.0)
    expected = uncond + 3.0 * (cond - uncond)
    assert torch.allclose(result, expected)


def test_cfg_combine_scale_one_recovers_cond():
    cond = torch.tensor([[2.0, 4.0]])
    uncond = torch.tensor([[1.0, 1.0]])
    result = cfg_combine(cond, uncond, guidance_scale=1.0)
    assert torch.allclose(result, cond)


def test_min_snr_weight_at_low_tau_clamps_to_gamma_term():
    tau = 0.05
    gamma = 5.0
    result = min_snr_weight(tau, gamma=gamma)
    assert result == pytest.approx(gamma * tau**2)


def test_min_snr_weight_at_high_tau_uses_snr_term():
    tau = 0.95
    gamma = 5.0
    result = min_snr_weight(tau, gamma=gamma)
    assert result == pytest.approx((1 - tau) ** 2)


def test_min_snr_weight_matches_closed_form_at_midpoint():
    tau = 0.5
    gamma = 5.0
    result = min_snr_weight(tau, gamma=gamma)
    expected = min((1 - tau) ** 2, gamma * tau**2)
    assert result == pytest.approx(expected)


def test_min_snr_weight_default_gamma_is_five():
    tau = 0.5
    assert min_snr_weight(tau) == pytest.approx(min((1 - tau) ** 2, 5.0 * tau**2))


def test_fake_score_denoising_loss_accepts_min_snr_weight():
    pred = torch.tensor([[2.0, 4.0]])
    target = torch.tensor([[1.0, 1.0]])
    tau = 0.5
    weight = min_snr_weight(tau)
    result = fake_score_denoising_loss(pred, target, omega_tau=weight)
    expected = (weight * (pred - target) ** 2).mean()
    assert torch.allclose(result, expected)
