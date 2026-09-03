"""Tests for TDM's step-count/interval sampling and the DINOv3 diversity metric."""

import pytest
import torch

from boo_musubi_tuner.tdm_distill.tdm_distill import (
    pairwise_cosine_diversity,
    sample_step_count,
    sample_trajectory_interval,
)


def test_sample_step_count_always_from_list():
    gen = torch.Generator().manual_seed(0)
    step_counts = [1, 2, 4, 8]
    seen = {sample_step_count(step_counts, generator=gen) for _ in range(50)}
    assert seen.issubset(set(step_counts))
    assert len(seen) > 1  # sanity: not degenerate to a single value over 50 draws


def test_sample_step_count_single_value_list():
    gen = torch.Generator().manual_seed(0)
    assert sample_step_count([4], generator=gen) == 4


def test_sample_trajectory_interval_in_range():
    gen = torch.Generator().manual_seed(1)
    for _ in range(50):
        i = sample_trajectory_interval(num_steps=8, generator=gen)
        assert 0 <= i < 7


def test_sample_trajectory_interval_two_steps_only_zero():
    gen = torch.Generator().manual_seed(2)
    assert sample_trajectory_interval(num_steps=2, generator=gen) == 0


def test_sample_trajectory_interval_single_step_returns_zero():
    assert sample_trajectory_interval(num_steps=1) == 0


def test_pairwise_cosine_diversity_identical_embeddings_is_zero():
    embeddings = torch.ones(4, 8)
    assert pairwise_cosine_diversity(embeddings) == pytest.approx(0.0, abs=1e-6)


def test_pairwise_cosine_diversity_orthogonal_embeddings_is_one():
    embeddings = torch.eye(4, 8)
    assert pairwise_cosine_diversity(embeddings) == pytest.approx(1.0, abs=1e-6)


def test_pairwise_cosine_diversity_opposite_pair_is_two():
    embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    assert pairwise_cosine_diversity(embeddings) == pytest.approx(2.0, abs=1e-6)


def test_pairwise_cosine_diversity_raises_below_two_samples():
    with pytest.raises(ValueError, match="N >= 2"):
        pairwise_cosine_diversity(torch.ones(1, 8))
