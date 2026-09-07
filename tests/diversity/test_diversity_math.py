"""Tests for architecture-generic diversity math: pairwise cosine diversity and the
differentiable group-diversity loss built on it. Dinov3ImageEmbedder itself is
integration-only (constructing it downloads real pretrained weights), so it has no unit
test here."""

import pytest
import torch

from boo_musubi_tuner.diversity.diversity import (
    diversity_loss_from_embeddings,
    pairwise_cosine_diversity,
)


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


def test_diversity_loss_is_negative_mean_pairwise_distance():
    embeddings = torch.eye(4, 8)  # orthogonal -> pairwise_cosine_diversity == 1.0
    loss = diversity_loss_from_embeddings(embeddings)
    assert loss == -1.0


def test_diversity_loss_more_diverse_is_more_negative():
    orthogonal = torch.eye(4, 8)
    identical = torch.ones(4, 8)
    assert diversity_loss_from_embeddings(orthogonal) < diversity_loss_from_embeddings(identical)
