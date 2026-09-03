"""Diversity-term tests. Dinov3ImageEmbedder itself is integration-only (real model download,
no unit test — mirrors the krea2-diversity probe project's Task 4 precedent); this file tests
only the pure-tensor group-diversity-loss composition."""

import torch

from boo_musubi_tuner.tdm_distill.tdm_distill import diversity_loss_from_embeddings


def test_diversity_loss_is_negative_mean_pairwise_distance():
    embeddings = torch.eye(4, 8)  # orthogonal -> pairwise_cosine_diversity == 1.0
    loss = diversity_loss_from_embeddings(embeddings)
    assert loss == -1.0


def test_diversity_loss_more_diverse_is_more_negative():
    orthogonal = torch.eye(4, 8)
    identical = torch.ones(4, 8)
    assert diversity_loss_from_embeddings(orthogonal) < diversity_loss_from_embeddings(identical)
