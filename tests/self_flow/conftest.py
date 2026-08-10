"""Shared fixtures: tiny CPU-only Flux2/K2 models for self-flow extension tests."""

import pytest
import torch

from musubi_tuner.flux_2.flux2_models import Flux2, Flux2Params
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT


@pytest.fixture
def tiny_params():
    """Minimal Flux2 config for fast CPU tests — not real weights.

    hidden_size=16, num_heads=2 -> pe_dim=8, axes_dim must sum to 8.
    """
    return Flux2Params(
        in_channels=4,
        context_in_dim=8,
        hidden_size=16,
        num_heads=2,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[2, 2, 2, 2],
        theta=2000,
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )


@pytest.fixture
def tiny_model(tiny_params):
    torch.manual_seed(0)
    model = Flux2(tiny_params, attn_mode="torch")
    model.eval()
    return model


@pytest.fixture
def tiny_params_guidance():
    """Same as tiny_params but with use_guidance_embed=True (production FLUX.2 dev path)."""
    return Flux2Params(
        in_channels=4,
        context_in_dim=8,
        hidden_size=16,
        num_heads=2,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[2, 2, 2, 2],
        theta=2000,
        mlp_ratio=2.0,
        use_guidance_embed=True,
    )


@pytest.fixture
def tiny_model_guidance(tiny_params_guidance):
    torch.manual_seed(0)
    model = Flux2(tiny_params_guidance, attn_mode="torch")
    model.eval()
    return model


def make_inputs(B=2, N_img=4, N_txt=3, in_channels=4, hidden_ctx=8, seed=0):
    """Build minimal packed inputs for Flux2.forward."""
    torch.manual_seed(seed)
    x = torch.randn(B, N_img, in_channels)
    x_ids = torch.zeros(B, N_img, 4, dtype=torch.long)
    ctx = torch.randn(B, N_txt, hidden_ctx)
    ctx_ids = torch.zeros(B, N_txt, 4, dtype=torch.long)
    guidance = None
    return x, x_ids, ctx, ctx_ids, guidance


@pytest.fixture
def tiny_k2_config():
    """Minimal K2 config for fast CPU tests — not real weights.

    features=32, heads=2 -> headdim=16 -> axes=[4,6,6] (sum=16, all even),
    the constraint SingleStreamDiT.__init__ asserts on. layers=3 gives room
    for student_feature_layer=0 / teacher_feature_layer=2 in tests.
    """
    return SingleMMDiTConfig(
        features=32,
        tdim=32,
        txtdim=32,
        heads=2,
        multiplier=1,
        layers=3,
        patch=2,
        channels=4,
        bias=False,
        theta=1e3,
        kvheads=None,
        txtlayers=1,
        txtheads=2,
        txtkvheads=2,
    )


@pytest.fixture
def tiny_k2_model(tiny_k2_config):
    torch.manual_seed(0)
    model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    model.eval()
    return model
