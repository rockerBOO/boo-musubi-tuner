"""Helper for building minimal K2 training batches in self-flow tests."""

import torch


def make_k2_batch(B=2, H=8, W=8, n_txt=3, patch=2, txtlayers=1, txtdim=32, seed=0):
    """Build minimal K2 training inputs: (batch dict, latents, noise).

    latents/noise are (B, C, 1, H, W) — K2 forces single-frame 5D latents.
    krea2_vl_embed is a list of per-sample (seq_len, txtlayers, txtdim) tensors
    (variable seq_len across the batch, matching the real varlen text cache).
    """
    torch.manual_seed(seed)
    channels = 4
    latents = torch.randn(B, channels, 1, H, W)
    noise = torch.randn_like(latents)
    vl_embed = [torch.randn(n_txt, txtlayers, txtdim) for _ in range(B)]
    batch = {"latents": latents, "krea2_vl_embed": vl_embed, "timesteps": None}
    return batch, latents, noise
