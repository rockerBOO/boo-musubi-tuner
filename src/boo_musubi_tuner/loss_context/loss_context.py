"""Stash flow-matching inputs in ``DiTOutput.extra`` for pluggable ``--loss_fn`` callables.

musubi-tuner's base ``compute_loss`` hands a ``--loss_fn`` callable a ``LossContext`` whose
``output`` only carries ``pred``/``target`` (velocity space). Losses that need the clean-latent
estimate ``x0_hat = noisy_model_input - sigma * pred`` (e.g. wavelet-loss's ``rectified_flow``,
``energy_beta`` and ``mottle_metrics``) read ``output.extra["noisy_model_input"]``, which the base
trainers don't provide. This mixin wraps ``call_dit`` to stash it -- and leaves ``compute_loss``
alone, so ``--loss_fn`` / ``--loss_fn_args`` keep working unchanged.

Stashed keys (the exact tensors ``call_dit`` received; references, no copies):

- ``noisy_model_input``: ``(1 - sigma) * latents + sigma * noise``
- ``latents``: clean latents
- ``noise``: the sampled noise

There is no feature flag: stashing references costs nothing, and losses that don't read
``extra`` are unaffected, so this trainer is a plain base trainer for any other ``--loss_fn``.

Internal extension point -- no API stability guarantees.
"""

import argparse

import torch
from accelerate import Accelerator
from musubi_tuner.training.trainer_base import DiTOutput

STASH_KEYS = ("noisy_model_input", "latents", "noise")


class LossContextMixin:
    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        output = super().call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
        )
        output.extra.update(noisy_model_input=noisy_model_input, latents=latents, noise=noise)
        return output
