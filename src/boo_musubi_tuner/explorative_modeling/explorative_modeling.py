"""Explorative Modeling (XM) — best-of-K training, architecture-generic mixin.

Reference: https://explorative-modeling.github.io/ (Forward XM).

For each training example, sample K noise candidates, score all K, and
backward only the lowest-loss candidate per example. No model edits required:
candidates are built and scored entirely through the existing
``NetworkTrainer`` extension seams (``call_dit``, ``compute_loss``).

Internal extension point — no API stability guarantees. Subclasses live in
this repo; if you fork, expect breakage on updates.
"""

import argparse
import logging

import torch
from accelerate import Accelerator

from musubi_tuner.training.timesteps import get_sigmas
from musubi_tuner.training.trainer_base import DiTOutput

logger = logging.getLogger(__name__)

# Sampling modes that produce continuous `t` in [0, 1] and off-schedule
# `timesteps = t * 1000 + 1` (see `NetworkTrainer.get_noisy_model_input_and_timesteps`
# in trainer_base.py). Mirrors that method's own dispatch condition exactly —
# keep in sync with it. Everything else (i.e. the "sigma" default) samples
# on-schedule discrete timesteps directly and must go through `get_sigmas`.
_CONTINUOUS_T_SAMPLING_MODES = frozenset(
    {
        "uniform",
        "sigmoid",
        "shift",
        "flux_shift",
        "qwen_shift",
        "krea2_shift",
        "ideogram4_shift",
        "logsnr",
        "qinglong_flux",
        "qinglong_qwen",
        "flux2_shift",
    }
)


def _select_winner(loss_stack: torch.Tensor) -> torch.Tensor:
    """Per-example argmin over K candidates.

    ``loss_stack``: ``(K, B)`` — per-candidate, per-example loss. Returns a
    ``(B,)`` long tensor: the winning candidate index for each example.
    """
    return loss_stack.argmin(dim=0)


def _gather_winner(stack: torch.Tensor, winner: torch.Tensor) -> torch.Tensor:
    """Select each example's winning candidate out of a stacked tensor.

    ``stack``: ``(K, B, ...)`` — candidate tensors sharing a per-example
    layout (noise, noisy input, pred, target, ...). ``winner``: ``(B,)``,
    as returned by ``_select_winner``. Returns ``(B, ...)``.
    """
    batch_idx = torch.arange(stack.shape[1], device=stack.device)
    return stack[winner, batch_idx]


class ExplorativeModelingMixin:
    """Mix in ahead of a concrete ``NetworkTrainer`` subclass, e.g.::

        class Krea2XMNetworkTrainer(ExplorativeModelingMixin, Krea2NetworkTrainer):
            pass

    Overrides ``process_batch`` and ``extra_metadata`` only — no architecture
    hooks, no model edits. When ``args.explorative_modeling`` is off, both
    delegate to ``super()`` unchanged.

    Known composability gaps (documented, not fixed here — see the design doc
    for detail):

    - Candidates 2..K are built by reimplementing the base trainer's noising
      math directly (the continuous-t / ``get_sigmas`` branches in
      ``process_batch`` below), not by re-invoking
      ``self.get_noisy_model_input_and_timesteps``. Only candidate 1 goes
      through that (possibly overridden) method. Architectures overriding
      ``get_noisy_model_input_and_timesteps`` — e.g. ``wan_train_network.py``
      (high/low-noise resampling) and ``hidream_o1_train_network.py`` (its own
      sigma construction plus noise-clip) — would get candidate 1 noised
      differently from candidates 2..K if mixed in with XM.
    - Scoring calls ``self.compute_loss(..., reduction="none")``. Architectures
      overriding ``compute_loss`` without a ``reduction`` parameter — e.g.
      ``ideogram4_train_network.py`` and ``hidream_o1_train_network.py`` —
      raise ``TypeError`` immediately when mixed in with XM.
    """

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not args.explorative_modeling:
            return super().process_batch(
                args,
                accelerator,
                transformer,
                network,
                batch,
                latents,
                noise,
                noise_scheduler,
                dit_dtype,
                network_dtype,
                vae,
                global_step,
            )

        k = args.explorative_modeling_k
        device = accelerator.device

        # Candidate 1: build the noisy input the normal way. This also fixes
        # `timesteps` for the whole step — every candidate shares the same
        # per-example timestep, only the noise varies (matches the paper's
        # pseudocode: `t` is sampled once, `z` is resampled per candidate).
        noisy_1, timesteps = self.get_noisy_model_input_and_timesteps(
            args, noise, latents, batch["timesteps"], noise_scheduler, device, dit_dtype
        )

        noises = [noise]
        noisy_inputs = [noisy_1]
        if k > 1:
            if args.timestep_sampling in _CONTINUOUS_T_SAMPLING_MODES:
                # `timesteps` is off-schedule by construction here (`t * 1000 + 1` for
                # continuous `t`). Invert that exact transform to recover `t` directly —
                # this matches the base trainer's own math for this family and avoids
                # routing through `get_sigmas`, which would nearest-neighbor-round
                # off-schedule `timesteps` back onto the discrete schedule (warning every
                # call and introducing a small systematic sigma offset vs. candidate 1).
                # Note: `t` is left in its natural (float32) dtype — no cast to
                # `latents.dtype` — matching how the base trainer keeps `t` before the
                # interpolation; casting down to bf16/fp16 here would quantize `t` and
                # reintroduce a sigma offset between candidate 1 and candidates 2..K.
                t = ((timesteps - 1.0) / 1000.0).view(-1, *([1] * (latents.ndim - 1))).to(device=device)
                for _ in range(k - 1):
                    noise_k = torch.randn_like(latents)
                    noisy_k = (1 - t) * latents + t * noise_k
                    noises.append(noise_k)
                    noisy_inputs.append(noisy_k)
            else:
                # "sigma" (the argparse default): `timesteps` is already on-schedule by
                # construction, so `get_sigmas` is exact here (no rounding, no warning) —
                # this is exactly what the base trainer's own "sigma" branch does.
                sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dit_dtype)
                for _ in range(k - 1):
                    noise_k = torch.randn_like(latents)
                    noisy_k = sigmas * noise_k + (1.0 - sigmas) * latents
                    noises.append(noise_k)
                    noisy_inputs.append(noisy_k)

        memory_efficient = args.explorative_modeling_memory_efficient
        scoring_context = torch.no_grad() if memory_efficient else torch.enable_grad()

        outputs = []
        per_example_losses = []
        with scoring_context:
            for noise_c, noisy_c in zip(noises, noisy_inputs):
                output_c = self.call_dit(args, accelerator, transformer, latents, batch, noise_c, noisy_c, timesteps, network_dtype)
                loss_c, _ = self.compute_loss(
                    args, output_c, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step, reduction="none"
                )
                per_example_losses.append(loss_c)
                if not memory_efficient:
                    outputs.append(output_c)

        loss_stack = torch.stack(per_example_losses, dim=0)  # (K, B)
        winner = _select_winner(loss_stack)
        candidate_std = loss_stack.std(dim=0).mean().item() if k >= 2 else 0.0
        candidate_min = loss_stack.min(dim=0).values.mean().item()
        candidate_max = loss_stack.max(dim=0).values.mean().item()
        candidate_avg = loss_stack.mean(dim=0).mean().item()

        if memory_efficient:
            noise_stack = torch.stack(noises, dim=0)
            noisy_stack = torch.stack(noisy_inputs, dim=0)
            winner_noise = _gather_winner(noise_stack, winner)
            winner_noisy = _gather_winner(noisy_stack, winner)
            winner_output = self.call_dit(
                args, accelerator, transformer, latents, batch, winner_noise, winner_noisy, timesteps, network_dtype
            )
        else:
            pred_stack = torch.stack([o.pred for o in outputs], dim=0)
            target_stack = torch.stack([o.target for o in outputs], dim=0)
            winner_output = DiTOutput(
                pred=_gather_winner(pred_stack, winner),
                target=_gather_winner(target_stack, winner),
            )

        loss, loss_metrics = self.compute_loss(
            args, winner_output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step
        )
        loss_metrics = dict(loss_metrics)
        loss_metrics["xm/k"] = float(k)
        loss_metrics["xm/candidate_loss_std"] = candidate_std
        loss_metrics["xm/candidate_loss_min"] = candidate_min
        loss_metrics["xm/candidate_loss_max"] = candidate_max
        loss_metrics["xm/candidate_loss_avg"] = candidate_avg
        return loss, loss_metrics

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        super().on_train_start(args, accelerator, network, transformer, optimizer)
        if not args.explorative_modeling:
            return
        logger.info(
            "Explorative Modeling (XM) enabled: k=%d, memory_efficient=%s",
            args.explorative_modeling_k,
            args.explorative_modeling_memory_efficient,
        )

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = dict(super().extra_metadata(args))
        if not args.explorative_modeling:
            return metadata
        metadata.update(
            {
                "ss_xm": True,
                "ss_xm_k": args.explorative_modeling_k,
                "ss_xm_memory_efficient": args.explorative_modeling_memory_efficient,
            }
        )
        return metadata


def explorative_modeling_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Explorative Modeling-specific CLI arguments."""
    parser.add_argument(
        "--explorative_modeling",
        action="store_true",
        help="Enable Explorative Modeling (XM) best-of-K training: sample K noise candidates per example and "
        "backward only the lowest-loss candidate. See https://explorative-modeling.github.io/.",
    )
    parser.add_argument(
        "--explorative_modeling_k",
        type=int,
        default=4,
        help="Number of noise candidates to explore per training example (K in Forward XM).",
    )
    parser.add_argument(
        "--explorative_modeling_memory_efficient",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Score K candidates under no_grad and backward only a regenerated forward on the per-example "
        "winner (near-constant memory regardless of K). Use --no-explorative_modeling_memory_efficient to keep "
        "all K forward graphs live and backward directly through the gathered winner (literal pseudocode, "
        "~Kx peak activation memory).",
    )
    return parser
