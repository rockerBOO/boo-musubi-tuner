"""TDM diversity-distillation training entry point for Krea 2 (K2).

Implements Trajectory Distribution Matching (TDM, arXiv:2503.06674) on the K2 backbone, with a
DINOv3 group-diversity term folded into the student loss. See docs/tdm-distill.md for usage,
flags, and known limitations.

Internal extension point — no API stability guarantees. Experimental.
"""

import argparse
import gc
import importlib
import itertools
import logging
import os
import sys

import torch
from accelerate import Accelerator
from musubi_tuner.hv_train_network import clean_memory_on_device, read_config_from_file, setup_parser_common
from musubi_tuner.krea2 import krea2_sampling, krea2_utils
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from safetensors.torch import load_file

from boo_musubi_tuner.tdm_distill.tdm_distill import (
    LoraRoleSwitcher,
    cfg_combine,
    diversity_loss_from_embeddings,
    fake_score_denoising_loss,
    min_snr_weight,
    pseudo_huber_c,
    pseudo_huber_loss,
    revised_sample,
    sample_step_count,
    sample_trajectory_interval,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2TdmDistillNetworkTrainer(Krea2NetworkTrainer):
    def __init__(self) -> None:
        super().__init__()
        self._role_switcher: LoraRoleSwitcher | None = None
        self._fake_score_optimizer = None
        self._fake_score_optimizer_train_fn = lambda: None
        self._fake_score_optimizer_eval_fn = lambda: None
        self._dinov3_embedder = None
        # Set when process_batch moved the VAE onto the training device for a grad-enabled
        # decode; consumed by on_post_optimizer_step (see the comment in process_batch).
        self._vae_ref = None
        self._vae_needs_cpu_return = False
        self._vae_frozen = False
        self._teacher_uncond_embed: torch.Tensor | None = None

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        super().handle_model_specific_args(args)
        if isinstance(args.tdm_step_counts, str):
            args.tdm_step_counts = [int(x) for x in args.tdm_step_counts.split(",")]

        if not args.tdm_distill:
            return

        if not args.tdm_turbo_lora_init:
            raise ValueError("--tdm_turbo_lora_init is required when --tdm_distill is set.")
        if not all(k > 0 for k in args.tdm_step_counts):
            raise ValueError(f"--tdm_step_counts ({args.tdm_step_counts}) must all be positive integers.")
        if args.tdm_diversity_group_size < 2:
            raise ValueError(
                f"--tdm_diversity_group_size ({args.tdm_diversity_group_size}) must be >= 2 "
                "(pairwise diversity is undefined for fewer than 2 samples)."
            )
        if args.tdm_guidance_scale is None:
            raise ValueError("--tdm_guidance_scale is required when --tdm_distill is set.")
        if args.tdm_guidance_scale > 1.0 and not args.text_encoder:
            raise ValueError(
                "--text_encoder is required when --tdm_guidance_scale > 1.0. CFG needs the Qwen3-VL "
                "encoder to build the unconditional (empty-prompt) embedding once at train start."
            )
        if args.gradient_accumulation_steps != 1:
            raise ValueError(
                f"--gradient_accumulation_steps ({args.gradient_accumulation_steps}) must be 1 when --tdm_distill is set. "
                "The student and the fake-score critic share the same LoRA parameters across two separately prepared "
                "optimizers, so the critic's mid-accumulation zero_grad() would wipe the student's accumulating grads."
            )
        if args.tdm_diversity_weight > 0.0 and not args.sample_prompts:
            raise ValueError(
                "--sample_prompts is required when --tdm_distill is set with --tdm_diversity_weight > 0. "
                "The base trainer only loads the VAE when sampling is configured, and the diversity term needs "
                "the VAE to decode latents to pixels. Point it at any prompt file (a minimal one is fine); "
                "this is about VAE availability, not about wanting sample images."
            )
        if args.tdm_diversity_weight == 0.0:
            logger.warning(
                "--tdm_diversity_weight is 0.0: TDM will train with no diversity term at all "
                "(plain step/guidance distillation). This is a valid ablation setting, not an error."
            )
        if args.network_dim is not None:
            raise ValueError(
                "--network_dim is not supported with --tdm_distill: the student/fake-score LoRA is built with "
                "each module's rank inferred directly from --tdm_turbo_lora_init's own weights, not a single "
                "uniform rank — the turbo LoRA is not guaranteed to use the same rank in every module."
            )
        if args.network_alpha != 1:
            raise ValueError(
                "--network_alpha is not supported with --tdm_distill: alpha is inferred per-module directly "
                "from --tdm_turbo_lora_init's own weights."
            )
        if args.network_weights is not None:
            raise ValueError(
                "--network_weights is not supported with --tdm_distill: the warm-start weights always come "
                "from --tdm_turbo_lora_init."
            )
        if args.dim_from_weights:
            raise ValueError(
                "--dim_from_weights is not supported with --tdm_distill: dim/alpha inference from "
                "--tdm_turbo_lora_init happens automatically."
            )

    def _build_network(self, args: argparse.Namespace, accelerator: Accelerator, transformer, vae, weight_dtype):
        if not args.tdm_distill:
            return super()._build_network(args, accelerator, transformer, vae, weight_dtype)

        # Mirrors trainer_base.py's own sys.path setup in _build_network, which lets short module
        # paths like `networks.lora_krea2` (relative to the musubi_tuner package dir) resolve. We
        # can't reuse that method here since it always builds a single-uniform-rank network from
        # --network_dim/--network_alpha; TDM's warm-start LoRA is not guaranteed to be uniform-rank
        # across modules (confirmed on a real Turbo LoRA: its projector module is rank 1 while every
        # transformer block is rank 64), so the network must be shaped from the weights file itself.
        sys.path.append(os.path.dirname(os.path.dirname(krea2_utils.__file__)))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        weights_sd = load_file(args.tdm_turbo_lora_init)
        network = network_module.create_arch_network_from_weights(1.0, weights_sd, unet=transformer)

        if hasattr(network_module, "prepare_network"):
            network.prepare_network(args)

        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

        info = network.load_weights(args.tdm_turbo_lora_init)
        accelerator.print(f"loaded Turbo LoRA warm-start weights from {args.tdm_turbo_lora_init}: {info}")

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
            network.enable_gradient_checkpointing()

        return network

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        super().on_train_start(args, accelerator, network, transformer, optimizer)
        if not args.tdm_distill:
            return

        unwrapped_nw = accelerator.unwrap_model(network)
        self._role_switcher = LoraRoleSwitcher(unwrapped_nw)
        self._role_switcher.init_from(unwrapped_nw.state_dict())

        if args.fake_score_learning_rate is None:
            args.fake_score_learning_rate = args.learning_rate * 10.0
        if args.fake_score_optimizer_type is None:
            args.fake_score_optimizer_type = args.optimizer_type

        fake_score_args = argparse.Namespace(**vars(args))
        fake_score_args.learning_rate = args.fake_score_learning_rate
        fake_score_args.optimizer_type = args.fake_score_optimizer_type
        _, _, self._fake_score_optimizer, self._fake_score_optimizer_train_fn, self._fake_score_optimizer_eval_fn = (
            self.get_optimizer(fake_score_args, list(unwrapped_nw.parameters()))
        )
        self._fake_score_optimizer = accelerator.prepare(self._fake_score_optimizer)
        # Mirrors trainer_base.py's own `optimizer_train_fn()` call right before its training loop
        # starts (see the "Set training mode" call site): for plain optimizers get_optimizer()
        # returns a no-op here, but for schedule-free optimizers (e.g. AdamWScheduleFree, which
        # construct in eval mode) this is required before the critic's first .step() call, or it
        # would silently train under eval-mode statistics.
        self._fake_score_optimizer_train_fn()

        self._dinov3_embedder = None

        if args.tdm_diversity_weight > 0.0:
            budget = max(args.tdm_step_counts) * args.tdm_diversity_group_size
            if budget > 16:
                logger.warning(
                    f"TDM diversity term: max(--tdm_step_counts)={max(args.tdm_step_counts)} * "
                    f"--tdm_diversity_group_size={args.tdm_diversity_group_size} = {budget} simultaneous "
                    "sample-forwards with retained activations for the diversity rollout. This is the "
                    "single largest activation consumer in the step and sets peak VRAM; consider lowering "
                    "one of these flags if you hit OOM. (Heuristic threshold, not a hard limit.)"
                )

        if args.tdm_guidance_scale > 1.0:
            encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=torch.bfloat16, device=accelerator.device)
            hiddens, mask = krea2_utils.get_krea2_prompt_embeds(encoder, [""])
            self._teacher_uncond_embed = hiddens[0][mask[0]].to("cpu")
            del encoder
            gc.collect()
            clean_memory_on_device(accelerator.device)

        logger.info(
            f"TDM distillation enabled: step_counts={args.tdm_step_counts}, "
            f"diversity_weight={args.tdm_diversity_weight}, diversity_group_size={args.tdm_diversity_group_size}, "
            f"fake_score_lr={args.fake_score_learning_rate}"
        )

    def on_post_optimizer_step(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        sync_gradients: bool,
        global_step: int,
    ) -> None:
        super().on_post_optimizer_step(args, accelerator, network, transformer, sync_gradients, global_step)
        if not args.tdm_distill:
            return
        # process_batch's diversity block leaves the VAE on the training device because its decode
        # is grad-enabled and the outer loop's backward() runs after process_batch returns. By now
        # backward() and the optimizer step are done and the graph is dead, so it is safe to hand
        # the VAE's VRAM back.
        if self._vae_needs_cpu_return:
            if self._vae_ref is not None:
                self._vae_ref.to("cpu")
            self._vae_ref = None
            self._vae_needs_cpu_return = False

    def on_before_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        super().on_before_sample_images(accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype)
        if not args.tdm_distill:
            return
        # Mirrors trainer_base.py's own optimizer_eval_fn() call immediately before it invokes
        # _do_sample() (which calls this hook, then sample_images(), then on_after_sample_images):
        # a schedule-free fake-score optimizer must be in eval mode for inference, same as the
        # main optimizer.
        self._fake_score_optimizer_eval_fn()

    def on_after_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        super().on_after_sample_images(accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype)
        if not args.tdm_distill:
            return
        # Mirrors trainer_base.py's own optimizer_train_fn() call right after _do_sample()
        # returns, restoring train mode for the next optimizer.step() call.
        self._fake_score_optimizer_train_fn()

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = dict(super().extra_metadata(args))
        if not args.tdm_distill:
            return metadata
        metadata.update(
            {
                "ss_tdm_distill": True,
                "ss_tdm_step_counts": args.tdm_step_counts,
                "ss_tdm_diversity_weight": args.tdm_diversity_weight,
                "ss_tdm_diversity_group_size": args.tdm_diversity_group_size,
                "ss_tdm_guidance_scale": args.tdm_guidance_scale,
                "ss_fake_score_learning_rate": args.fake_score_learning_rate,
                "ss_fake_score_optimizer_type": args.fake_score_optimizer_type,
            }
        )
        return metadata

    def on_post_save(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        ckpt_name: str,
        save_dtype,
        metadata: dict,
        force_sync_upload: bool,
    ) -> None:
        super().on_post_save(args, accelerator, network, transformer, ckpt_name, save_dtype, metadata, force_sync_upload)
        if not args.tdm_distill:
            return
        # This hook fires after the checkpoint is already written, so it cannot protect the save
        # itself. It is state hygiene: leave the student role active for whatever runs next.
        # process_batch already guarantees that, so this is normally a no-op.
        self._role_switcher.use_student()

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict,
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype,
        **kwargs,
    ):
        """Extends Krea2NetworkTrainer.call_dit.

        TDM's process_batch calls this multiple times per training step (student rollout steps,
        fake-score, teacher, updated fake-score) with only one backward in between. ModelOffloader's
        block-swap ring assumes one forward is immediately followed by its own backward -- by the
        second call in a step, blocks the first call swapped out have nothing to swap them back in
        yet, so the next forward hits a CPU-resident block and crashes. Resetting block placement to
        its canonical layout before every forward (the same reset every *_generate_*.py script does
        before a fresh forward) sidesteps the ring's forward/backward coupling entirely.
        """
        if args.tdm_distill and args.blocks_to_swap:
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        return super().call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
        )

    def _student_rollout(
        self,
        args,
        accelerator: Accelerator,
        transformer,
        batch: dict,
        num_steps: int,
        grad_from_step: int,
        device,
        dit_dtype,
        network_dtype,
    ) -> "tuple[list, list[float]]":
        """K-step Euler trajectory from noise, using self.call_dit at each step so the
        currently-active LoRA role (staged externally via LoraRoleSwitcher) is respected.
        Steps before grad_from_step run under no_grad (cheap — most of the trajectory only
        needs to exist to reach the sampled interval, not to be differentiated through)."""
        model = accelerator.unwrap_model(transformer)
        patch = model.config.patch
        vl_embed = batch["krea2_vl_embed"]
        bsize = len(vl_embed)
        # Latents are (B, C, T, H, W) for K2 (single frame, T=1), so H/W are the last two dims.
        lat_h, lat_w = batch["latents"].shape[-2], batch["latents"].shape[-1]

        noise = torch.randn(bsize, model.config.channels, 1, lat_h, lat_w, device=device, dtype=dit_dtype)
        imglen = (lat_h // patch) * (lat_w // patch)
        x1 = (256 // (8 * patch)) ** 2
        x2 = (1280 // (8 * patch)) ** 2
        ts = krea2_sampling.timesteps(imglen, num_steps, x1, x2, y1=0.5, y2=1.15, mu=1.15)

        img = noise
        trajectory = [img]
        for step_idx, (tcurr, tprev) in enumerate(itertools.pairwise(ts)):
            grad_ctx = torch.enable_grad() if step_idx >= grad_from_step else torch.no_grad()
            timesteps_t = torch.full((bsize,), tcurr * 1000.0, device=device, dtype=torch.float32)
            with grad_ctx:
                output = self.call_dit(args, accelerator, transformer, img, batch, noise, img, timesteps_t, network_dtype)
                # call_dit's DiTOutput.pred is the model's direct velocity prediction (not yet
                # compared to target = noise - latents), so integrate it directly on img — matches
                # do_inference's own `img = img + (tprev - tcurr) * v` in krea2_train_network.py.
                img = img + (tprev - tcurr) * output.pred
            trajectory.append(img)
        return trajectory, list(ts)

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict,
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step: int,
    ) -> "tuple[torch.Tensor, dict[str, float]]":
        """TDM step replacing vanilla flow matching. Data-free: `latents`/`noise` args are
        ignored — only batch["krea2_vl_embed"] (the prompt) is used. Returns the TDM student
        loss, plus the DINOv3 group-diversity term added on top when
        --tdm_diversity_weight > 0."""
        if not args.tdm_distill:
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

        device = accelerator.device
        switcher = self._role_switcher

        group_size = args.tdm_diversity_group_size
        # call_dit reads batch["latents"] directly (to derive bsize and the flow-matching
        # target), so single_prompt_batch needs its own group_size-sized latents, not just the
        # text embedding -- repeating a batch-size-1 latents tensor against group_size text
        # embeddings would raise KeyError (missing key) or a shape mismatch (wrong size).
        single_prompt_batch = {
            "krea2_vl_embed": [batch["krea2_vl_embed"][0]] * group_size,
            "latents": batch["latents"][:1].repeat(group_size, *([1] * (batch["latents"].dim() - 1))),
        }

        # Draw from the global RNG (generator=None): a freshly constructed torch.Generator has a
        # fixed default seed, so building one per call would pick the *same* K and interval on
        # every training step. The global RNG still honours the run's torch.manual_seed.
        num_steps = sample_step_count(args.tdm_step_counts, generator=None)
        interval = sample_trajectory_interval(num_steps, generator=None)

        # Steps 1-4 switch the shared LoRA network between student/fake-score/teacher roles.
        # The whole sequence runs inside try/finally so that switcher.use_student() in the
        # finally block always runs -- on normal completion *and* on any exception raised partway
        # through a role switch -- restoring the canonical "student" resting state that the next
        # call to process_batch (and on_post_save, if training stops here) expects. This finally
        # is also what step 5 used to do as a standalone inline call; folding it in here means
        # the invariant is structural rather than incidental.
        try:
            # 1. build the trajectory up to (and including) x_ti under the student role. The whole
            #    rollout runs under no_grad (grad_from_step=num_steps): the grad-carrying copy of
            #    the final incremental step is recomputed in step 6, *after* all role swaps are
            #    done. Building it here instead would leave a live graph over the LoRA weights
            #    across the fake-score/teacher swaps, and LoraRoleSwitcher's load_state_dict bumps
            #    those params' version counters — the outer loop's deferred backward() would then
            #    raise "variable needed for gradient computation has been modified by an inplace
            #    operation".
            switcher.use_student()
            trajectory, ts = self._student_rollout(
                args,
                accelerator,
                transformer,
                batch,
                num_steps,
                grad_from_step=num_steps,
                device=device,
                dit_dtype=dit_dtype,
                network_dtype=network_dtype,
            )
            x_i, x_ti = trajectory[interval], trajectory[interval + 1]
            t_i, t_i_plus_1 = ts[interval], ts[interval + 1]

            # 2. diffuse x_ti with fresh noise to get x_tau, roughly halfway into the interval.
            tau = (t_i + t_i_plus_1) / 2.0
            fresh_noise = torch.randn_like(x_ti.detach())
            x_tau = (1 - tau) * x_ti.detach() + tau * fresh_noise
            timesteps_tau = torch.full((x_ti.shape[0],), tau * 1000.0, device=device, dtype=torch.float32)

            # 3. fake-score update: predict at x_tau, denoise toward x_ti, step its own optimizer.
            switcher.use_fake_score()
            fake_score_pred = self.call_dit(
                args, accelerator, transformer, x_tau, batch, fresh_noise, x_tau, timesteps_tau, network_dtype
            ).pred
            fake_score_target = (fresh_noise - x_ti.detach()).detach()
            omega_tau = min_snr_weight(tau)
            fake_score_loss = fake_score_denoising_loss(fake_score_pred, fake_score_target, omega_tau=omega_tau)
            accelerator.backward(fake_score_loss)
            self._fake_score_optimizer.step()
            self._fake_score_optimizer.zero_grad(set_to_none=True)

            # 4. real score (teacher, frozen) and fake score (just-updated) at x_tau -> revised target.
            switcher.use_teacher()
            with torch.no_grad():
                cond_score = self.call_dit(
                    args, accelerator, transformer, x_tau, batch, fresh_noise, x_tau, timesteps_tau, network_dtype
                ).pred
                if args.tdm_guidance_scale > 1.0:
                    bsize = x_tau.shape[0]
                    uncond_batch = dict(batch)
                    uncond_batch["krea2_vl_embed"] = [self._teacher_uncond_embed] * bsize
                    uncond_score = self.call_dit(
                        args, accelerator, transformer, x_tau, uncond_batch, fresh_noise, x_tau, timesteps_tau, network_dtype
                    ).pred
                    real_score = cfg_combine(cond_score, uncond_score, args.tdm_guidance_scale)
                else:
                    real_score = cond_score
            switcher.use_fake_score()
            with torch.no_grad():
                fake_score_updated = self.call_dit(
                    args, accelerator, transformer, x_tau, batch, fresh_noise, x_tau, timesteps_tau, network_dtype
                ).pred
            # Negative on purpose: krea2_sampling.timesteps runs 1 -> 0, so t_i_plus_1 < t_i. That
            # sign is what reconciles velocity-space predictions with the score-space convention
            # TDM's revised target is written in. Do not "fix" it to abs() or a swapped
            # subtraction. Magnitude: lambda_tau is deliberately the raw (signed) interval width
            # t_i_plus_1 - t_i, not a separately-tunable step size -- it is the one-step Euler
            # discrepancy between the real and fake flows integrated over exactly this interval,
            # matching how x_ti_student itself is advanced in step 6 below.
            lambda_tau = t_i_plus_1 - t_i
            x_revised = revised_sample(x_ti.detach(), real_score, fake_score_updated, lambda_tau).detach()
        finally:
            # 5. student role goes live now, once, and stays live through both the diversity
            #    term's rollout (if enabled) and the student loss forward below.
            #    LoraRoleSwitcher.use_student() calls load_state_dict on the live LoRA weights,
            #    which bumps their version counters -- calling it again after a graph has been
            #    built over those weights would make the outer loop's deferred backward() raise
            #    "variable needed for gradient computation has been modified by an inplace
            #    operation". So nothing below this point may call use_student() (or any other role
            #    switch) again.
            switcher.use_student()

        div_loss = None
        if args.tdm_diversity_weight > 0.0 and vae is not None:
            if self._dinov3_embedder is None:
                from boo_musubi_tuner.tdm_distill.tdm_distill import Dinov3ImageEmbedder

                self._dinov3_embedder = Dinov3ImageEmbedder(device=str(accelerator.device))

            # This whole chain (rollout -> VAE decode -> DINOv3 embed -> diversity loss) must stay
            # differentiable: the diversity term is a real training signal, so gradient has to reach
            # the student LoRA weights. Hence grad_from_step=0 (every rollout step grad-enabled), no
            # torch.no_grad() around the decode, and Dinov3ImageEmbedder.embed_differentiable
            # (which runs without @torch.no_grad() and takes an already-decoded tensor, not
            # PIL/numpy images, so it stays graph-preserving end to end). The VAE's own
            # params are frozen and in no optimizer, but the decode *operation* still has to run
            # with grad so pixels carry a graph back to final_latent.
            #
            # This rollout runs *before* the student loss forward below (not after) so that the
            # last forward pass in this function is the one whose graph accelerator.backward(loss)
            # actually walks. With block swap + gradient checkpointing, backward's recompute needs
            # the block-swap ring in exactly the state its own forward left it in; any later forward
            # pass (this rollout, if it ran after) repositions blocks for its own purposes and
            # desyncs that ring before backward gets to it.
            group_trajectory, _ = self._student_rollout(
                args,
                accelerator,
                transformer,
                single_prompt_batch,
                num_steps,
                grad_from_step=0,
                device=device,
                dit_dtype=dit_dtype,
                network_dtype=network_dtype,
            )
            final_latent = group_trajectory[-1]
            # Defensive/redundant: trainer_base.py already does vae.requires_grad_(False) once at
            # load time, so this re-freeze is not the thing preventing .grad accumulation in the
            # common case. It is kept because this code cannot rely on that having already run for
            # every possible vae object reaching this point, and requires_grad_(False) is cheap
            # and idempotent to repeat. Freeze once, lazily: on_train_start is not handed the vae,
            # so first use here is the earliest hook that has it.
            if not self._vae_frozen and hasattr(vae, "parameters"):
                for p in vae.parameters():
                    p.requires_grad_(False)
                self._vae_frozen = True

            # vae is kept on CPU between uses to save VRAM (see load_vae/on_before_sample_images);
            # move it onto the training device for this decode. Unlike the base trainer's
            # sample-image decode (krea2_train_network.py), we must NOT move it straight back to
            # CPU: this decode is grad-enabled and nn.Module.to() rebinds param.data in place on
            # the very tensors autograd saved for backward. The outer loop's accelerator.backward()
            # runs *after* process_batch returns, so a CPU round-trip here would crash with a
            # device mismatch. Instead the VAE stays resident on the training device for the rest
            # of the step (extra VRAM cost while the diversity term is enabled) and is returned to
            # CPU in on_post_optimizer_step, once backward has consumed the graph.
            vae.to(device)
            pixels = vae.decode_to_pixels(final_latent.to(vae.dtype))  # (group_size, C, H, W) in [0, 1]
            self._vae_ref = vae
            self._vae_needs_cpu_return = True
            pixel_batch = torch.clamp(pixels.float(), 0.0, 1.0)
            embeddings = self._dinov3_embedder.embed_differentiable(pixel_batch)
            div_loss = diversity_loss_from_embeddings(embeddings)

        # 6. student loss: re-run the single Euler step landing on x_ti, now with grad. This must
        #    be the last forward pass in process_batch -- see the note on switcher.use_student()
        #    and the diversity rollout above for why.
        timesteps_i = torch.full((x_i.shape[0],), t_i * 1000.0, device=device, dtype=torch.float32)
        student_pred = self.call_dit(
            args, accelerator, transformer, x_i, batch, trajectory[0], x_i, timesteps_i, network_dtype
        ).pred
        x_ti_student = x_i + (t_i_plus_1 - t_i) * student_pred

        data_dim = x_ti.shape[1:].numel()
        loss = pseudo_huber_loss(x_ti_student, x_revised, c=pseudo_huber_c(data_dim))

        loss_metrics = {
            "loss/tdm": loss.detach().item(),
            "loss/fake_score": fake_score_loss.detach().item(),
            "tdm/k": float(num_steps),
            "tdm/interval": float(interval),
            "tdm/tau": float(tau),
            "tdm/omega_tau": float(omega_tau),
        }

        if div_loss is not None:
            loss = loss + args.tdm_diversity_weight * div_loss
            # div_loss is a graph-carrying tensor; metrics are plain floats by convention.
            loss_metrics["loss/diversity"] = div_loss.detach().item()
            loss_metrics["tdm/diversity_score"] = -div_loss.detach().item()

        return loss, loss_metrics


def tdm_distill_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """TDM diversity-distillation-specific CLI arguments."""
    parser.add_argument(
        "--tdm_distill",
        action="store_true",
        help="Enable TDM (Trajectory Distribution Matching) diversity distillation training.",
    )
    parser.add_argument(
        "--tdm_turbo_lora_init",
        type=str,
        default=None,
        help="Path to the K2 Turbo LoRA used to warm-start both the student and fake-score networks.",
    )
    parser.add_argument(
        "--tdm_step_counts",
        type=str,
        default="1,2,4,8",
        help="Comma-separated student sampling-step counts K, one sampled uniformly per training iteration.",
    )
    parser.add_argument(
        "--tdm_diversity_group_size",
        type=int,
        default=4,
        help="Number of same-prompt, different-seed samples per training step used for the diversity term.",
    )
    parser.add_argument(
        "--tdm_diversity_weight",
        type=float,
        default=0.1,
        help="Constant weight on the DINOv3 group-diversity loss term (not annealed).",
    )
    parser.add_argument(
        "--tdm_guidance_scale",
        type=float,
        default=None,
        help="CFG scale for the teacher's real-score forward: uncond + scale * (cond - uncond). "
        "<= 1.0 disables CFG (single conditional teacher forward, no extra cost). No default -- "
        "every --tdm_distill run must set this explicitly.",
    )
    parser.add_argument(
        "--fake_score_learning_rate",
        type=float,
        default=None,
        help="LR for the fake-score critic's own optimizer. Defaults to 10x --learning_rate (paper ratio).",
    )
    parser.add_argument(
        "--fake_score_optimizer_type",
        type=str,
        default=None,
        help="Optimizer type for the fake-score critic. Defaults to --optimizer_type.",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = tdm_distill_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2TdmDistillNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
