"""Standalone DINOv3 diversity fine-tuning pass for Krea 2 (K2).

Decoupled from TDM distillation, this extension fine-tunes an existing LoRA checkpoint (Turbo
or TDM-trained) purely to increase same-prompt/different-seed sample diversity via DINOv3
group-diversity loss. See docs/dino_diversity.md for usage, flags, and known limitations.

Internal extension point — no API stability guarantees. Experimental.
"""

import argparse
import importlib
import itertools
import logging
import os
import sys

import torch
from accelerate import Accelerator
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.krea2 import krea2_sampling, krea2_utils
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2DinoDiversityNetworkTrainer(Krea2NetworkTrainer):
    def __init__(self) -> None:
        super().__init__()
        # Placeholder attributes for future seam implementations.
        self._dinov3_embedder = None
        self._vae_ref = None
        self._vae_needs_cpu_return = False
        self._vae_frozen = False

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        super().handle_model_specific_args(args)

        if not args.dino_diversity:
            return

        if not args.dino_diversity_lora_init:
            raise ValueError("--dino_diversity_lora_init is required when --dino_diversity is set.")

        if args.dino_diversity_group_size < 2:
            raise ValueError(
                f"--dino_diversity_group_size ({args.dino_diversity_group_size}) must be >= 2 "
                "(pairwise diversity is undefined for fewer than 2 samples)."
            )

        if args.dino_diversity_step_count < 1:
            raise ValueError(f"--dino_diversity_step_count ({args.dino_diversity_step_count}) must be >= 1.")

        if args.network_dim is not None:
            raise ValueError(
                "--network_dim is rejected when --dino_diversity is set. Rank/alpha come from "
                "--dino_diversity_lora_init's own weights."
            )

        if args.network_alpha != 1:
            raise ValueError(
                "--network_alpha is rejected when --dino_diversity is set. Rank/alpha come from "
                "--dino_diversity_lora_init's own weights."
            )

        if args.network_weights is not None:
            raise ValueError(
                "--network_weights is rejected when --dino_diversity is set. Rank/alpha come from "
                "--dino_diversity_lora_init's own weights."
            )

        if args.dim_from_weights:
            raise ValueError(
                "--dim_from_weights is rejected when --dino_diversity is set. Rank/alpha come from "
                "--dino_diversity_lora_init's own weights."
            )

        if not args.dataset_config:
            raise ValueError(
                "--dataset_config is required (this pass runs through the base trainer's dataloader/"
                "sampling machinery for now, even though real cached latents are never used as training content)."
            )

    def _build_network(self, args: argparse.Namespace, accelerator: Accelerator, transformer, vae, weight_dtype):
        if not args.dino_diversity:
            return super()._build_network(args, accelerator, transformer, vae, weight_dtype)

        # Mirrors trainer_base.py's own sys.path setup in _build_network, which lets short module
        # paths like `networks.lora_krea2` (relative to the musubi_tuner package dir) resolve. We
        # can't reuse that method here since it always builds a single-uniform-rank network from
        # --network_dim/--network_alpha; dino_diversity's warm-start LoRA is not guaranteed to be
        # uniform-rank across modules, so the network must be shaped from the weights file itself.
        sys.path.append(os.path.dirname(os.path.dirname(krea2_utils.__file__)))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        weights_sd = load_file(args.dino_diversity_lora_init)
        network = network_module.create_arch_network_from_weights(1.0, weights_sd, unet=transformer)

        # create_arch_network_from_weights (the "from weights" construction path) never forwards
        # dropout kwargs into LoRANetwork -- true of musubi-tuner's own --dim_from_weights path too
        # (trainer_base.py calls the same function with no net_kwargs), not something specific to
        # this trainer. --network_args's rank_dropout/module_dropout (the standard mechanism
        # elsewhere) are consequently inert here. LoRAModule reads dropout/rank_dropout/
        # module_dropout as plain instance attributes at forward time (see lora.py's
        # LoRAModule.forward), so setting them directly on each already-constructed module after
        # the fact applies them without needing a different construction path.
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value
        rank_dropout = net_kwargs.get("rank_dropout")
        module_dropout = net_kwargs.get("module_dropout")
        if args.network_dropout is not None or rank_dropout is not None or module_dropout is not None:
            for lora_module in network.unet_loras:
                lora_module.dropout = args.network_dropout
                lora_module.rank_dropout = float(rank_dropout) if rank_dropout is not None else None
                lora_module.module_dropout = float(module_dropout) if module_dropout is not None else None

        if hasattr(network_module, "prepare_network"):
            network.prepare_network(args)

        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)

        info = network.load_weights(args.dino_diversity_lora_init)
        accelerator.print(f"loaded DINOv3 diversity LoRA warm-start weights from {args.dino_diversity_lora_init}: {info}")

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
        if not args.dino_diversity:
            return

        self._dinov3_embedder = None

        budget = args.dino_diversity_step_count * args.dino_diversity_group_size
        if budget > 16:
            logger.warning(
                f"DINOv3 diversity term: --dino_diversity_step_count={args.dino_diversity_step_count} * "
                f"--dino_diversity_group_size={args.dino_diversity_group_size} = {budget} simultaneous "
                "sample-forwards with retained activations for the diversity rollout. This is the "
                "single largest activation consumer in the step and sets peak VRAM; consider lowering "
                "one of these flags if you hit OOM. (Heuristic threshold, not a hard limit.)"
            )

        logger.info(
            f"DINOv3 diversity fine-tuning enabled: step_count={args.dino_diversity_step_count}, "
            f"group_size={args.dino_diversity_group_size}"
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
        if not args.dino_diversity:
            return
        # process_batch's diversity rollout leaves the VAE on the training device because its
        # decode is grad-enabled and the outer loop's backward() runs after process_batch returns.
        # By now backward() and the optimizer step are done and the graph is dead, so it is safe to
        # hand the VAE's VRAM back.
        if self._vae_needs_cpu_return:
            if self._vae_ref is not None:
                self._vae_ref.to("cpu")
            self._vae_ref = None
            self._vae_needs_cpu_return = False

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

        _rollout calls this once per Euler step within a single training step (one call per group
        member's rollout step), with only one backward at the very end. ModelOffloader's block-swap
        ring assumes one forward is immediately followed by its own backward -- by the second call,
        blocks the first call swapped out have nothing to swap them back in yet, so the next forward
        hits a CPU-resident block and crashes. Resetting block placement to its canonical layout
        before every forward (the same reset every *_generate_*.py script does before a fresh
        forward) sidesteps the ring's forward/backward coupling entirely.
        """
        if args.dino_diversity and args.blocks_to_swap:
            accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        return super().call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
        )

    def _rollout(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        batch: dict,
        num_steps: int,
        device,
        dit_dtype,
        network_dtype,
        noise: "torch.Tensor | None" = None,
    ) -> "tuple[list, list[float]]":
        """K-step Euler trajectory from noise, using self.call_dit at each step.

        Structurally mirrors the sibling tdm_distill extension's _student_rollout, but always
        fully grad-enabled: dino_diversity's entire training signal is the group-diversity loss
        over the rendered images, so every step (including the very first) must stay in the
        autograd graph. There is no grad_from_step split like TDM's -- TDM only differentiates
        through the last sampled interval because the rest of its trajectory only needs to exist to
        reach that interval, not to be trained through; here the whole rollout is what gets scored.

        `noise`, if given, is used as the starting point instead of drawing a fresh one -- lets a
        caller reproduce an earlier rollout's exact trajectory rather than sampling a different
        noise draw the second time.
        """
        model = accelerator.unwrap_model(transformer)
        patch = model.config.patch
        vl_embed = batch["krea2_vl_embed"]
        bsize = len(vl_embed)
        # Latents are (B, C, T, H, W) for K2 (single frame, T=1), so H/W are the last two dims.
        lat_h, lat_w = batch["latents"].shape[-2], batch["latents"].shape[-1]

        if noise is None:
            noise = torch.randn(bsize, model.config.channels, 1, lat_h, lat_w, device=device, dtype=dit_dtype)
        imglen = (lat_h // patch) * (lat_w // patch)
        x1 = (256 // (8 * patch)) ** 2
        x2 = (1280 // (8 * patch)) ** 2
        ts = krea2_sampling.timesteps(imglen, num_steps, x1, x2, y1=0.5, y2=1.15, mu=1.15)

        img = noise
        trajectory = [img]
        for tcurr, tprev in itertools.pairwise(ts):
            timesteps_t = torch.full((bsize,), tcurr * 1000.0, device=device, dtype=torch.float32)
            with torch.enable_grad():
                output = self.call_dit(args, accelerator, transformer, img, batch, noise, img, timesteps_t, network_dtype)
                # call_dit's DiTOutput.pred is the model's direct velocity prediction (not yet
                # compared to a target), so integrate it directly on img -- matches do_inference's
                # own `img = img + (tprev - tcurr) * v` in krea2_train_network.py.
                img = img + (tprev - tcurr) * output.pred
            trajectory.append(img)
        return trajectory, list(ts)


def dino_diversity_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Standalone DINOv3 diversity fine-tuning-specific CLI arguments."""
    parser.add_argument(
        "--dino_diversity",
        action="store_true",
        help="Enable standalone DINOv3 diversity fine-tuning pass on an existing LoRA checkpoint.",
    )
    parser.add_argument(
        "--dino_diversity_lora_init",
        type=str,
        default=None,
        help="Path to the LoRA checkpoint to warm-start from (Turbo LoRA or TDM-trained checkpoint).",
    )
    parser.add_argument(
        "--dino_diversity_group_size",
        type=int,
        default=4,
        help="Number of same-prompt, different-seed samples per training step.",
    )
    parser.add_argument(
        "--dino_diversity_step_count",
        type=int,
        default=8,
        help="Euler rollout steps per sample (full inference-length rollout).",
    )
    parser.add_argument(
        "--dino_diversity_memory_efficient",
        action="store_true",
        help="Two-pass per-sample gradient accumulation to bound peak VRAM. Slower; use when VRAM-constrained.",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = dino_diversity_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2DinoDiversityNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
