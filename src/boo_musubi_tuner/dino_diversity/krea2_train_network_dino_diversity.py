"""Standalone DINOv3 diversity fine-tuning pass for Krea 2 (K2).

Decoupled from TDM distillation, this extension fine-tunes an existing LoRA checkpoint (Turbo
or TDM-trained) purely to increase same-prompt/different-seed sample diversity via DINOv3
group-diversity loss. See docs/dino_diversity.md for usage, flags, and known limitations.

Internal extension point — no API stability guarantees. Experimental.
"""

import argparse
import logging

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2DinoDiversityNetworkTrainer(Krea2NetworkTrainer):
    def __init__(self) -> None:
        super().__init__()
        # Placeholder attributes for future seam implementations.
        self._dinov3_embedder = None
        self._vae_ref = None
        self._vae_needs_cpu_return = False

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
