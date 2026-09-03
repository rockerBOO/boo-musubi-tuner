"""TDM diversity-distillation training entry point for Krea 2 (K2).

Implements Trajectory Distribution Matching (TDM, arXiv:2503.06674) on the K2 backbone, with a
DINOv3 group-diversity term folded into the student loss. See
docs/superpowers/specs/2026-09-03-krea2-tdm-diversity-distill-design.md for the full design and
docs/tdm-distill.md for usage/flags/limitations.

Internal extension point — no API stability guarantees. Experimental: see docs/tdm-distill.md for
known simplifications vs. the paper.
"""

import argparse
import logging

from accelerate import Accelerator
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser

from boo_musubi_tuner.tdm_distill.tdm_distill import LoraRoleSwitcher

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2TdmDistillNetworkTrainer(Krea2NetworkTrainer):
    def __init__(self) -> None:
        super().__init__()
        self._role_switcher: LoraRoleSwitcher | None = None
        self._fake_score_optimizer = None

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
        if args.tdm_diversity_weight == 0.0:
            logger.warning(
                "--tdm_diversity_weight is 0.0: TDM will train with no diversity term at all "
                "(plain step/guidance distillation). This is a valid ablation setting, not an error."
            )

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
        info = unwrapped_nw.load_weights(args.tdm_turbo_lora_init)
        accelerator.print(f"loaded Turbo LoRA warm-start weights from {args.tdm_turbo_lora_init}: {info}")

        self._role_switcher = LoraRoleSwitcher(unwrapped_nw)
        self._role_switcher.init_from(unwrapped_nw.state_dict())

        if args.fake_score_learning_rate is None:
            args.fake_score_learning_rate = args.learning_rate * 10.0
        if args.fake_score_optimizer_type is None:
            args.fake_score_optimizer_type = args.optimizer_type

        fake_score_args = argparse.Namespace(**vars(args))
        fake_score_args.learning_rate = args.fake_score_learning_rate
        fake_score_args.optimizer_type = args.fake_score_optimizer_type
        _, _, self._fake_score_optimizer, _, _ = self.get_optimizer(fake_score_args, list(unwrapped_nw.parameters()))
        self._fake_score_optimizer = accelerator.prepare(self._fake_score_optimizer)

        logger.info(
            f"TDM distillation enabled: step_counts={args.tdm_step_counts}, "
            f"diversity_weight={args.tdm_diversity_weight}, diversity_group_size={args.tdm_diversity_group_size}, "
            f"fake_score_lr={args.fake_score_learning_rate}"
        )

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
            }
        )
        return metadata


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
