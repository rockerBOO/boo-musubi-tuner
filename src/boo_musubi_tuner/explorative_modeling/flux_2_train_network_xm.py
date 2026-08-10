"""Explorative Modeling (XM) training entry point for FLUX.2.

Best-of-K training: mixes ``ExplorativeModelingMixin`` into
``Flux2NetworkTrainer``. XM needs no architecture-specific code — no model
edits, no feature hooks, no teacher/student state — so this file is only
wiring: mixin composition + arg parser + main.

FLUX.2's trainer only overrides ``call_dit``, not ``compute_loss`` or
``get_noisy_model_input_and_timesteps``, so it composes with XM without
hitting either of the composability gaps documented on
``ExplorativeModelingMixin``.

Reference: https://explorative-modeling.github.io/
"""

import logging

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from boo_musubi_tuner.explorative_modeling.explorative_modeling import ExplorativeModelingMixin, explorative_modeling_setup_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Flux2XMNetworkTrainer(ExplorativeModelingMixin, Flux2NetworkTrainer):
    pass


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = explorative_modeling_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if args.vae_dtype is None:
        args.vae_dtype = "float32"  # make float32 as default for VAE

    trainer = Flux2XMNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
