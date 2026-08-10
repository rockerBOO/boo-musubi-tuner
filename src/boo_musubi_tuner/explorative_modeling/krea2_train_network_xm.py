"""Explorative Modeling (XM) training entry point for Krea 2 (K2).

Best-of-K training: mixes ``ExplorativeModelingMixin`` into
``Krea2NetworkTrainer``. XM needs no architecture-specific code — no model
edits, no feature hooks, no teacher/student state — so unlike the Self-Flow
skeleton (``flux_2_train_network_self_flow.py``), this file is only wiring
(mixin composition + arg parser + main), not a stub.

Reference: https://explorative-modeling.github.io/
"""

import logging

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from boo_musubi_tuner.explorative_modeling.explorative_modeling import ExplorativeModelingMixin, explorative_modeling_setup_parser

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2XMNetworkTrainer(ExplorativeModelingMixin, Krea2NetworkTrainer):
    pass


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = explorative_modeling_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2XMNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
