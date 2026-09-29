"""Krea 2 (K2) base trainer + ``noisy_model_input``/``latents``/``noise`` in ``DiTOutput.extra``.

Use with a ``--loss_fn`` that needs the clean-latent estimate, e.g.::

    --loss_fn wavelet_loss.musubi.WaveletPlusX0Huber \\
    --loss_fn_args alpha=1.0 loss_type='x0_huber' energy_beta=0.1 mottle_metrics=True
"""

import logging

from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser

from boo_musubi_tuner.loss_context.loss_context import LossContextMixin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Krea2LossContextNetworkTrainer(LossContextMixin, Krea2NetworkTrainer):
    pass


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2LossContextNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
