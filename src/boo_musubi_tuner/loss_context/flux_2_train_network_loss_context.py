"""FLUX.2 base trainer + ``noisy_model_input``/``latents``/``noise`` in ``DiTOutput.extra``.

Use with a ``--loss_fn`` that needs the clean-latent estimate, e.g.::

    --loss_fn wavelet_loss.musubi.WaveletPlusX0Huber \\
    --loss_fn_args alpha=1.0 loss_type='x0_huber' energy_beta=0.1 mottle_metrics=True
"""

import logging

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common

from boo_musubi_tuner.loss_context.loss_context import LossContextMixin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Flux2LossContextNetworkTrainer(LossContextMixin, Flux2NetworkTrainer):
    pass


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if args.vae_dtype is None:
        args.vae_dtype = "float32"  # make float32 as default for VAE

    trainer = Flux2LossContextNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
