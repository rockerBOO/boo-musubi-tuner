"""Smoke tests for the Krea 2 Explorative Modeling entry point.

No real training run — just verifies the mixin composition and CLI wiring.
"""

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from boo_musubi_tuner.explorative_modeling.krea2_train_network_xm import Krea2XMNetworkTrainer
from boo_musubi_tuner.explorative_modeling.explorative_modeling import ExplorativeModelingMixin, explorative_modeling_setup_parser
from musubi_tuner.training.parser_common import setup_parser_common


def test_trainer_mixes_in_explorative_modeling_and_krea2():
    trainer = Krea2XMNetworkTrainer()
    assert isinstance(trainer, ExplorativeModelingMixin)
    assert isinstance(trainer, Krea2NetworkTrainer)


def test_mixin_precedes_krea2_in_mro_so_process_batch_override_wins():
    mro = [cls.__name__ for cls in Krea2XMNetworkTrainer.__mro__]
    assert mro.index("ExplorativeModelingMixin") < mro.index("Krea2NetworkTrainer")


def test_parser_wiring_includes_xm_and_krea2_flags():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = explorative_modeling_setup_parser(parser)

    args, _ = parser.parse_known_args(["--explorative_modeling", "--explorative_modeling_k", "6"])

    assert args.explorative_modeling is True
    assert args.explorative_modeling_k == 6
    assert args.fp8_scaled is False  # krea2_setup_parser flag present and defaulted
