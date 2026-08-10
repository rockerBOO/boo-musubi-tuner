from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner.krea2_train_network import krea2_setup_parser

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import (
    Krea2SelfFlowNetworkTrainer,
    self_flow_setup_parser,
)


def test_trainer_instantiates():
    trainer = Krea2SelfFlowNetworkTrainer()
    assert trainer.architecture is not None


def test_parser_builds():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = self_flow_setup_parser(parser)
    args = parser.parse_args([])
    assert args is not None
