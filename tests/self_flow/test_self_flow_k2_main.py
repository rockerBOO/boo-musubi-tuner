"""Smoke test: the CLI entry point parses args and constructs the trainer
without touching disk or GPU (train() itself is not invoked)."""

from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner.krea2_train_network import krea2_setup_parser

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import Krea2SelfFlowNetworkTrainer, self_flow_setup_parser


def test_self_flow_args_present_in_parser():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = self_flow_setup_parser(parser)
    args = parser.parse_args(["--self_flow", "--mask_ratio", "0.3"])
    assert args.self_flow is True
    assert args.mask_ratio == 0.3
    assert args.ema_decay == 0.999  # default


def test_trainer_class_has_all_overrides():
    trainer = Krea2SelfFlowNetworkTrainer()
    for method in (
        "handle_model_specific_args",
        "extra_trainable_params",
        "on_transformer_loaded",
        "on_train_start",
        "call_dit",
        "process_batch",
        "on_post_optimizer_step",
        "on_post_save",
        "extra_metadata",
        "extra_step_logs",
        "on_before_sample_images",
        "on_after_sample_images",
    ):
        assert hasattr(trainer, method), f"missing override: {method}"
