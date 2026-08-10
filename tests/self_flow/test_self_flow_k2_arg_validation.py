"""Tests for handle_model_specific_args validation in Krea2SelfFlowNetworkTrainer."""

import pytest
from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner.krea2_train_network import krea2_setup_parser

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import (
    Krea2SelfFlowNetworkTrainer,
    self_flow_setup_parser,
)


def make_args(**overrides):
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = self_flow_setup_parser(parser)
    args = parser.parse_args([])
    args.self_flow = True
    args.gradient_checkpointing = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_num_timestep_buckets_raises_with_self_flow():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(num_timestep_buckets=4)
    with pytest.raises(ValueError, match="num_timestep_buckets"):
        trainer.handle_model_specific_args(args)


def test_num_timestep_buckets_none_does_not_raise():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(num_timestep_buckets=None)
    trainer.handle_model_specific_args(args)


def test_num_timestep_buckets_without_self_flow_does_not_raise():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False, num_timestep_buckets=4)
    trainer.handle_model_specific_args(args)


def test_student_ge_teacher_raises():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(student_feature_layer=5, teacher_feature_layer=3)
    with pytest.raises(ValueError, match="student_feature_layer"):
        trainer.handle_model_specific_args(args)


def test_mask_ratio_gt_half_raises():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(mask_ratio=0.6)
    with pytest.raises(ValueError, match="mask_ratio"):
        trainer.handle_model_specific_args(args)


def test_negative_mask_ratio_raises():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(mask_ratio=-0.1)
    with pytest.raises(ValueError, match="mask_ratio"):
        trainer.handle_model_specific_args(args)
