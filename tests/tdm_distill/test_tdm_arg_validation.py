"""Tests for handle_model_specific_args validation in Krea2TdmDistillNetworkTrainer."""

import pytest
from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner.krea2_train_network import krea2_setup_parser

from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import (
    Krea2TdmDistillNetworkTrainer,
    tdm_distill_setup_parser,
)


def make_args(**overrides):
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = tdm_distill_setup_parser(parser)
    args = parser.parse_args([])
    args.tdm_distill = True
    args.tdm_turbo_lora_init = "/mnt/900/lora/krea2/krea2_turbo_lora_rank_64_bf16.safetensors"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_step_counts_parsed_as_int_list():
    args = make_args(tdm_step_counts="1,2,4,8")
    trainer = Krea2TdmDistillNetworkTrainer()
    trainer.handle_model_specific_args(args)
    assert args.tdm_step_counts == [1, 2, 4, 8]


def test_missing_turbo_lora_init_raises():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_turbo_lora_init=None)
    with pytest.raises(ValueError, match="tdm_turbo_lora_init"):
        trainer.handle_model_specific_args(args)


def test_missing_turbo_lora_init_ok_when_tdm_distill_off():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False, tdm_turbo_lora_init=None)
    trainer.handle_model_specific_args(args)


def test_non_positive_step_count_raises():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_step_counts="1,0,4")
    with pytest.raises(ValueError, match="tdm_step_counts"):
        trainer.handle_model_specific_args(args)


def test_diversity_group_size_below_two_raises():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_diversity_group_size=1)
    with pytest.raises(ValueError, match="tdm_diversity_group_size"):
        trainer.handle_model_specific_args(args)


def test_zero_diversity_weight_warns_not_raises(caplog):
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_diversity_weight=0.0)
    with caplog.at_level("WARNING"):
        trainer.handle_model_specific_args(args)
    assert any("tdm_diversity_weight" in rec.message for rec in caplog.records)
