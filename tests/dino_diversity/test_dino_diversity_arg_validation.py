"""Tests for handle_model_specific_args validation in Krea2DinoDiversityNetworkTrainer."""

import pytest
from musubi_tuner.hv_train_network import setup_parser_common
from musubi_tuner.krea2_train_network import krea2_setup_parser

from boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity import (
    Krea2DinoDiversityNetworkTrainer,
    dino_diversity_setup_parser,
)


def make_args(**overrides):
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = dino_diversity_setup_parser(parser)
    args = parser.parse_args([])
    args.dino_diversity = True
    args.dino_diversity_lora_init = "/mnt/900/lora/krea2/krea2_turbo_lora_rank_64_bf16.safetensors"
    # Required by base trainer.
    args.dataset_config = "dummy_config.json"
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_missing_lora_init_raises():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity_lora_init=None)
    with pytest.raises(ValueError, match="dino_diversity_lora_init"):
        trainer.handle_model_specific_args(args)


def test_missing_lora_init_ok_when_dino_diversity_off():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity=False, dino_diversity_lora_init=None)
    trainer.handle_model_specific_args(args)


def test_group_size_below_two_raises():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity_group_size=1)
    with pytest.raises(ValueError, match="dino_diversity_group_size"):
        trainer.handle_model_specific_args(args)


def test_group_size_at_least_two_ok():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity_group_size=2, network_alpha=1)
    trainer.handle_model_specific_args(args)


def test_step_count_zero_raises():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity_step_count=0)
    with pytest.raises(ValueError, match="dino_diversity_step_count"):
        trainer.handle_model_specific_args(args)


def test_step_count_positive_ok():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity_step_count=1, network_alpha=1)
    trainer.handle_model_specific_args(args)


def test_network_dim_rejected():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(network_dim=64)
    with pytest.raises(ValueError, match="network_dim"):
        trainer.handle_model_specific_args(args)


def test_network_alpha_rejected():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(network_alpha=64.0)
    with pytest.raises(ValueError, match="network_alpha"):
        trainer.handle_model_specific_args(args)


def test_network_weights_rejected():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(network_weights="/path/to/other_lora.safetensors", network_alpha=1)
    with pytest.raises(ValueError, match="network_weights"):
        trainer.handle_model_specific_args(args)


def test_dim_from_weights_rejected():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dim_from_weights=True, network_alpha=1)
    with pytest.raises(ValueError, match="dim_from_weights"):
        trainer.handle_model_specific_args(args)


def test_missing_dataset_config_raises():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dataset_config=None, network_alpha=1)
    with pytest.raises(ValueError, match="dataset_config"):
        trainer.handle_model_specific_args(args)


def test_all_constraints_ok_when_dino_diversity_off():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(
        dino_diversity=False,
        dino_diversity_lora_init=None,
        dino_diversity_group_size=1,
        dino_diversity_step_count=0,
        network_dim=64,
        network_alpha=64.0,
        network_weights="/path/to/other_lora.safetensors",
        dim_from_weights=True,
        dataset_config=None,
    )
    trainer.handle_model_specific_args(args)
