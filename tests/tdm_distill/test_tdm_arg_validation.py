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
    # Required by default since the default diversity weight is > 0 (the VAE is only loaded when
    # sampling is configured).
    args.sample_prompts = "prompts.txt"
    args.tdm_guidance_scale = 1.0
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


def test_missing_sample_prompts_raises_when_diversity_enabled():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(sample_prompts=None)
    with pytest.raises(ValueError, match="sample_prompts"):
        trainer.handle_model_specific_args(args)


def test_missing_sample_prompts_ok_when_diversity_disabled():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(sample_prompts=None, tdm_diversity_weight=0.0)
    trainer.handle_model_specific_args(args)


def test_gradient_accumulation_above_one_raises():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(gradient_accumulation_steps=2)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        trainer.handle_model_specific_args(args)


def test_gradient_accumulation_above_one_ok_when_tdm_distill_off():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False, gradient_accumulation_steps=2)
    trainer.handle_model_specific_args(args)


def test_missing_guidance_scale_raises():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_guidance_scale=None)
    with pytest.raises(ValueError, match="tdm_guidance_scale"):
        trainer.handle_model_specific_args(args)


def test_missing_guidance_scale_ok_when_tdm_distill_off():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False, tdm_guidance_scale=None)
    trainer.handle_model_specific_args(args)


def test_guidance_scale_above_one_requires_text_encoder():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_guidance_scale=3.5, text_encoder=None)
    with pytest.raises(ValueError, match="text_encoder"):
        trainer.handle_model_specific_args(args)


def test_guidance_scale_at_or_below_one_ok_without_text_encoder():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_guidance_scale=1.0, text_encoder=None)
    trainer.handle_model_specific_args(args)


def test_guidance_scale_above_one_ok_with_text_encoder():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_guidance_scale=3.5, text_encoder="/path/to/qwen3_vl.safetensors")
    trainer.handle_model_specific_args(args)


def test_guidance_scale_above_one_requires_sample_prompts():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(
        tdm_guidance_scale=3.5,
        text_encoder="/path/to/qwen3_vl.safetensors",
        sample_prompts=None,
        tdm_diversity_weight=0.0,
    )
    with pytest.raises(ValueError, match="sample_prompts"):
        trainer.handle_model_specific_args(args)


def test_guidance_scale_at_or_below_one_ok_without_sample_prompts():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_guidance_scale=1.0, sample_prompts=None, tdm_diversity_weight=0.0)
    trainer.handle_model_specific_args(args)


def test_network_dim_rejected():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(network_dim=64)
    with pytest.raises(ValueError, match="network_dim"):
        trainer.handle_model_specific_args(args)


def test_network_alpha_rejected():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(network_alpha=64.0)
    with pytest.raises(ValueError, match="network_alpha"):
        trainer.handle_model_specific_args(args)


def test_network_weights_rejected():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(network_weights="/path/to/other_lora.safetensors")
    with pytest.raises(ValueError, match="network_weights"):
        trainer.handle_model_specific_args(args)


def test_dim_from_weights_rejected():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(dim_from_weights=True)
    with pytest.raises(ValueError, match="dim_from_weights"):
        trainer.handle_model_specific_args(args)


def test_network_dim_default_ok():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(network_dim=None, network_alpha=1)
    trainer.handle_model_specific_args(args)


def test_diversity_step_count_defaults_to_one():
    args = make_args()
    assert args.tdm_diversity_step_count == 1


def test_diversity_step_count_zero_raises():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_diversity_step_count=0)
    with pytest.raises(ValueError, match="tdm_diversity_step_count"):
        trainer.handle_model_specific_args(args)


def test_diversity_step_count_exceeding_max_k_raises():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_step_counts="1,2,4", tdm_diversity_step_count=5)
    with pytest.raises(ValueError, match="tdm_diversity_step_count"):
        trainer.handle_model_specific_args(args)


def test_diversity_step_count_at_max_k_ok():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_step_counts="1,2,4", tdm_diversity_step_count=4)
    trainer.handle_model_specific_args(args)


def test_diversity_step_count_out_of_range_ok_when_tdm_distill_off():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False, tdm_diversity_step_count=0)
    trainer.handle_model_specific_args(args)
