import pytest
import torch

import boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill as tdm_module
from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import Krea2TdmDistillNetworkTrainer
from boo_musubi_tuner.tdm_distill.tdm_distill import LoraRoleSwitcher
from tests.tdm_distill.conftest import FakeAccelerator, StubLoraNetwork

from .test_tdm_arg_validation import make_args


def test_on_train_start_builds_role_switcher_and_fake_score_optimizer():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_step_counts="1,2,4,8", learning_rate=1e-4, optimizer_type="AdamW")
    # Warm-start loading now happens in _build_network (not on_train_start), so the network
    # arrives here already carrying its warm-started weights.
    net = StubLoraNetwork(init_value=3.0)
    acc = FakeAccelerator()

    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)

    assert isinstance(trainer._role_switcher, LoraRoleSwitcher)
    assert torch.equal(trainer._role_switcher.student_state["lora_w"], torch.full((4,), 3.0))
    assert torch.equal(trainer._role_switcher.fake_score_state["lora_w"], torch.full((4,), 3.0))
    assert trainer._fake_score_optimizer is not None


def test_on_train_start_noop_without_tdm_distill():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False)
    net = StubLoraNetwork()
    acc = FakeAccelerator()
    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)
    assert trainer._role_switcher is None
    assert trainer._fake_score_optimizer is None


def test_fake_score_lr_defaults_to_10x_learning_rate():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(learning_rate=2e-6, fake_score_learning_rate=None, optimizer_type="AdamW")
    net = StubLoraNetwork()
    acc = FakeAccelerator()
    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)
    assert args.fake_score_learning_rate == pytest.approx(2e-5)


def test_fake_score_lr_explicit_value_kept():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(learning_rate=2e-6, fake_score_learning_rate=5e-4, optimizer_type="AdamW")
    net = StubLoraNetwork()
    acc = FakeAccelerator()
    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)
    assert args.fake_score_learning_rate == 5e-4


def test_extra_metadata_reports_tdm_keys():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_step_counts=[1, 2, 4, 8], tdm_diversity_weight=0.1, tdm_diversity_group_size=4)
    metadata = trainer.extra_metadata(args)
    assert metadata["ss_tdm_distill"] is True
    assert metadata["ss_tdm_step_counts"] == [1, 2, 4, 8]


def test_extra_metadata_empty_without_tdm_distill():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False)
    assert trainer.extra_metadata(args) == {}


class _FakeEncoder:
    pass


def test_process_sample_prompts_caches_uncond_embed_when_cfg_enabled(monkeypatch):
    calls = {"prompts_seen": []}

    def fake_load(path, dtype, device):
        calls["load_args"] = (path, dtype, device)
        return _FakeEncoder()

    def fake_get_embeds(encoder, prompts):
        calls["prompts_seen"].append(prompts)
        # (B=1, seq=3, L=1, D=2), mask marks first 2 tokens valid
        hiddens = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]]])
        mask = torch.tensor([[True, True, False]])
        return hiddens, mask

    monkeypatch.setattr(tdm_module.krea2_utils, "load_krea2_text_encoder", fake_load)
    monkeypatch.setattr(tdm_module.krea2_utils, "get_krea2_prompt_embeds", fake_get_embeds)
    monkeypatch.setattr(tdm_module, "load_prompts", lambda path: [{"prompt": "a cat"}])

    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(
        tdm_guidance_scale=3.5,
        text_encoder="/path/to/qwen3_vl.safetensors",
        optimizer_type="AdamW",
        sample_prompts="prompts.txt",
    )
    acc = FakeAccelerator()

    trainer.handle_model_specific_args(args)
    sample_parameters = trainer.process_sample_prompts(args, acc, args.sample_prompts)

    assert calls["load_args"] == ("/path/to/qwen3_vl.safetensors", torch.bfloat16, acc.device)
    assert calls["prompts_seen"] == [["a cat"], [""]]
    # hiddens[0][mask[0]] gathers the first 2 (valid) of 3 token rows -> shape (2, L=1, D=2)
    expected_embed = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    assert torch.equal(trainer._teacher_uncond_embed, expected_embed)
    assert torch.equal(sample_parameters[0]["krea2_vl_embed"], expected_embed)


def test_process_sample_prompts_delegates_to_super_when_cfg_disabled(monkeypatch):
    calls = []

    def fake_super_process_sample_prompts(self, args, accelerator, sample_prompts):
        calls.append(sample_prompts)
        return "vanilla-sample-parameters"

    monkeypatch.setattr(tdm_module.Krea2NetworkTrainer, "process_sample_prompts", fake_super_process_sample_prompts)

    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_guidance_scale=1.0, optimizer_type="AdamW")
    acc = FakeAccelerator()

    trainer.handle_model_specific_args(args)
    result = trainer.process_sample_prompts(args, acc, "prompts.txt")

    assert result == "vanilla-sample-parameters"
    assert calls == ["prompts.txt"]
    assert trainer._teacher_uncond_embed is None
