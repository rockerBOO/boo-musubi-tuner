import pytest
import torch

from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import Krea2TdmDistillNetworkTrainer
from boo_musubi_tuner.tdm_distill.tdm_distill import LoraRoleSwitcher
from tests.tdm_distill.conftest import FakeAccelerator, StubLoraNetwork

from .test_tdm_arg_validation import make_args


def test_on_train_start_builds_role_switcher_and_fake_score_optimizer():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_step_counts="1,2,4,8", learning_rate=1e-4, optimizer_type="AdamW")
    net = StubLoraNetwork(init_value=3.0)
    net._weight_registry = {args.tdm_turbo_lora_init: 3.0}

    def load_weights(path):
        net.lora_w.data.fill_(net._weight_registry[path])
        return f"loaded {path}"

    net.load_weights = load_weights
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
    net.load_weights = lambda path: "ok"
    acc = FakeAccelerator()
    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)
    assert args.fake_score_learning_rate == pytest.approx(2e-5)


def test_fake_score_lr_explicit_value_kept():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(learning_rate=2e-6, fake_score_learning_rate=5e-4, optimizer_type="AdamW")
    net = StubLoraNetwork()
    net.load_weights = lambda path: "ok"
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
