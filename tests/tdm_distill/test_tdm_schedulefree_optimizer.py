"""Tests for schedule-free optimizer support on the fake-score critic (finding #1 of the final
whole-branch review): --fake_score_optimizer_type must be trained/eval'd the same way
trainer_base.py trains/evals its own main optimizer around image sampling, or a schedule-free
optimizer (which constructs in eval mode) would silently step the critic under eval-mode stats.
"""

from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import Krea2TdmDistillNetworkTrainer
from tests.tdm_distill.conftest import FakeAccelerator, FakeScheduleFreeOptimizer, StubLoraNetwork

from .test_tdm_arg_validation import make_args


def _trainer_with_schedulefree_fake_score(**arg_overrides):
    trainer = Krea2TdmDistillNetworkTrainer()
    arg_overrides.setdefault("fake_score_optimizer_type", "tests.tdm_distill.conftest.FakeScheduleFreeOptimizer")
    args = make_args(optimizer_type="AdamW", **arg_overrides)
    net = StubLoraNetwork(init_value=3.0)
    net.load_weights = lambda path: "ok"
    acc = FakeAccelerator()
    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)
    return trainer, args, acc, net


def test_fake_score_optimizer_train_fn_called_once_on_construction():
    trainer, _, _, _ = _trainer_with_schedulefree_fake_score()
    assert isinstance(trainer._fake_score_optimizer, FakeScheduleFreeOptimizer)
    assert trainer._fake_score_optimizer.mode_log == ["train"]


def test_sample_images_hooks_toggle_fake_score_optimizer_eval_then_train():
    trainer, args, acc, net = _trainer_with_schedulefree_fake_score()
    fake_score_opt = trainer._fake_score_optimizer
    assert fake_score_opt.mode_log == ["train"]

    trainer.on_before_sample_images(
        acc, args, epoch=0, steps=0, vae=None, transformer=None, network=net, sample_parameters=None, dit_dtype=None
    )
    assert fake_score_opt.mode_log == ["train", "eval"]

    trainer.on_after_sample_images(
        acc, args, epoch=0, steps=0, vae=None, transformer=None, network=net, sample_parameters=None, dit_dtype=None
    )
    assert fake_score_opt.mode_log == ["train", "eval", "train"]


def test_sample_images_hooks_noop_without_tdm_distill():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False)
    acc = FakeAccelerator()
    # No optimizer was ever built (on_train_start returns immediately for tdm_distill=False), so
    # the default no-op train/eval fns must not raise and must not touch anything.
    trainer.on_before_sample_images(
        acc, args, epoch=0, steps=0, vae=None, transformer=None, network=None, sample_parameters=None, dit_dtype=None
    )
    trainer.on_after_sample_images(
        acc, args, epoch=0, steps=0, vae=None, transformer=None, network=None, sample_parameters=None, dit_dtype=None
    )
    assert trainer._fake_score_optimizer is None


def test_plain_optimizer_fake_score_train_eval_fns_are_noop():
    """A plain (non-schedule-free) optimizer type has no .train()/.eval(); get_optimizer's
    fallback no-op lambdas must be captured and called without error."""
    trainer, args, acc, net = _trainer_with_schedulefree_fake_score(fake_score_optimizer_type="AdamW")
    # AdamW has no .train()/.eval(), so get_optimizer's fallback lambdas were captured.
    assert not hasattr(trainer._fake_score_optimizer, "mode_log")
    trainer.on_before_sample_images(
        acc, args, epoch=0, steps=0, vae=None, transformer=None, network=net, sample_parameters=None, dit_dtype=None
    )
    trainer.on_after_sample_images(
        acc, args, epoch=0, steps=0, vae=None, transformer=None, network=net, sample_parameters=None, dit_dtype=None
    )
