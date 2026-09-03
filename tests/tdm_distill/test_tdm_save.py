import torch

from boo_musubi_tuner.tdm_distill.krea2_train_network_tdm_distill import Krea2TdmDistillNetworkTrainer
from tests.tdm_distill.conftest import FakeAccelerator, StubLoraNetwork

from .test_tdm_arg_validation import make_args


def test_on_post_save_ensures_student_weights_active():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(learning_rate=1e-4, optimizer_type="AdamW")
    net = StubLoraNetwork(init_value=1.0)
    net.load_weights = lambda path: "ok"
    acc = FakeAccelerator()
    trainer.handle_model_specific_args(args)
    trainer.on_train_start(args, acc, net, None, None)

    trainer._role_switcher.use_fake_score()
    net.lora_w.data.fill_(99.0)  # simulate a fake-score-only weight state, must not leak to save

    trainer.on_post_save(args, acc, net, None, "ckpt.safetensors", None, {}, False)
    assert torch.equal(net.lora_w.detach(), trainer._role_switcher.student_state["lora_w"])


def test_on_post_save_noop_without_tdm_distill():
    trainer = Krea2TdmDistillNetworkTrainer()
    args = make_args(tdm_distill=False)
    net = StubLoraNetwork(init_value=1.0)
    acc = FakeAccelerator()
    trainer.on_post_save(args, acc, net, None, "ckpt.safetensors", None, {}, False)
    assert torch.equal(net.lora_w.detach(), torch.ones(4))
