import torch

from boo_musubi_tuner.tdm_distill.tdm_distill import LoraRoleSwitcher
from tests.tdm_distill.conftest import StubLoraNetwork


def test_use_teacher_sets_multiplier_zero_no_weight_copy():
    net = StubLoraNetwork(init_value=5.0)
    switcher = LoraRoleSwitcher(net)
    switcher.init_from(net.state_dict())
    switcher.use_teacher()
    assert net.multiplier == 0.0
    assert torch.equal(net.lora_w.detach(), torch.full((4,), 5.0))  # weights untouched


def test_use_student_loads_student_state_and_enables():
    net = StubLoraNetwork(init_value=1.0)
    switcher = LoraRoleSwitcher(net)
    switcher.init_from({"lora_w": torch.full((4,), 3.0)})
    switcher.use_student()
    assert net.multiplier == 1.0
    assert torch.equal(net.lora_w.detach(), torch.full((4,), 3.0))


def test_use_fake_score_loads_independent_state():
    net = StubLoraNetwork(init_value=1.0)
    switcher = LoraRoleSwitcher(net)
    switcher.init_from({"lora_w": torch.full((4,), 3.0)})
    switcher.use_student()
    net.lora_w.data.fill_(7.0)  # simulate an optimizer step on the student
    switcher.use_fake_score()
    assert torch.equal(net.lora_w.detach(), torch.full((4,), 3.0))  # fake-score still at init


def test_switching_back_to_student_preserves_in_place_update():
    net = StubLoraNetwork(init_value=1.0)
    switcher = LoraRoleSwitcher(net)
    switcher.init_from({"lora_w": torch.full((4,), 3.0)})
    switcher.use_student()
    net.lora_w.data.fill_(9.0)  # simulate an optimizer step on the student
    switcher.use_fake_score()
    switcher.use_student()
    assert torch.equal(net.lora_w.detach(), torch.full((4,), 9.0))  # student update was preserved


def test_student_state_and_fake_score_state_properties():
    net = StubLoraNetwork(init_value=1.0)
    switcher = LoraRoleSwitcher(net)
    switcher.init_from({"lora_w": torch.full((4,), 2.0)})
    assert torch.equal(switcher.student_state["lora_w"], torch.full((4,), 2.0))
    assert torch.equal(switcher.fake_score_state["lora_w"], torch.full((4,), 2.0))
