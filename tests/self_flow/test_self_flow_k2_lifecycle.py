import torch

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import Krea2SelfFlowNetworkTrainer

from .test_self_flow_k2_arg_validation import make_args

_LOGGER_NAME = "boo_musubi_tuner.self_flow.krea2_train_network_self_flow"


class FakeAccelerator:
    device = torch.device("cpu")

    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def autocast(self):
        return self._NullCtx()

    def unwrap_model(self, m):
        return m

    def prepare(self, m):
        return m

    def print(self, *args, **kwargs):
        pass


class StubNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_w = torch.nn.Parameter(torch.ones(4))
        self._weight_registry: dict[str, float] = {}

    def load_weights(self, path: str):
        val = self._weight_registry.get(path, 0.0)
        self.lora_w.data.fill_(val)
        return f"loaded from {path}"


def test_on_transformer_loaded_installs_hooks_and_defaults_layers(tiny_k2_model):
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args()
    trainer.handle_model_specific_args(args)
    trainer.on_transformer_loaded(args, FakeAccelerator(), tiny_k2_model)
    num_blocks = len(tiny_k2_model.blocks)
    assert args.student_feature_layer == max(0, int(num_blocks * 0.3))
    assert args.teacher_feature_layer == min(num_blocks - 1, int(num_blocks * 0.7))
    assert trainer._modulation_controller is not None
    assert trainer._feature_extractor is not None


def test_on_transformer_loaded_noop_without_self_flow(tiny_k2_model):
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    trainer.on_transformer_loaded(args, FakeAccelerator(), tiny_k2_model)
    assert trainer._modulation_controller is None
    assert trainer._feature_extractor is None


def test_extra_trainable_params_builds_rep_proj(tiny_k2_model):
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args()
    result = trainer.extra_trainable_params(args, FakeAccelerator(), None, tiny_k2_model, [])
    assert trainer.rep_proj is not None
    assert len(result) == 1
    assert next(iter(trainer.rep_proj.parameters())) in list(result[0]["params"])


def test_on_train_start_snapshots_ema():
    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args()
    net = StubNetwork()
    trainer.on_train_start(args, FakeAccelerator(), net, None, None)
    assert torch.equal(trainer.ema_lora_state["lora_w"], torch.ones(4))
    net.lora_w.data.fill_(5.0)
    assert torch.equal(trainer.ema_lora_state["lora_w"], torch.ones(4))


def test_no_self_flow_is_noop():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    trainer.on_train_start(args, FakeAccelerator(), StubNetwork(), None, None)
    assert trainer.ema_lora_state is None


def test_network_weights_ema_without_network_weights_raises():
    import pytest

    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args(network_weights_ema="ema.safetensors", network_weights=None)
    with pytest.raises(ValueError, match="network_weights"):
        trainer.on_train_start(args, FakeAccelerator(), StubNetwork(), None, None)


def test_on_post_optimizer_step_lerps_only_on_sync():
    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args(ema_decay=0.9)
    net = StubNetwork()
    trainer.on_train_start(args, FakeAccelerator(), net, None, None)

    net.lora_w.data.fill_(2.0)
    trainer.on_post_optimizer_step(args, FakeAccelerator(), net, None, sync_gradients=False, global_step=1)
    assert torch.equal(trainer.ema_lora_state["lora_w"], torch.ones(4))  # unchanged

    trainer.on_post_optimizer_step(args, FakeAccelerator(), net, None, sync_gradients=True, global_step=1)
    assert torch.allclose(trainer.ema_lora_state["lora_w"], torch.full((4,), 1.1))


def test_extra_metadata_reports_self_flow_keys():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(student_feature_layer=0, teacher_feature_layer=2)
    metadata = trainer.extra_metadata(args)
    assert metadata["ss_self_flow"] is True
    assert metadata["ss_self_flow_mask_ratio"] == args.mask_ratio


def test_extra_metadata_empty_without_self_flow():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    assert trainer.extra_metadata(args) == {}


def test_extra_step_logs_drains_self_flow_logs():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args()
    trainer._self_flow_logs = {"self_flow/gamma": 0.8}
    assert trainer.extra_step_logs(args, {}) == {"self_flow/gamma": 0.8}


def test_extra_step_logs_empty_without_self_flow():
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    trainer._self_flow_logs = {"self_flow/gamma": 0.8}
    assert trainer.extra_step_logs(args, {}) == {}


def _resume_warning_fired(caplog):
    return any(
        "resume" in rec.message.lower() and "ema" in rec.message.lower() for rec in caplog.records if rec.name == _LOGGER_NAME
    )


def test_on_train_start_warns_on_resume_without_companion_weights(caplog):
    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args(resume="/some/path/state", network_weights_ema=None, network_weights_proj=None)
    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        trainer.on_train_start(args, FakeAccelerator(), StubNetwork(), None, None)
    assert _resume_warning_fired(caplog)


def test_on_train_start_no_resume_warning_when_network_weights_ema_set(caplog):
    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args(
        resume="/some/path/state",
        network_weights_ema="ema.safetensors",
        network_weights="student.safetensors",
        network_weights_proj=None,
    )
    net = StubNetwork()
    net._weight_registry["ema.safetensors"] = 3.0
    net._weight_registry["student.safetensors"] = 1.0
    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        trainer.on_train_start(args, FakeAccelerator(), net, None, None)
    assert not _resume_warning_fired(caplog)


def test_on_train_start_no_resume_warning_when_network_weights_proj_set(caplog):
    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args(resume="/some/path/state", network_weights_ema=None, network_weights_proj="proj.safetensors")
    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        trainer.on_train_start(args, FakeAccelerator(), StubNetwork(), None, None)
    assert not _resume_warning_fired(caplog)


def test_on_train_start_no_resume_warning_when_resume_not_set(caplog):
    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args(resume=None, network_weights_ema=None, network_weights_proj=None)
    with caplog.at_level("WARNING", logger=_LOGGER_NAME):
        trainer.on_train_start(args, FakeAccelerator(), StubNetwork(), None, None)
    assert not _resume_warning_fired(caplog)
