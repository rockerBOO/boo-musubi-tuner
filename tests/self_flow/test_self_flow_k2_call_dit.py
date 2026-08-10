import torch

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import Krea2SelfFlowNetworkTrainer
from tests.self_flow.conftest_k2_self_flow import make_k2_batch

from .test_self_flow_k2_arg_validation import make_args
from .test_self_flow_k2_lifecycle import FakeAccelerator


def _trainer_with_model(tiny_k2_model):
    trainer = Krea2SelfFlowNetworkTrainer()
    args = make_args(student_feature_layer=0, teacher_feature_layer=2)
    trainer.handle_model_specific_args(args)
    trainer.on_transformer_loaded(args, FakeAccelerator(), tiny_k2_model)
    return trainer, args


def test_call_dit_returns_features_and_pred(tiny_k2_model):
    trainer, args = _trainer_with_model(tiny_k2_model)
    batch, latents, noise = make_k2_batch(B=2, H=8, W=8, n_txt=3)
    timesteps = torch.tensor([500.0, 800.0])
    noisy = latents

    output = trainer.call_dit(
        args,
        FakeAccelerator(),
        tiny_k2_model,
        latents,
        batch,
        noise,
        noisy,
        timesteps,
        torch.float32,
        hidden_features=True,
        feature_layer=0,
    )
    imglen = (8 // tiny_k2_model.config.patch) ** 2
    assert output.pred.shape[0] == 2
    assert output.extra["features"] is not None
    assert output.extra["features"].shape == (2, imglen, tiny_k2_model.config.features)


def test_call_dit_per_token_map_staged_and_cleared(tiny_k2_model):
    trainer, args = _trainer_with_model(tiny_k2_model)
    batch, latents, noise = make_k2_batch(B=2, H=8, W=8, n_txt=3)
    timesteps = torch.tensor([500.0, 800.0])
    imglen = (8 // tiny_k2_model.config.patch) ** 2
    tau = torch.full((2, imglen), 800.0)

    trainer.call_dit(
        args,
        FakeAccelerator(),
        tiny_k2_model,
        latents,
        batch,
        noise,
        latents,
        timesteps,
        torch.float32,
        hidden_features=True,
        feature_layer=0,
        per_token_timesteps=tau,
    )
    assert trainer._modulation_controller._tau is None  # cleared after forward


def test_call_dit_hidden_features_requires_grad_in_train_mode(tiny_k2_model):
    trainer, args = _trainer_with_model(tiny_k2_model)
    tiny_k2_model.train()
    batch, latents, noise = make_k2_batch(B=2, H=8, W=8, n_txt=3)
    timesteps = torch.tensor([500.0, 800.0])

    with torch.enable_grad():
        output = trainer.call_dit(
            args,
            FakeAccelerator(),
            tiny_k2_model,
            latents,
            batch,
            noise,
            latents,
            timesteps,
            torch.float32,
            hidden_features=True,
            feature_layer=0,
        )

    features = output.extra["features"]
    assert features is not None
    assert features.requires_grad is True


def test_call_dit_vanilla_fallthrough_without_kwargs(tiny_k2_model):
    trainer, args = _trainer_with_model(tiny_k2_model)
    batch, latents, noise = make_k2_batch(B=2, H=8, W=8, n_txt=3)
    timesteps = torch.tensor([500.0, 800.0])

    output = trainer.call_dit(args, FakeAccelerator(), tiny_k2_model, latents, batch, noise, latents, timesteps, torch.float32)
    assert output.extra == {}
