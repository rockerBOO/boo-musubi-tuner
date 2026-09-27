"""LossContextMixin: stashes call_dit inputs and keeps the base --loss_fn path intact."""

import argparse

import pytest
import torch
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer

from boo_musubi_tuner.loss_context.flux_2_train_network_loss_context import Flux2LossContextNetworkTrainer
from boo_musubi_tuner.loss_context.krea2_train_network_loss_context import Krea2LossContextNetworkTrainer
from boo_musubi_tuner.loss_context.loss_context import STASH_KEYS, LossContextMixin


class _FakeBase(NetworkTrainer):
    """Stands in for a model trainer: velocity target, pred = target + small error."""

    def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kw):
        target = noise - latents
        pred = target + 0.1 * torch.randn_like(target)
        return DiTOutput(pred=pred.requires_grad_(True), target=target)


class _Trainer(LossContextMixin, _FakeBase):
    pass


def _step_inputs(shape=(2, 8, 32, 32), t=(100.0, 600.0)):
    torch.manual_seed(0)
    latents, noise = torch.randn(shape), torch.randn(shape)
    ts = torch.tensor(t)
    s = (ts / 1000).view(-1, *([1] * (len(shape) - 1)))
    return latents, noise, (1 - s) * latents + s * noise, ts


def _call(trainer, shape=(2, 8, 32, 32)):
    latents, noise, noisy, ts = _step_inputs(shape)
    out = trainer.call_dit(argparse.Namespace(), None, None, latents, {}, noise, noisy, ts, torch.float32)
    return out, (latents, noise, noisy, ts)


def test_stashes_the_exact_tensors():
    out, (latents, noise, noisy, _) = _call(_Trainer())
    assert set(STASH_KEYS) <= out.extra.keys()
    assert out.extra["noisy_model_input"] is noisy
    assert out.extra["latents"] is latents
    assert out.extra["noise"] is noise


@pytest.mark.parametrize("shape", [(2, 8, 32, 32), (2, 8, 1, 32, 32)])  # FLUX.2 4D, Krea 2 5D
def test_base_compute_loss_runs_wavelet_loss_fn_that_needs_the_stash(shape):
    pytest.importorskip("wavelet_loss")
    trainer = _Trainer()
    args = argparse.Namespace(
        loss_fn="wavelet_loss.musubi.WaveletPlusX0Huber",
        loss_fn_args=[
            "alpha=1.0",
            "loss_type='x0_huber'",
            "energy_beta=0.1",
            "mottle_metrics=True",
            "transform_type='swt'",
            "wavelet='sym7'",
            "level=2",
            "band_weights={'ll': 0.0, 'lh': 1.0, 'hl': 1.0, 'hh': 1.0}",
        ],
        weighting_scheme="none",
    )
    trainer._resolved_loss_fn = None
    out, (_, _, _, ts) = _call(trainer, shape)
    loss, metrics = trainer.compute_loss(args, out, ts, None, torch.float32, torch.float32, 0)
    loss.backward()
    assert torch.isfinite(loss) and out.pred.grad is not None
    assert "loss/energy" in metrics and any(k.startswith("mottle/M_flat_") for k in metrics)


@pytest.mark.parametrize(
    "trainer_cls, base_cls",
    [(Flux2LossContextNetworkTrainer, Flux2NetworkTrainer), (Krea2LossContextNetworkTrainer, Krea2NetworkTrainer)],
)
def test_mixin_precedes_base_in_mro(trainer_cls, base_cls):
    mro = trainer_cls.__mro__
    assert mro.index(LossContextMixin) < mro.index(base_cls)
    assert trainer_cls.compute_loss is NetworkTrainer.compute_loss  # --loss_fn path untouched


@pytest.mark.parametrize("setup", [flux2_setup_parser, krea2_setup_parser])
def test_parser_wiring_accepts_loss_fn(setup):
    parser = setup(setup_parser_common())
    args, _ = parser.parse_known_args(["--loss_fn", "wavelet_loss.musubi.WaveletPlusX0Huber", "--loss_fn_args", "alpha=1.0"])
    assert args.loss_fn == "wavelet_loss.musubi.WaveletPlusX0Huber"
