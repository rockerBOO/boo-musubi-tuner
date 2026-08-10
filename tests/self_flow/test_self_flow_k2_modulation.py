import torch
from einops import repeat

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import PerTokenModulationController


def _build_forward_inputs(model, B=1, H=8, W=8, n_txt=3):
    patch = model.config.patch
    h_, w_ = H // patch, W // patch
    imglen = h_ * w_
    img = torch.randn(B, imglen, model.config.channels * patch * patch)
    context = torch.randn(B, n_txt, model.config.txtlayers, model.config.txtdim)
    imgids = torch.zeros((h_, w_, 3))
    imgids[..., 1] = torch.arange(h_)[:, None]
    imgids[..., 2] = torch.arange(w_)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=B, three=3)
    txtpos = torch.zeros(B, n_txt, 3)
    pos = torch.cat((imgpos, txtpos), dim=1)
    mask = torch.ones(B, imglen + n_txt, dtype=torch.bool)
    fulllen = imglen + n_txt
    padlen = (-fulllen) % 256
    N = fulllen + padlen
    return img, context, pos, mask, imglen, N


def test_pass_through_when_not_staged(tiny_k2_model):
    torch.manual_seed(0)
    img, context, pos, mask, imglen, N = _build_forward_inputs(tiny_k2_model)
    t = torch.rand(1)

    # tiny_k2_model is already .eval()'d (see conftest.py), so there is
    # no dropout/stochastic-layer randomness between calls: reusing the exact same
    # input tensors across all three forward passes makes bit-identity comparisons
    # meaningful (torch.equal, not just shape).
    baseline = tiny_k2_model(img=img, context=context, t=t, pos=pos, mask=mask)

    controller = PerTokenModulationController()
    controller.install(tiny_k2_model)
    with_hooks_unstaged = tiny_k2_model(img=img, context=context, t=t, pos=pos, mask=mask)
    assert torch.equal(with_hooks_unstaged, baseline), (
        "installing the controller without staging must be a bit-identical pass-through"
    )

    controller.remove()
    after_remove = tiny_k2_model(img=img, context=context, t=t, pos=pos, mask=mask)
    assert torch.equal(after_remove, baseline), "remove() must restore the original bound forward method exactly"


def test_staged_produces_per_token_variation(tiny_k2_model):
    torch.manual_seed(0)
    img, context, pos, mask, imglen, N = _build_forward_inputs(tiny_k2_model)

    controller = PerTokenModulationController()
    controller.install(tiny_k2_model)

    tau = torch.full((1, N), 0.5)
    tau[:, : imglen // 2] = 0.1
    tau[:, imglen // 2 : imglen] = 0.9
    controller.stage(tau)
    out = tiny_k2_model(img=img, context=context, t=torch.rand(1), pos=pos, mask=mask)
    controller.remove()

    assert out.shape == (1, imglen, tiny_k2_model.config.channels * tiny_k2_model.config.patch**2)
    assert not torch.allclose(out[:, 0], out[:, imglen - 1]), "different per-token tau must yield different outputs"


def test_clear_disables_staging(tiny_k2_model):
    torch.manual_seed(0)
    img, context, pos, mask, imglen, N = _build_forward_inputs(tiny_k2_model)
    controller = PerTokenModulationController()
    controller.install(tiny_k2_model)
    controller.stage(torch.full((1, N), 0.5))
    controller.clear()
    # After clear(), forward must not crash (falls back to original vanilla path)
    out = tiny_k2_model(img=img, context=context, t=torch.rand(1), pos=pos, mask=mask)
    controller.remove()
    assert out.shape[1] == imglen


def test_missing_tmlp_raises():
    class FakeModel:
        pass

    controller = PerTokenModulationController()
    import pytest

    with pytest.raises(AttributeError, match="tmlp"):
        controller.install(FakeModel())
