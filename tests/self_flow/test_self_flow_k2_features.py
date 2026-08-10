import pytest
import torch
from einops import repeat

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import BlockFeatureExtractor


def _forward(model, B=1, H=8, W=8, n_txt=3):
    patch = model.config.patch
    h_, w_ = H // patch, W // patch
    imglen = h_ * w_
    img = torch.randn(B, imglen, model.config.channels * patch**2)
    context = torch.randn(B, n_txt, model.config.txtlayers, model.config.txtdim)
    imgids = torch.zeros((h_, w_, 3))
    imgids[..., 1] = torch.arange(h_)[:, None]
    imgids[..., 2] = torch.arange(w_)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=B, three=3)
    txtpos = torch.zeros(B, n_txt, 3)
    pos = torch.cat((imgpos, txtpos), dim=1)
    mask = torch.ones(B, imglen + n_txt, dtype=torch.bool)
    model(img=img, context=context, t=torch.rand(B), pos=pos, mask=mask)
    return imglen


def test_install_out_of_range_raises(tiny_k2_model):
    extractor = BlockFeatureExtractor()
    num_blocks = len(tiny_k2_model.blocks)
    with pytest.raises(ValueError, match="out of range"):
        extractor.install(tiny_k2_model, [num_blocks])


def test_arm_without_install_raises(tiny_k2_model):
    extractor = BlockFeatureExtractor()
    extractor.install(tiny_k2_model, [0])
    with pytest.raises(ValueError, match="not installed"):
        extractor.arm(1, imglen=16)


def test_captures_image_token_prefix_only(tiny_k2_model):
    extractor = BlockFeatureExtractor()
    extractor.install(tiny_k2_model, [0, 2])
    extractor.arm(0, imglen=0)  # placeholder, real imglen set right before forward

    imglen = None

    def run_and_capture(layer):
        nonlocal imglen
        B, H, W, n_txt = 1, 8, 8, 3
        patch = tiny_k2_model.config.patch
        h_, w_ = H // patch, W // patch
        imglen = h_ * w_
        extractor.arm(layer, imglen)
        _forward(tiny_k2_model, B=B, H=H, W=W, n_txt=n_txt)
        return extractor.drain()

    feat0 = run_and_capture(0)
    feat2 = run_and_capture(2)
    extractor.remove()

    assert feat0 is not None and feat2 is not None
    assert feat0.shape == (1, imglen, tiny_k2_model.config.features)
    assert feat2.shape == (1, imglen, tiny_k2_model.config.features)
    assert not torch.equal(feat0, feat2)  # different layers, different features


def test_drain_clears_state(tiny_k2_model):
    extractor = BlockFeatureExtractor()
    extractor.install(tiny_k2_model, [0])
    extractor.arm(0, imglen=16)
    _forward(tiny_k2_model)
    feat = extractor.drain()
    assert feat is not None
    assert extractor.drain() is None  # second drain returns None: state cleared
