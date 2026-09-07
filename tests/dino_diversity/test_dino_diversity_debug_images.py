"""Tests for the debug image dump: the exact post-VAE-decode, pre-DINOv3 pixel tensor must be
inspectable on disk, since a bad decode/rollout can still produce a numerically-plausible
diversity loss."""

import torch

from boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity import Krea2DinoDiversityNetworkTrainer

from .test_dino_diversity_arg_validation import make_args


def test_debug_images_saved_when_enabled(tmp_path):
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity_debug_save_images=True, output_dir=str(tmp_path))
    trainer.handle_model_specific_args(args)

    pixel_batch = torch.rand(3, 3, 16, 16)
    trainer._maybe_save_debug_images(args, pixel_batch, global_step=5)

    saved = sorted((tmp_path / "dino_diversity_debug").glob("*.png"))
    assert len(saved) == 3
    assert saved[0].name == "step00000005_sample00.png"


def test_debug_images_not_saved_when_disabled(tmp_path):
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity_debug_save_images=False, output_dir=str(tmp_path))
    trainer.handle_model_specific_args(args)

    pixel_batch = torch.rand(2, 3, 16, 16)
    trainer._maybe_save_debug_images(args, pixel_batch, global_step=5)

    assert not (tmp_path / "dino_diversity_debug").exists()


def test_debug_images_respect_save_every_n_steps(tmp_path):
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(
        dino_diversity_debug_save_images=True,
        dino_diversity_debug_save_every_n_steps=10,
        output_dir=str(tmp_path),
    )
    trainer.handle_model_specific_args(args)

    pixel_batch = torch.rand(1, 3, 16, 16)
    trainer._maybe_save_debug_images(args, pixel_batch, global_step=5)
    assert not (tmp_path / "dino_diversity_debug").exists()

    trainer._maybe_save_debug_images(args, pixel_batch, global_step=10)
    saved = list((tmp_path / "dino_diversity_debug").glob("*.png"))
    assert len(saved) == 1
