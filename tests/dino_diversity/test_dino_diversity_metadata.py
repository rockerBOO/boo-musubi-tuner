from boo_musubi_tuner.dino_diversity.krea2_train_network_dino_diversity import Krea2DinoDiversityNetworkTrainer

from .test_dino_diversity_arg_validation import make_args


def test_extra_metadata_includes_dino_diversity_fields():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity_group_size=6, dino_diversity_step_count=8)
    trainer.handle_model_specific_args(args)
    metadata = trainer.extra_metadata(args)
    assert metadata["ss_dino_diversity"] is True
    assert metadata["ss_dino_diversity_group_size"] == 6
    assert metadata["ss_dino_diversity_step_count"] == 8


def test_extra_metadata_vanilla_fallthrough_when_disabled():
    trainer = Krea2DinoDiversityNetworkTrainer()
    args = make_args(dino_diversity=False)
    metadata_before = dict(trainer.extra_metadata(args))
    assert "ss_dino_diversity" not in metadata_before
