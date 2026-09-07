import torch
from musubi_tuner.krea2.krea2_mmdit import SingleStreamDiT

from tests.self_flow.conftest_k2_self_flow import make_k2_batch
from tests.tdm_distill.test_tdm_process_batch import (
    _prepared_trainer,
    _StubDinov3Embedder,
    _StubVae,
    make_noise_scheduler,
)


def test_diversity_rollout_uses_its_own_step_count_not_main_k(tiny_k2_config):
    """Main K is sampled from --tdm_step_counts (forced to 8 here); the diversity rollout must
    use --tdm_diversity_step_count (set to 2) instead, regardless of what K got drawn."""
    torch.manual_seed(3)
    tiny_k2_model = SingleStreamDiT(tiny_k2_config, attn_mode="torch")
    tiny_k2_model.eval()
    trainer, args, acc, net, handle = _prepared_trainer(
        tiny_k2_model,
        tdm_step_counts="8",
        tdm_diversity_group_size=3,
        tdm_diversity_step_count=2,
    )
    for p in tiny_k2_model.parameters():
        p.requires_grad_(True)
    trainer._dinov3_embedder = _StubDinov3Embedder()
    batch, latents, noise = make_k2_batch(B=1, H=8, W=8, n_txt=3)
    scheduler = make_noise_scheduler(args)
    vae = _StubVae()

    rollout_step_counts = []
    orig_rollout = trainer._student_rollout

    def spy_rollout(*a, **kw):
        rollout_step_counts.append(a[4] if len(a) > 4 else kw.get("num_steps"))
        return orig_rollout(*a, **kw)

    trainer._student_rollout = spy_rollout

    trainer.process_batch(
        args, acc, tiny_k2_model, net, batch, latents, noise, scheduler, torch.float32, torch.float32, vae, global_step=0
    )
    handle.remove()

    # Two rollouts: main student rollout (K=8, the only value in tdm_step_counts) and the
    # diversity rollout (tdm_diversity_step_count=2).
    assert rollout_step_counts == [8, 2]
