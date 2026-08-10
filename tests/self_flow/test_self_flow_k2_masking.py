import torch

from boo_musubi_tuner.self_flow.krea2_train_network_self_flow import apply_per_token_mask


def test_mask_ratio_zero_returns_student_unchanged():
    student = torch.randn(2, 4, 1, 8, 8)
    teacher = torch.randn(2, 4, 1, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 0.0, patch=2, device=torch.device("cpu"))
    assert torch.equal(out, student)
    assert mask.shape == (2, 16)  # (8/2) * (8/2) = 16 patch tokens
    assert not mask.any()


def test_mask_ratio_one_returns_teacher():
    student = torch.zeros(2, 4, 1, 8, 8)
    teacher = torch.ones(2, 4, 1, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 1.0, patch=2, device=torch.device("cpu"))
    assert torch.equal(out, teacher)
    assert mask.all()


def test_masked_tokens_expand_to_full_patch_block():
    """A masked patch-token must select the teacher value across its ENTIRE
    patch x patch pixel block, not just one pixel — this is the
    repeat_interleave expansion the design requires."""
    torch.manual_seed(0)
    student = torch.zeros(1, 4, 1, 8, 8)
    teacher = torch.ones(1, 4, 1, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 0.5, patch=2, device=torch.device("cpu"))
    h_tok, w_tok = 4, 4  # 8/2
    mask_grid = mask.view(1, h_tok, w_tok)
    for i in range(h_tok):
        for j in range(w_tok):
            block = out[0, :, 0, i * 2 : i * 2 + 2, j * 2 : j * 2 + 2]
            if mask_grid[0, i, j]:
                assert torch.equal(block, torch.ones_like(block)), f"token ({i},{j}) masked but block is not all-teacher"
            else:
                assert torch.equal(block, torch.zeros_like(block)), f"token ({i},{j}) unmasked but block is not all-student"


def test_approximate_ratio():
    torch.manual_seed(0)
    student = torch.zeros(8, 4, 1, 64, 64)
    teacher = torch.ones(8, 4, 1, 64, 64)
    _, mask = apply_per_token_mask(student, teacher, 0.25, patch=2, device=torch.device("cpu"))
    assert abs(mask.float().mean().item() - 0.25) < 0.03


def test_tensor_ratio_per_sample_zero_one():
    torch.manual_seed(42)
    B = 2
    student = torch.zeros(B, 4, 1, 8, 8)
    teacher = torch.ones(B, 4, 1, 8, 8)
    ratio = torch.tensor([0.0, 1.0])
    out, mask = apply_per_token_mask(student, teacher, ratio, patch=2, device=torch.device("cpu"))
    assert not mask[0].any()
    assert torch.equal(out[0], student[0])
    assert mask[1].all()
    assert torch.equal(out[1], teacher[1])


def test_patch_size_one_matches_pixel_granularity():
    """patch=1 degenerates to one token per pixel — sanity check against the
    conceptually simplest case."""
    torch.manual_seed(0)
    student = torch.zeros(1, 4, 1, 4, 4)
    teacher = torch.ones(1, 4, 1, 4, 4)
    out, mask = apply_per_token_mask(student, teacher, 0.5, patch=1, device=torch.device("cpu"))
    assert mask.shape == (1, 16)
    mask_spatial = mask.view(1, 1, 1, 4, 4).expand_as(student)
    assert torch.equal(out[mask_spatial], teacher[mask_spatial])
    assert torch.equal(out[~mask_spatial], student[~mask_spatial])
