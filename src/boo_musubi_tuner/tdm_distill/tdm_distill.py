"""TDM (Trajectory Distribution Matching, arXiv:2503.06674) distillation core math,
architecture-generic — pure tensor ops, no model-specific code. Model wiring lives in
per-architecture trainer files (e.g. krea2_train_network_tdm_distill.py), mirroring
self_flow.py's split between shared math and per-architecture trainers.

Internal extension point — no API stability guarantees. Experimental: see
docs/tdm-distill.md for known simplifications vs. the paper.
"""

import math

import torch
import torch.nn.functional as F

from boo_musubi_tuner.diversity.diversity import (
    Dinov3ImageEmbedder,
    diversity_loss_from_embeddings,
    pairwise_cosine_diversity,
)

__all__ = [
    "Dinov3ImageEmbedder",
    "LoraRoleSwitcher",
    "cfg_combine",
    "critic_importance_weight",
    "diversity_loss_from_embeddings",
    "fake_score_denoising_loss",
    "forward_transition",
    "gaussian_blur_latent",
    "min_snr_weight",
    "mottle_excess_loss",
    "pairwise_cosine_diversity",
    "pseudo_huber_c",
    "pseudo_huber_loss",
    "revised_sample",
    "sample_step_count",
    "sample_trajectory_interval",
]


def pseudo_huber_c(data_dim: int) -> float:
    """TDM's pseudo-Huber constant (Eq. 11): 0.00054 * sqrt(d), d = per-sample element count."""
    return 0.00054 * math.sqrt(data_dim)


def pseudo_huber_loss(student_x: torch.Tensor, revised_x: torch.Tensor, c: float) -> torch.Tensor:
    """TDM's surrogate training objective (Eq. 11): per-example pseudo-Huber distance between
    the student's x_ti and the stopgrad revised target, mean-reduced over the batch."""
    diff_sq = (student_x - revised_x).pow(2)
    per_example = diff_sq.flatten(1).sum(dim=1)
    return (torch.sqrt(per_example + c**2) - c).mean()


def revised_sample(
    x_ti: torch.Tensor,
    real_score: torch.Tensor,
    fake_score: torch.Tensor,
    lambda_tau: "float | torch.Tensor",
) -> torch.Tensor:
    """TDM's revised target (Eq. 11 body): x_ti after one gradient-descent step toward the
    real score, away from the fake score. Caller is responsible for detaching the result
    (stopgrad) before use as a training target."""
    return x_ti + lambda_tau * (real_score - fake_score)


def cfg_combine(cond_score: torch.Tensor, uncond_score: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """Classifier-free guidance combination: uncond + scale * (cond - uncond)."""
    return uncond_score + guidance_scale * (cond_score - uncond_score)


def fake_score_denoising_loss(
    fake_score_pred: torch.Tensor,
    target: torch.Tensor,
    omega_tau: "float | torch.Tensor" = 1.0,
) -> torch.Tensor:
    """Fake-score critic training loss — a simplified (non-importance-weighted) version of
    TDM's Eq. 7: weighted MSE between the fake-score network's prediction and the denoising
    target. omega_tau is a plain scalar/per-example weight, not the paper's full importance-
    sampling ratio (documented simplification, see docs/tdm-distill.md)."""
    return (omega_tau * (fake_score_pred - target) ** 2).mean()


def forward_transition(x_t: torch.Tensor, t: float, tau: float, noise: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """Diffuse an ODE-trajectory sample x_t (already at noise level t) up to the noisier level tau >= t.

    Uses the forward transition kernel of x_s = (1-s)*x0 + s*eps: x_tau = a*x_t + s_std*noise with
    a = (1-tau)/(1-t) and s_std**2 = tau**2 - (a*t)**2. Unlike (1-tau)*x_t + tau*noise, which treats
    x_t as clean data, this keeps x_tau on the teacher's marginal at level tau.

    Returns (x_tau, a). Needs t < 1 and tau >= t.
    """
    a = (1.0 - tau) / (1.0 - t)
    var = max(tau**2 - (a * t) ** 2, 0.0)
    return a * x_t + math.sqrt(var) * noise, a


def critic_importance_weight(mixed_noise: torch.Tensor, rand_noise: torch.Tensor) -> torch.Tensor:
    """Per-example importance weight for the fake-score loss (TDM Eq. 7, official-code form):
    exp(-0.5*mean(mixed**2)) / exp(-0.5*mean(rand**2)), where mixed_noise is the noise x_tau
    carries relative to the clean estimate and rand_noise is the fresh noise actually drawn."""
    m = mixed_noise.flatten(1).pow(2).mean(dim=1)
    r = rand_noise.flatten(1).pow(2).mean(dim=1)
    return torch.exp(-0.5 * (m - r))


def gaussian_blur_latent(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over the last two dims of a (B, C, [1,] H, W) latent. Differentiable."""
    shape = x.shape
    h, w = shape[-2:]
    x4 = x.reshape(-1, 1, h, w)
    radius = max(math.ceil(3 * sigma), 1)
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (coords / sigma) ** 2)
    k = k / k.sum()
    x4 = F.pad(x4, (radius, radius, 0, 0), mode="reflect")
    x4 = F.conv2d(x4, k.view(1, 1, 1, -1))
    x4 = F.pad(x4, (0, 0, radius, radius), mode="reflect")
    x4 = F.conv2d(x4, k.view(1, 1, -1, 1))
    return x4.reshape(shape)


def mottle_excess_loss(
    x0_student: torch.Tensor,
    x0_ref: torch.Tensor,
    margin: float = 0.1,
    flat_quantile: float = 0.5,
    sigma: float = 1.5,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Opt-in anti-mottle term. Penalizes the student's clean estimate for carrying more
    latent-pixel high-pass energy than the reference (teacher) estimate in flat regions.

    High-pass = x - gaussian(x, sigma) per channel. Flat region = the lowest `flat_quantile`
    of the reference's local gradient magnitude (per example). Energy is matched, not minimized:
    only the excess over reference*(1+margin) is penalized, as a ratio, so detail the teacher
    also has is left alone. Gradient flows through the student only.

    Returns (loss, mean energy ratio student/reference) for logging.
    """
    ref = x0_ref.detach()
    hp_s = x0_student - gaussian_blur_latent(x0_student, sigma)
    hp_r = ref - gaussian_blur_latent(ref, sigma)

    low = gaussian_blur_latent(ref, 2.0)
    gy = low[..., 1:, :-1] - low[..., :-1, :-1]
    gx = low[..., :-1, 1:] - low[..., :-1, :-1]
    grad = (gx.pow(2) + gy.pow(2)).mean(dim=1)  # (B, [1,] h-1, w-1), mean over channels
    grad = F.pad(grad, (0, 1, 0, 1), mode="replicate")
    b = grad.shape[0]
    thr = torch.quantile(grad.reshape(b, -1).float(), flat_quantile, dim=1).view(b, *([1] * (grad.dim() - 1)))
    mask = (grad <= thr).to(x0_student.dtype).unsqueeze(1)  # (B, 1, [1,] H, W)

    denom = mask.flatten(1).sum(dim=1).clamp(min=1.0).view(b, 1)  # per example
    e_s = (hp_s.pow(2) * mask).flatten(2).sum(dim=2) / denom  # (B, C)
    e_r = (hp_r.pow(2) * mask).flatten(2).sum(dim=2) / denom
    ratio = e_s / (e_r + 1e-8)
    excess = F.relu(e_s - e_r * (1.0 + margin)) / (e_r + 1e-8)
    return excess.mean(), ratio.detach().mean()


def min_snr_weight(tau: float, gamma: float = 5.0) -> float:
    """Min-SNR-clamped timestep importance weight (omega_tau, TDM Eq. 7), re-derived for this
    module's velocity-space critic target (v = eps - x0, under x_tau = (1-tau)*x0 + tau*eps).
    Provably equivalent to running gamma-clamped min-SNR (Hang et al. 2023) on an x0-space loss:
    ||x0_pred - x0||^2 = tau**2 * ||v_pred - v||^2 for this parameterization, so weighting the
    v-loss by min(SNR(tau), gamma) * tau**2 reproduces the identical per-sample loss value."""
    return min((1 - tau) ** 2, gamma * tau**2)


def sample_step_count(step_counts: list[int], generator: "torch.Generator | None" = None) -> int:
    """Uniformly draw one student sampling-step count K from the configured list (TDM Eq. 8's
    sampling-steps-aware objective — makes the student usable across multiple step budgets)."""
    idx = torch.randint(0, len(step_counts), (1,), generator=generator).item()
    return step_counts[idx]


def sample_trajectory_interval(num_steps: int, generator: "torch.Generator | None" = None) -> int:
    """Uniformly pick one interval index along a num_steps-step trajectory (TDM's non-overlapping
    interval sampling — a single fake-score forward suffices per training iteration). A K-step
    trajectory has K intervals, indices 0..K-1; index K-1 is the interval landing on t = 0."""
    return int(torch.randint(0, num_steps, (1,), generator=generator).item())


class LoraRoleSwitcher:
    """Swaps one physical LoRA network between teacher/student/fake-score roles.

    Teacher = the network's multiplier forced to 0 (zero-cost — no weight copy, the wrapped
    transformer's forward reduces to its frozen raw base behavior). Student and fake-score are
    two independent trainable weight states of the *same* LoRA modules, held here and swapped
    into the network's live state_dict via load_state_dict immediately before each forward —
    the same trick self_flow's Krea2SelfFlowNetworkTrainer uses for its EMA teacher, extended
    from two states to three roles.
    """

    def __init__(self, network) -> None:
        self._network = network
        self._student_state: dict | None = None
        self._fake_score_state: dict | None = None
        self._active: str | None = None  # "student" | "fake_score" | None (teacher/uninitialized)

    def init_from(self, source_state: dict) -> None:
        """Seed both student and fake-score states from a shared warm start (the Turbo LoRA)."""
        self._student_state = {k: v.detach().clone() for k, v in source_state.items()}
        self._fake_score_state = {k: v.detach().clone() for k, v in source_state.items()}

    def use_teacher(self) -> None:
        self._sync_active_out()
        self._network.set_multiplier(0.0)
        self._active = None

    def use_student(self) -> None:
        self._sync_active_out()
        self._network.load_state_dict(self._student_state)
        self._network.set_multiplier(1.0)
        self._active = "student"

    def use_fake_score(self) -> None:
        self._sync_active_out()
        self._network.load_state_dict(self._fake_score_state)
        self._network.set_multiplier(1.0)
        self._active = "fake_score"

    @property
    def student_state(self) -> dict:
        if self._active == "student":
            self._sync_active_out()
        return self._student_state

    @property
    def fake_score_state(self) -> dict:
        if self._active == "fake_score":
            self._sync_active_out()
        return self._fake_score_state

    def _sync_active_out(self) -> None:
        """Write the currently-live weights back into their owning state dict before swapping
        away, so in-place optimizer updates made while a role was active are not lost."""
        if self._active == "student":
            self._student_state = {k: v.detach().clone() for k, v in self._network.state_dict().items()}
        elif self._active == "fake_score":
            self._fake_score_state = {k: v.detach().clone() for k, v in self._network.state_dict().items()}
