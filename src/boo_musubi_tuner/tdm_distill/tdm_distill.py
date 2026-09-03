"""TDM (Trajectory Distribution Matching, arXiv:2503.06674) distillation core math,
architecture-generic — pure tensor ops, no model-specific code. Model wiring lives in
per-architecture trainer files (e.g. krea2_train_network_tdm_distill.py), mirroring
self_flow.py's split between shared math and per-architecture trainers.

Internal extension point — no API stability guarantees. Experimental: see
docs/tdm-distill.md for known simplifications vs. the paper.
"""

import math

import torch
from transformers import AutoImageProcessor, AutoModel


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


def sample_step_count(step_counts: list[int], generator: "torch.Generator | None" = None) -> int:
    """Uniformly draw one student sampling-step count K from the configured list (TDM Eq. 8's
    sampling-steps-aware objective — makes the student usable across multiple step budgets)."""
    idx = torch.randint(0, len(step_counts), (1,), generator=generator).item()
    return step_counts[idx]


def sample_trajectory_interval(num_steps: int, generator: "torch.Generator | None" = None) -> int:
    """Uniformly pick one interval index along a num_steps-point trajectory (TDM's non-overlapping
    interval sampling — a single fake-score forward suffices per training iteration). A
    single-step trajectory (num_steps == 1) has exactly one interval, at index 0."""
    if num_steps == 1:
        return 0
    return int(torch.randint(0, num_steps - 1, (1,), generator=generator).item())


def pairwise_cosine_diversity(embeddings: torch.Tensor) -> float:
    """Mean pairwise cosine distance (1 - cosine_similarity) across all unordered pairs of rows
    in embeddings (N, D). Higher = more diverse. Ported from the krea2-diversity probe project's
    identical metric (see docs/tdm-distill.md)."""
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError(f"pairwise_cosine_diversity requires N >= 2 embeddings, got N={n}")
    normed = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    sim_matrix = normed @ normed.T
    triu_indices = torch.triu_indices(n, n, offset=1)
    pair_sims = sim_matrix[triu_indices[0], triu_indices[1]]
    distances = 1.0 - pair_sims
    return distances.mean().item()


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


def diversity_loss_from_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Negated mean pairwise cosine distance across all unordered pairs, kept fully
    differentiable (no .item()/float conversion) so gradient can flow back through
    embeddings to whatever produced them. Minimizing this loss maximizes diversity.

    Deliberately duplicates pairwise_cosine_diversity's math rather than calling it: that
    helper ends in .item() (it is the metric-only reporting form) which would sever the graph.
    Used as L_div, added (weighted, not annealed) to the TDM student loss."""
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError(f"diversity_loss_from_embeddings requires N >= 2 embeddings, got N={n}")
    normed = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    sim_matrix = normed @ normed.T
    triu_indices = torch.triu_indices(n, n, offset=1)
    pair_sims = sim_matrix[triu_indices[0], triu_indices[1]]
    distances = 1.0 - pair_sims
    return -distances.mean()


class Dinov3ImageEmbedder:
    """Wraps a DINOv3 backbone for image-only embedding extraction (CLS token). Ported from the
    krea2-diversity probe project's identical class — requires one-time Hugging Face license
    acceptance for facebook/dinov3-vitb16-pretrain-lvd1689m plus `huggingface-cli login` before
    first use; no unit test here (integration-only, see docs/tdm-distill.md)."""

    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", device: str = "cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        # embed_differentiable runs without torch.no_grad() (gradient must reach the *input*
        # pixels), and .eval() does not stop grad accumulation. This backbone is frozen and in no
        # optimizer, so without requires_grad_(False) every backward would park a permanent .grad
        # buffer on each parameter (~model-sized leak, nothing ever zeroes it). Freezing the
        # parameters does not sever the graph through them: autograd still backprops to the input.
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Preprocessing params mirrored off the real processor for embed_differentiable's manual
        # tensor path. DINOv3ViTImageProcessorFast exposes size={"height": 224, "width": 224} and
        # crop_size=None; the getattr fallbacks cover processor variants that expose crop_size
        # instead, and land on DINOv3's documented ImageNet defaults if neither is present.
        size = getattr(self.processor, "size", None) or getattr(self.processor, "crop_size", None) or {}
        self.image_size = int(size.get("height") or size.get("shortest_edge") or 224)
        self.image_mean = list(getattr(self.processor, "image_mean", None) or [0.485, 0.456, 0.406])
        self.image_std = list(getattr(self.processor, "image_std", None) or [0.229, 0.224, 0.225])

    @torch.no_grad()
    def embed(self, images: list) -> torch.Tensor:
        """Non-differentiable convenience path for inference/eval over PIL/numpy images.
        Training code that needs gradient must use embed_differentiable instead."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu()

    def embed_differentiable(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Embed an already-decoded (N, C, H, W) float tensor in [0, 1] (already on self.device),
        keeping the autograd graph intact back to pixel_values.

        Deliberately does NOT route through self.processor: that expects PIL/numpy input and its
        resize/rescale/normalize pipeline is not differentiable. Resize + normalize are done here
        with plain tensor ops using the same parameters DINOv3ViTImageProcessorFast reports for
        facebook/dinov3-vitb16-pretrain-lvd1689m (size 224x224, bilinear resample,
        image_mean/image_std = ImageNet stats), so the model still sees what it was trained on.
        No @torch.no_grad() here; self.model being in .eval() mode only disables dropout /
        batchnorm stat updates and does not block gradient computation.
        """
        resized = torch.nn.functional.interpolate(
            pixel_values,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = torch.tensor(self.image_mean, device=resized.device, dtype=resized.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.image_std, device=resized.device, dtype=resized.dtype).view(1, -1, 1, 1)
        normalized = (resized - mean) / std
        outputs = self.model(pixel_values=normalized.to(self.model.dtype))
        return outputs.last_hidden_state[:, 0, :]
