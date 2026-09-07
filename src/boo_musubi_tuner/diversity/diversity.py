"""Architecture-generic diversity math shared across extensions that penalize same-prompt
mode collapse (e.g. tdm_distill, dino_diversity). Pure tensor ops plus one HF-model wrapper —
no model-specific (Krea2/TDM/etc.) code belongs here.
"""

import torch
from transformers import AutoImageProcessor, AutoModel


def pairwise_cosine_diversity(embeddings: torch.Tensor) -> float:
    """Mean pairwise cosine distance (1 - cosine_similarity) across all unordered pairs of rows
    in embeddings (N, D). Higher = more diverse. This is the non-differentiable, metric-only
    form (ends in .item()); diversity_loss_from_embeddings below duplicates this math in a
    graph-preserving form for use as an actual training loss."""
    n = embeddings.shape[0]
    if n < 2:
        raise ValueError(f"pairwise_cosine_diversity requires N >= 2 embeddings, got N={n}")
    normed = torch.nn.functional.normalize(embeddings.float(), dim=-1)
    sim_matrix = normed @ normed.T
    triu_indices = torch.triu_indices(n, n, offset=1)
    pair_sims = sim_matrix[triu_indices[0], triu_indices[1]]
    distances = 1.0 - pair_sims
    return distances.mean().item()


def diversity_loss_from_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """Negated mean pairwise cosine distance across all unordered pairs, kept fully
    differentiable (no .item()/float conversion) so gradient can flow back through
    embeddings to whatever produced them. Minimizing this loss maximizes diversity.

    Deliberately duplicates pairwise_cosine_diversity's math rather than calling it: that
    helper ends in .item() (it is the metric-only reporting form) which would sever the graph."""
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
    """Wraps a DINOv3 backbone for image-only embedding extraction (CLS token). Requires
    one-time Hugging Face license acceptance for facebook/dinov3-vitb16-pretrain-lvd1689m plus
    `huggingface-cli login` (or `hf auth login`) before first use. No unit test here: constructing
    it downloads real pretrained weights, so it is exercised only by integration/manual runs, not
    the CPU-only test suite."""

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
