"""Self-Flow training entry point for Krea 2 (K2).

Implements Self-Supervised Flow Matching (Self-Flow, arXiv:2603.06507) on the
K2 backbone via extension seams and runtime forward hooks/monkeypatches —
zero modifications to ``krea2_mmdit.py`` or any base trainer are required.

Per-token conditioning is achieved with one ``register_forward_hook`` on
``model.tmlp`` (which drives every block's modulation automatically via
``tproj``, a pure elementwise broadcast) plus one runtime instance-attribute
monkeypatch of ``model.last.modulation.forward`` (the final layer's
``SimpleModulation`` cannot take a per-token vec via hooking alone — its
internal scale/shift broadcast trick only supports a token-axis size of 1 or
2, and raises inside the original forward before any hook can intervene).

Limitations (first pass, matching the FLUX.2 port): coupling-prob decay
schedules are constant-only, patch-locality mask modes are not ported.

Internal extension point — no API stability guarantees.
"""

import argparse
import logging
import os

import torch
from accelerate import Accelerator
from musubi_tuner.hv_train_network import read_config_from_file, setup_parser_common
from musubi_tuner.krea2.krea2_mmdit import temb
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from musubi_tuner.utils import huggingface_utils
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# region self-flow math helpers (ported verbatim from flux_2_train_network_self_flow.py —
# these 7 functions are architecture-agnostic pure tensor ops; only apply_per_token_mask
# differs for K2, see Task 3)


def assign_teacher_student_timesteps(timesteps_a: torch.Tensor, timesteps_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample teacher/student split: teacher = min (cleaner), student = max.

    Paper Eq. 4: masked tokens get s, unmasked get t; teacher sees the uniform
    cleaner timestep tau_min = min(t, s). See build_per_token_timestep_map for
    the per-token map construction.
    """
    t_a = timesteps_a.float()
    t_b = timesteps_b.float()
    timesteps_teacher = torch.min(t_a, t_b).to(timesteps_a.dtype)
    timesteps_student = torch.max(t_a, t_b).to(timesteps_a.dtype)
    return timesteps_teacher, timesteps_student


def reconstruct_noisy_input(latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Flow-matching interpolation (1-t)*latents + t*noise; timesteps in [1, 1001]."""
    t = (timesteps.float() - 1.0) / 1000.0
    if latents.ndim == 5:
        t_exp = t.view(-1, 1, 1, 1, 1)
    else:
        t_exp = t.view(-1, 1, 1, 1)
    return (1 - t_exp) * latents + t_exp * noise


def apply_per_token_mask(
    noisy_input_student: torch.Tensor,
    noisy_input_teacher: torch.Tensor,
    mask_ratio: "float | torch.Tensor",
    patch: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-patch-token masking (paper Eq. 4-5), K2-specific.

    Unlike FLUX.2 (patch=1, one token per latent pixel), K2 groups
    ``patch x patch`` latent pixels into one DiT token before the DiT ever
    sees it. The random mask/mismatch draw must therefore happen once per
    DiT token (shape (B, h_*w_)), not once per latent pixel — otherwise the
    mask used here (pixel-space) and the mask used for the modulation hook
    (token-space) would disagree. Each token's draw is expanded to its full
    ``patch x patch`` pixel block via repeat_interleave before selecting
    between student/teacher.

    ``noisy_input_student``/``noisy_input_teacher`` are (B, C, 1, H, W) —
    K2 always has a single-frame T=1 axis.

    Returns (masked_input, mask_tok) with mask_tok (B, h_*w_) bool,
    True = masked (this sample's patch-token takes the teacher/cleaner value).
    """
    B, C, T, H, W = noisy_input_student.shape
    assert T == 1, f"K2 expects single-frame latents, got T={T}"
    h_tok, w_tok = H // patch, W // patch
    N = h_tok * w_tok

    ratio = torch.as_tensor(mask_ratio, dtype=torch.float32, device=device)
    if ratio.ndim == 0:
        ratio = ratio.expand(B)
    mask_tok = torch.rand(B, N, device=device) < ratio.unsqueeze(1)  # (B, N)

    mask_grid = mask_tok.view(B, 1, 1, h_tok, w_tok)
    mask_pixel = mask_grid.repeat_interleave(patch, dim=3).repeat_interleave(patch, dim=4)
    mask_pixel = mask_pixel.expand(B, C, T, H, W)

    masked_input = torch.where(mask_pixel, noisy_input_teacher, noisy_input_student)
    return masked_input, mask_tok


def build_per_token_timestep_map(
    timesteps_teacher: torch.Tensor,
    timesteps_student: torch.Tensor,
    mask_flat: torch.Tensor,
    mismatch_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token timestep map for dual-timestep conditioning (paper Eq. 4).

    Unmasked tokens get the student timestep. Masked tokens get the teacher
    timestep, except with probability mismatch_prob they get the student
    timestep (deliberate mismatch — experimental, 0.0 = paper behaviour).
    Returns (per_token_timesteps (B, N), mismatch_mask (B, N) bool).
    """
    B, N = mask_flat.shape
    t_student = timesteps_student.unsqueeze(1).expand(B, N)
    t_teacher = timesteps_teacher.unsqueeze(1).expand(B, N)

    if mismatch_prob <= 0.0:
        per_token_t = torch.where(mask_flat, t_teacher, t_student)
        return per_token_t, torch.zeros_like(mask_flat)

    coin = torch.rand(B, N, device=mask_flat.device)
    mismatch_mask = mask_flat & (coin < mismatch_prob)
    per_token_t = torch.where(mask_flat & ~mismatch_mask, t_teacher, t_student)
    return per_token_t, mismatch_mask


def update_ema_weights(
    ema_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    decay: float,
) -> None:
    """In-place EMA update: ema = decay * ema + (1 - decay) * current."""
    with torch.no_grad():
        for k, v in current_state.items():
            if v.is_floating_point():
                ema_state[k].lerp_(v, 1 - decay)
            else:
                ema_state[k].copy_(v)


def compute_ema_weight_drift(
    ema_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Mean L2 distance between EMA and current weights (floating-point only)."""
    with torch.no_grad():
        dists = [
            torch.linalg.vector_norm(current_state[k].float() - v.float()) for k, v in ema_state.items() if v.is_floating_point()
        ]
        if not dists:
            return torch.tensor(0.0)
        return torch.stack(dists).mean()


def compute_representation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    rep_proj: torch.nn.Module,
) -> torch.Tensor:
    """L_rep (paper Eq. 6): negative mean cosine similarity of projected student vs teacher."""
    student_proj = rep_proj(student_features)
    cos_sim = torch.nn.functional.cosine_similarity(student_proj, teacher_features, dim=-1)
    return -cos_sim.mean()


def effective_gamma(gamma: float, global_step: int, warmup_steps: int) -> float:
    """Linear warmup of the L_rep weight: 0 -> gamma over warmup_steps, then constant."""
    if warmup_steps <= 0:
        return gamma
    return gamma * min(1.0, global_step / warmup_steps)


# endregion self-flow math helpers


class PerTokenModulationController:
    """Reroutes K2's modulation path to per-token timesteps via one forward
    hook plus one instance-attribute monkeypatch.

    Verified mechanism: ``model.tmlp``'s output is (B, 1, features) — the middle "1" is a global
    token axis that nn.Linear/GELU leave untouched all the way through
    ``tproj`` and into every ``SingleStreamBlock``'s modulation (a pure
    elementwise add, no shape assumptions). Hooking ``tmlp`` to expand that
    axis to (B, N, features) is therefore sufficient for every block.

    The final layer is the exception: ``LastLayer.modulation``
    (``SimpleModulation``) does ``vec + rearrange(self.lin, "two d -> 1 two d")``,
    which only broadcasts when vec's middle dim is 1 or 2 — with real N this
    raises inside the original forward, before any ``register_forward_hook``
    could override the return value. So instead of a hook, the modulation
    submodule's bound ``forward`` method is replaced at the instance level
    (not the class, not the source file) with an equivalent elementwise
    computation that works for any leading shape.

    Install on the raw (unwrapped) model before accelerator.prepare.
    """

    def __init__(self) -> None:
        self._handles: list = []
        self._tau: torch.Tensor | None = None
        self._tdim: int | None = None
        self._last_mod: torch.nn.Module | None = None
        self._orig_last_mod_forward = None

    def install(self, model) -> None:
        if not hasattr(model, "tmlp"):
            raise AttributeError(
                "K2 model has no module 'tmlp' — upstream may have renamed it; the Self-Flow per-token hooks need updating."
            )
        self._tdim = model.config.tdim
        self._handles.append(model.tmlp.register_forward_hook(self._tmlp_hook))

        self._last_mod = model.last.modulation
        self._orig_last_mod_forward = self._last_mod.forward
        self._last_mod.forward = self._last_mod_forward

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._last_mod is not None:
            self._last_mod.forward = self._orig_last_mod_forward
            self._last_mod = None
            self._orig_last_mod_forward = None

    def stage(self, tau: torch.Tensor) -> None:
        """Arm the hook for the next forward. tau (B, N), model scale [0, 1]."""
        self._tau = tau

    def clear(self) -> None:
        self._tau = None

    def _tmlp_hook(self, module, inputs, output):
        if self._tau is None:
            return None
        B, N = self._tau.shape
        emb = temb(self._tau.reshape(-1), self._tdim, device=output.device, dtype=output.dtype)
        emb = emb.reshape(B * N, self._tdim)
        # module.forward (not module.__call__): bypasses hooks, so no recursion
        out = module.forward(emb)
        return out.reshape(B, N, -1)

    def _last_mod_forward(self, vec):
        if self._tau is None:
            return self._orig_last_mod_forward(vec)
        scale = vec + self._last_mod.lin[0]
        shift = vec + self._last_mod.lin[1]
        return scale, shift


class BlockFeatureExtractor:
    """Captures hidden states from K2 SingleStreamBlocks via forward hooks.

    K2 has a single ``model.blocks`` list (no double/single split like
    FLUX.2), so layer indexing is a direct ``model.blocks[layer]`` — no
    branching logic needed. Each block returns the full ``combined``
    (img+txt+pad) tensor; K2's image-first token ordering makes the
    image-token slice a simple prefix (``output[:, :imglen, :]``).

    Blocks self-checkpoint internally, so hook outputs are differentiable
    even with gradient checkpointing enabled.
    """

    def __init__(self) -> None:
        self._handles: list = []
        self._installed_layers: set[int] = set()
        self._armed_layer: int | None = None
        self._imglen: int | None = None
        self._features: torch.Tensor | None = None

    def install(self, model, layer_indices: list[int]) -> None:
        num_blocks = len(model.blocks)
        for layer in sorted(set(layer_indices)):
            if not 0 <= layer < num_blocks:
                raise ValueError(f"feature layer {layer} out of range (model has {num_blocks} blocks)")
            handle = model.blocks[layer].register_forward_hook(self._make_hook(layer))
            self._handles.append(handle)
            self._installed_layers.add(layer)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def arm(self, layer: int, imglen: int) -> None:
        if layer not in self._installed_layers:
            raise ValueError(f"feature layer {layer} was not installed (installed: {sorted(self._installed_layers)})")
        self._armed_layer = layer
        self._imglen = imglen
        self._features = None

    def drain(self) -> torch.Tensor | None:
        features = self._features
        self._features = None
        self._armed_layer = None
        self._imglen = None
        return features

    def _make_hook(self, layer: int):
        def hook(module, inputs, output):
            if self._armed_layer == layer:
                self._features = output[:, : self._imglen, :]

        return hook


class Krea2SelfFlowNetworkTrainer(Krea2NetworkTrainer):
    """K2 + Self-Flow trainer.

    Owned state (set during the relevant lifecycle seams, used across steps):
    - ``self.rep_proj``: representation projection head (paper Eq. 6).
    - ``self.ema_lora_state``: EMA snapshot of LoRA weights (the "teacher").
    - ``self._feature_extractor``: BlockFeatureExtractor, installed in
      ``on_transformer_loaded``.
    - ``self._modulation_controller``: PerTokenModulationController, installed
      in ``on_transformer_loaded``.
    - ``self._self_flow_logs``: per-step state metrics drained by
      ``extra_step_logs``.
    - ``self._saved_student_state``: snapshot of student LoRA weights while
      EMA weights are swapped in for sampling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rep_proj: torch.nn.Module | None = None
        self.ema_lora_state: dict | None = None
        self._feature_extractor: BlockFeatureExtractor | None = None
        self._modulation_controller: PerTokenModulationController | None = None
        self._self_flow_logs: dict = {}
        self._saved_student_state: dict | None = None

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        super().handle_model_specific_args(args)
        if args.self_flow:
            if (
                args.student_feature_layer is not None
                and args.teacher_feature_layer is not None
                and args.student_feature_layer >= args.teacher_feature_layer
            ):
                raise ValueError(
                    f"--student_feature_layer ({args.student_feature_layer}) must be less than "
                    f"--teacher_feature_layer ({args.teacher_feature_layer})."
                )
            if not 0.0 <= args.mask_ratio <= 0.5:
                raise ValueError(f"--mask_ratio ({args.mask_ratio}) must be in [0, 0.5] (paper constraint R_M <= 0.5)")
            num_buckets = getattr(args, "num_timestep_buckets", None)
            if num_buckets is not None and num_buckets >= 2:
                raise ValueError(
                    f"--num_timestep_buckets ({num_buckets}) is incompatible with --self_flow: "
                    "bucketed timestep sampling forces both self-flow draws to the same bucket "
                    "value (t == s), collapsing the dual-timestep schedule to a no-op. "
                    "Unset --num_timestep_buckets when using --self_flow."
                )
            if getattr(args, "compile", False):
                logger.warning("--compile with --self_flow: forward hooks cause graph breaks; expect reduced compile benefit.")

    def extra_trainable_params(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        trainable_params: list,
    ) -> list:
        trainable_params = super().extra_trainable_params(args, accelerator, network, transformer, trainable_params)
        if not args.self_flow:
            return trainable_params

        hidden_size = accelerator.unwrap_model(transformer).config.features
        self.rep_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        if args.network_weights_proj is not None:
            proj_state = load_file(args.network_weights_proj)
            self.rep_proj.load_state_dict(proj_state)
            accelerator.print(f"loaded projection head weights from {args.network_weights_proj}")

        if trainable_params:
            trainable_params[0]["params"] = list(trainable_params[0]["params"]) + list(self.rep_proj.parameters())
        else:
            trainable_params = [{"params": list(self.rep_proj.parameters())}]
        return trainable_params

    def on_transformer_loaded(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
    ) -> None:
        super().on_transformer_loaded(args, accelerator, transformer)
        if not args.self_flow:
            return

        num_blocks = len(transformer.blocks)
        if args.student_feature_layer is None:
            args.student_feature_layer = max(0, int(num_blocks * 0.3))
        if args.teacher_feature_layer is None:
            args.teacher_feature_layer = min(num_blocks - 1, int(num_blocks * 0.7))
        if args.student_feature_layer >= args.teacher_feature_layer:
            raise ValueError(
                f"--student_feature_layer ({args.student_feature_layer}) must be less than "
                f"--teacher_feature_layer ({args.teacher_feature_layer})."
            )

        self._modulation_controller = PerTokenModulationController()
        self._modulation_controller.install(transformer)
        self._feature_extractor = BlockFeatureExtractor()
        self._feature_extractor.install(transformer, [args.student_feature_layer, args.teacher_feature_layer])
        logger.info(
            f"Self-Flow hooks installed: student_layer={args.student_feature_layer}, "
            f"teacher_layer={args.teacher_feature_layer} (of {num_blocks} blocks)"
        )

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        super().on_train_start(args, accelerator, network, transformer, optimizer)
        if not args.self_flow:
            return

        unwrapped_nw = accelerator.unwrap_model(network)
        self.ema_lora_state = {k: v.detach().clone() for k, v in unwrapped_nw.state_dict().items()}
        if args.network_weights_ema is not None:
            if args.network_weights is None:
                raise ValueError("--network_weights_ema requires --network_weights to be set")
            unwrapped_nw = accelerator.unwrap_model(network)
            info = unwrapped_nw.load_weights(args.network_weights_ema)
            accelerator.print(f"load EMA network weights from {args.network_weights_ema}: {info}")
            self.ema_lora_state = {k: v.detach().clone() for k, v in unwrapped_nw.state_dict().items()}
            unwrapped_nw.load_weights(args.network_weights)

        self.rep_proj = accelerator.prepare(self.rep_proj)

        if getattr(args, "resume", None) is not None and args.network_weights_ema is None and args.network_weights_proj is None:
            logger.warning(
                "Resuming training (--resume) with Self-Flow enabled, but neither --network_weights_ema nor "
                "--network_weights_proj was supplied: the EMA teacher will be silently re-snapshotted from the "
                "resumed student weights (degenerating L_rep until the EMA drifts again), and the projection "
                "head will be randomly re-initialized. Pass --network_weights_ema/--network_weights_proj to "
                "resume the teacher and projection head from saved companion files."
            )

        if args.self_flow_teacher_coupling_decay != "constant":
            logger.warning(
                "self_flow_teacher_coupling_decay schedules are not implemented in this extension yet; "
                "using constant coupling probability."
            )
        logger.info(
            f"Self-Flow enabled: gamma={args.self_flow_gamma}, mask_ratio={args.mask_ratio}, "
            f"ema_decay={args.ema_decay}, student_layer={args.student_feature_layer}, "
            f"teacher_layer={args.teacher_feature_layer}"
        )

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ):
        """Extends Krea2NetworkTrainer.call_dit.

        Recognised kwargs:
        - ``hidden_features`` (bool): when True, return captured features in
          ``DiTOutput.extra["features"]`` (drained from ``self._feature_extractor``).
        - ``feature_layer`` (int): which registered layer's output to return.
        - ``per_token_timesteps`` (Tensor): (B, imglen) per-image-token timestep
          map for dual-timestep conditioning. Staged into
          ``self._modulation_controller`` before the forward and cleared after.
        """
        hidden_features = kwargs.pop("hidden_features", False)
        feature_layer = kwargs.pop("feature_layer", None)
        per_token_timesteps = kwargs.pop("per_token_timesteps", None)

        if not hidden_features and per_token_timesteps is None:
            return super().call_dit(
                args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
            )

        model = accelerator.unwrap_model(transformer)
        patch = model.config.patch
        nmi = noisy_model_input.squeeze(2)
        _, _, lat_h, lat_w = nmi.shape
        imglen = (lat_h // patch) * (lat_w // patch)

        if per_token_timesteps is not None:
            vl_embed = batch["krea2_vl_embed"]
            max_len = max(x.shape[0] for x in vl_embed)
            fulllen = imglen + max_len
            padlen = (-fulllen) % 256
            N = fulllen + padlen
            B = per_token_timesteps.shape[0]
            # model scale: base call_dit divides 1D timesteps by 1000; mirror that here
            tau_img = per_token_timesteps.to(device=accelerator.device) / 1000.0
            tau_global = (timesteps.to(device=accelerator.device) / 1000.0).unsqueeze(1).expand(B, N - imglen)
            tau_full = torch.cat([tau_img, tau_global], dim=1)
            self._modulation_controller.stage(tau_full)
        if hidden_features:
            self._feature_extractor.arm(feature_layer, imglen)
        features = None
        try:
            output = super().call_dit(
                args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
            )
        finally:
            self._modulation_controller.clear()
            if hidden_features:
                features = self._feature_extractor.drain()
        if hidden_features:
            output.extra["features"] = features
        return output

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Self-Flow step replacing vanilla flow matching. See
        flux_2_train_network_self_flow.py's process_batch for the reference
        this mirrors; the only architecture-specific differences are the
        patch-token-aware apply_per_token_mask (Task 3) and the block-swap
        re-prepare call (K2's SingleStreamDiT has its own
        prepare_block_swap_before_forward, same as FLUX.2's transformer)."""
        if not args.self_flow:
            return super().process_batch(
                args,
                accelerator,
                transformer,
                network,
                batch,
                latents,
                noise,
                noise_scheduler,
                dit_dtype,
                network_dtype,
                vae,
                global_step,
            )

        # 1. two timestep draws; per-sample teacher = min, student = max (paper Sec 3.3)
        _, timesteps_a = self.get_noisy_model_input_and_timesteps(
            args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
        )
        _, timesteps_b = self.get_noisy_model_input_and_timesteps(
            args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
        )
        timesteps_teacher, timesteps_student = assign_teacher_student_timesteps(timesteps_a, timesteps_b)

        # 2. noisy inputs + per-patch-token mask (masked tokens take the cleaner teacher noise)
        noisy_input_teacher = reconstruct_noisy_input(latents, noise, timesteps_teacher)
        noisy_input_student = reconstruct_noisy_input(latents, noise, timesteps_student)
        model = accelerator.unwrap_model(transformer)
        patch = model.config.patch
        B = latents.shape[0]
        coin = torch.rand(B, device=accelerator.device) < 0.5
        effective_ratio = torch.where(
            coin,
            torch.full((B,), args.mask_ratio, device=accelerator.device),
            torch.full((B,), 1.0 - args.mask_ratio, device=accelerator.device),
        )
        noisy_input_student, mask_flat = apply_per_token_mask(
            noisy_input_student, noisy_input_teacher, effective_ratio, patch, accelerator.device
        )

        coupling_prob = args.self_flow_teacher_coupling_prob
        gate_open = coupling_prob > 0.0 and torch.rand(1).item() < coupling_prob
        effective_mismatch = args.self_flow_teacher_mismatch_ratio if gate_open else 0.0
        per_token_timesteps_student, mismatch_mask = build_per_token_timestep_map(
            timesteps_teacher, timesteps_student, mask_flat, mismatch_prob=effective_mismatch
        )

        # 3. teacher forward: EMA weights, no grad, uniform (cleaner) timestep
        unwrapped_nw = accelerator.unwrap_model(network)
        student_lora_state = {k: v.clone() for k, v in unwrapped_nw.state_dict().items()}
        unwrapped_nw.load_state_dict(self.ema_lora_state)
        try:
            with torch.no_grad():
                output = self.call_dit(
                    args,
                    accelerator,
                    transformer,
                    latents,
                    batch,
                    noise,
                    noisy_input_teacher,
                    timesteps_teacher,
                    network_dtype,
                    hidden_features=True,
                    feature_layer=args.teacher_feature_layer,
                )
                feat_teacher = output.extra.get("features")
                feat_teacher = feat_teacher.detach() if feat_teacher is not None else None
        finally:
            unwrapped_nw.load_state_dict(student_lora_state)
        # block swap ran forward-only for the teacher; re-prepare device placement
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()

        # 4. student forward: gradients flow, per-token timesteps via hooks
        output = self.call_dit(
            args,
            accelerator,
            transformer,
            latents,
            batch,
            noise,
            noisy_input_student,
            timesteps_student,
            network_dtype,
            hidden_features=True,
            feature_layer=args.student_feature_layer,
            per_token_timesteps=per_token_timesteps_student,
        )
        feat_student = output.extra.get("features")

        # 5. L_gen via the base loss (weighting from student timesteps), then L_rep
        L_gen, gen_metrics = self.compute_loss(
            args, output, timesteps_student, noise_scheduler, dit_dtype, network_dtype, global_step
        )

        if feat_student is None or feat_teacher is None:
            raise RuntimeError(
                f"Self-Flow: feature capture returned None "
                f"(student={feat_student is not None}, teacher={feat_teacher is not None}) "
                "— feature hooks misconfigured"
            )

        gamma = effective_gamma(args.self_flow_gamma, global_step, args.self_flow_gamma_warmup_steps)
        L_rep = compute_representation_loss(feat_student, feat_teacher, self.rep_proj)
        loss = L_gen + gamma * L_rep
        loss_metrics = {
            **gen_metrics,
            "loss/gen": L_gen.detach().item(),
            "loss/rep": L_rep.detach().item(),
        }

        # 6. state metrics drained later by extra_step_logs
        ema_drift = compute_ema_weight_drift(self.ema_lora_state, unwrapped_nw.state_dict())
        self._self_flow_logs = {
            "self_flow/gamma": gamma,
            "self_flow/feat_cosine_sim": -L_rep.detach().item(),
            "self_flow/timestep_student_mean": timesteps_student.float().mean().item(),
            "self_flow/timestep_teacher_mean": timesteps_teacher.float().mean().item(),
            "self_flow/timestep_diff": (timesteps_student.float().mean() - timesteps_teacher.float().mean()).item(),
            "self_flow/teacher_coupling_prob": coupling_prob,
            "self_flow/actual_mask_ratio": mask_flat.float().mean().item(),
            "self_flow/cleaner_fraction_mean": effective_ratio.mean().item(),
            "self_flow/ema_weight_drift": ema_drift.item(),
        }
        masked_count = mask_flat.sum().item()
        mismatch_count = mismatch_mask.sum().item()
        self._self_flow_logs["self_flow/mismatch_patch_count"] = mismatch_count
        self._self_flow_logs["self_flow/mismatch_patch_frac"] = mismatch_count / masked_count if masked_count > 0 else 0.0

        return loss, loss_metrics

    def on_post_optimizer_step(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        sync_gradients: bool,
        global_step: int,
    ) -> None:
        """EMA update of LoRA weights only on real optimizer steps."""
        super().on_post_optimizer_step(args, accelerator, network, transformer, sync_gradients, global_step)
        if not args.self_flow or not sync_gradients or self.ema_lora_state is None:
            return
        update_ema_weights(self.ema_lora_state, accelerator.unwrap_model(network).state_dict(), args.ema_decay)

    def on_post_save(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        ckpt_name: str,
        save_dtype,
        metadata: dict,
        force_sync_upload: bool,
    ) -> None:
        """Save EMA (teacher) and projection-head companion files."""
        super().on_post_save(args, accelerator, network, transformer, ckpt_name, save_dtype, metadata, force_sync_upload)
        if not args.self_flow:
            return

        unwrapped_nw = accelerator.unwrap_model(network)

        if self.rep_proj is not None:
            proj_ckpt_name = ckpt_name.replace(".safetensors", "-proj.safetensors")
            proj_ckpt_file = os.path.join(args.output_dir, proj_ckpt_name)
            accelerator.print(f"saving projection head: {proj_ckpt_file}")
            proj_state = {
                k: v.to(save_dtype) if save_dtype is not None else v
                for k, v in accelerator.unwrap_model(self.rep_proj).state_dict().items()
            }
            save_file(proj_state, proj_ckpt_file)
            if args.huggingface_repo_id is not None:
                huggingface_utils.upload(args, proj_ckpt_file, "/" + proj_ckpt_name, force_sync_upload=force_sync_upload)

        if self.ema_lora_state is not None:
            ema_ckpt_name = ckpt_name.replace(".safetensors", "-ema.safetensors")
            ema_ckpt_file = os.path.join(args.output_dir, ema_ckpt_name)
            accelerator.print(f"saving EMA checkpoint: {ema_ckpt_file}")
            student_state = {k: v.clone() for k, v in unwrapped_nw.state_dict().items()}
            unwrapped_nw.load_state_dict(self.ema_lora_state)
            try:
                unwrapped_nw.save_weights(ema_ckpt_file, save_dtype, metadata)
            finally:
                unwrapped_nw.load_state_dict(student_state)
            if args.huggingface_repo_id is not None:
                huggingface_utils.upload(args, ema_ckpt_file, "/" + ema_ckpt_name, force_sync_upload=force_sync_upload)

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        """Return ``ss_self_flow_*`` keys for embedding into safetensors metadata."""
        metadata = dict(super().extra_metadata(args))
        if not args.self_flow:
            return metadata
        metadata.update(
            {
                "ss_self_flow": True,
                "ss_self_flow_gamma": args.self_flow_gamma,
                "ss_self_flow_gamma_warmup_steps": args.self_flow_gamma_warmup_steps,
                "ss_self_flow_mask_ratio": args.mask_ratio,
                "ss_self_flow_ema_decay": args.ema_decay,
                "ss_self_flow_student_layer": args.student_feature_layer,
                "ss_self_flow_teacher_layer": args.teacher_feature_layer,
                "ss_self_flow_teacher_coupling_prob": args.self_flow_teacher_coupling_prob,
                "ss_self_flow_teacher_coupling_decay": args.self_flow_teacher_coupling_decay,
                "ss_self_flow_teacher_mismatch_ratio": args.self_flow_teacher_mismatch_ratio,
            }
        )
        return metadata

    def on_before_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        """Compose Krea2NetworkTrainer's RAW->Turbo base-weight swap (if
        --turbo_dit) with Self-Flow's student->EMA network-weight swap.
        These touch different objects (transformer base weights vs. LoRA
        network weights) so they don't conflict, but both must run."""
        super().on_before_sample_images(accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype)
        if not args.self_flow or self.ema_lora_state is None:
            return
        network = accelerator.unwrap_model(network)
        self._saved_student_state = {k: v.clone() for k, v in network.state_dict().items()}
        network.load_state_dict(self.ema_lora_state)

    def on_after_sample_images(
        self, accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype
    ) -> None:
        """Reverse in LIFO order: restore student LoRA weights first, then
        let Krea2NetworkTrainer restore RAW base weights."""
        if args.self_flow and self.ema_lora_state is not None and self._saved_student_state is not None:
            network = accelerator.unwrap_model(network)
            network.load_state_dict(self._saved_student_state)
            self._saved_student_state = None
        super().on_after_sample_images(accelerator, args, epoch, steps, vae, transformer, network, sample_parameters, dit_dtype)

    def extra_step_logs(self, args: argparse.Namespace, logs: dict) -> dict:
        """Drain per-step Self-Flow metrics into the trainer's log payload."""
        if not args.self_flow:
            return {}
        return dict(self._self_flow_logs)


def self_flow_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Self-Flow-specific CLI arguments. Mirrors flux_2_train_network_self_flow.py's additions."""
    parser.add_argument(
        "--self_flow",
        action="store_true",
        help="Enable Self-Flow training (dual-timestep scheduling + representation alignment).",
    )
    parser.add_argument(
        "--self_flow_gamma",
        type=float,
        default=0.8,
        help="Weight for representation alignment loss L_rep (paper Eq. 7). 0 disables L_rep.",
    )
    parser.add_argument(
        "--self_flow_gamma_warmup_steps",
        type=int,
        default=0,
        help="Linearly ramp gamma from 0 to --self_flow_gamma over this many steps. 0 disables warmup.",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.25,
        help="Token mask ratio for dual-timestep scheduling (paper Eq. 4, R_M <= 0.5).",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay for the Self-Flow teacher LoRA weights.",
    )
    parser.add_argument(
        "--student_feature_layer",
        type=int,
        default=None,
        help="Global block index for student feature extraction (l in paper, must be < teacher_feature_layer). Recommended ~0.3 * num_blocks.",
    )
    parser.add_argument(
        "--teacher_feature_layer",
        type=int,
        default=None,
        help="Global block index for teacher feature extraction (k in paper, must be > student_feature_layer). Recommended ~0.7 * num_blocks.",
    )
    parser.add_argument(
        "--self_flow_teacher_coupling_prob",
        type=float,
        default=0.0,
        help="Per-step gate probability for applying timestep mismatch on masked patches. 0 disables mismatch.",
    )
    parser.add_argument(
        "--self_flow_teacher_coupling_decay",
        type=str,
        default="constant",
        choices=["constant", "cosine", "linear", "rex"],
        help="Decay schedule for --self_flow_teacher_coupling_prob.",
    )
    parser.add_argument(
        "--self_flow_teacher_coupling_decay_steps",
        type=int,
        default=None,
        help="Steps over which to decay coupling prob to 0. Defaults to max_train_steps.",
    )
    parser.add_argument(
        "--self_flow_teacher_mismatch_ratio",
        type=float,
        default=1.0,
        help="When the coupling gate fires, fraction of masked patches receiving the timestep mismatch.",
    )
    parser.add_argument(
        "--network_weights_ema",
        type=str,
        default=None,
        help="Pretrained EMA (teacher) weights for resumption. Requires --network_weights.",
    )
    parser.add_argument(
        "--network_weights_proj",
        type=str,
        default=None,
        help="Pretrained projection head weights for resumption.",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)
    parser = self_flow_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2SelfFlowNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
