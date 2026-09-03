# Extension implementation internals

Model/feature-specific implementation notes that agents need when touching a specific extension's internals.
For end-user docs (flags, usage), see the corresponding `docs/<extension>.md`. For the general extension-seam
mechanics all extensions follow, see CLAUDE.md's "Extension pattern" section.

## self_flow: per-token conditioning without model edits

Self-Flow needs different timesteps for masked vs. unmasked tokens, and both Self-Flow and its feature
distillation need internal block hidden states — accomplished entirely with `register_forward_hook`/
`register_forward_pre_hook` on the unmodified Flux2 model (`PerTokenModulationController`,
`BlockFeatureExtractor` in `flux_2_train_network_self_flow.py`). Flux2's `Modulation`/`LastLayer` broadcast 3D
modulation vectors natively, so rerouting `time_in` output through a hook to produce a `(B, N, D)` per-token
vector instead of `(B, D)` works with zero model changes. When hooks aren't staged, they're pass-throughs — the
model is bit-identical to vanilla.

Krea 2's final transformer layer can't take a per-token timestep via a forward hook alone (its modulation only
broadcasts a token axis of size 1 or 2 and raises before a hook could intervene), so the Krea 2 trainer patches
that one layer's `forward` at the instance level instead. Every other layer, and the whole FLUX.2 trainer, use
plain forward hooks with no model code changes.

## explorative_modeling: composability gaps

`ExplorativeModelingMixin` (`explorative_modeling.py`) is architecture-generic: it only overrides
`process_batch`/`extra_metadata`/`on_train_start`, and is meant to be mixed in ahead of a concrete trainer
(`class Flux2XMNetworkTrainer(ExplorativeModelingMixin, Flux2NetworkTrainer)`). It has two documented
composability gaps (see the class docstring):

- It reimplements the noising math for candidates 2..K rather than re-calling
  `get_noisy_model_input_and_timesteps` — breaks for architectures overriding that method with non-trivial
  resampling (e.g. Wan's high/low-noise resampling).
- It always calls `compute_loss(..., reduction="none")` — breaks for architectures whose `compute_loss` override
  doesn't accept `reduction` (e.g. Ideogram 4, HiDream-O1).

Check these before mixing XM into a new architecture. It also needs a musubi-tuner core patch — see
[docs/explorative-modeling.md#requirements](../explorative-modeling.md#requirements).

## tdm_distill: single-transformer three-role LoRA swap

`LoraRoleSwitcher` (`tdm_distill.py`) extends `self_flow`'s EMA-teacher `state_dict` swap trick from
two states (current/EMA) to three roles on one physical LoRA network: teacher (multiplier forced to
0, zero weight-copy cost), student, and fake-score (both independently trainable, swapped via
`load_state_dict` immediately before each forward). This avoids needing a second resident full
transformer for the teacher — K2 raw's native behavior with the LoRA's multiplier at 0 already *is*
the teacher, since the base weights are always K2 raw's.

`process_batch` is data-free: it discards the dataloader's cached `latents`/`noise` entirely and
only uses `batch["krea2_vl_embed"]` (the caption). This is the one place in this repo where a
`process_batch` override doesn't just add a loss term on top of the real training signal — it
replaces the training signal outright when `--tdm_distill` is on.

The fake-score critic is trained by a second, extension-owned optimizer (`self._fake_score_optimizer`,
built via a second call to `self.get_optimizer(...)` inside `on_train_start`, `accelerate.prepare`d
separately), backward/stepped manually inside `process_batch` — the only extension in this repo with
more than one live optimizer. `process_batch` must always leave the LoRA network on its **student**
role when it returns, since the base trainer's outer loop calls `accelerator.backward`/
`optimizer.step()` on whatever `network`'s current live weights are. `on_post_save` reasserts this
(`switcher.use_student()`), but note that hook fires *after* the checkpoint file is already written,
so it is post-save state hygiene for whatever runs next — not a save-time safety net. Keeping the
fake-score critic (a training-only auxiliary network) out of saved checkpoints depends entirely on
`process_batch` holding the invariant above.

**The DINOv3 diversity term is a real, fully differentiable training signal**, not a passive metric
computed for logging. `Dinov3ImageEmbedder.embed_differentiable` (`tdm_distill.py`) deliberately
avoids the class's other method, `embed()` — that one is `@torch.no_grad()` and takes PIL/numpy
input via `self.processor`, both of which sever the autograd graph. `embed_differentiable` instead
takes an already-decoded `(N, C, H, W)` tensor and does resize/normalize with plain differentiable
tensor ops, keeping the graph intact from the DINOv3 CLS-token embedding all the way back through
the VAE decode and the student rollout to the LoRA weights. This required two things elsewhere in
`process_batch`: (1) the group-diversity rollout is called with `grad_from_step=0` (every step
grad-enabled), unlike the main TDM rollout, which only needs its final step's graph and runs the
rest under `no_grad` for efficiency; and (2) the VAE decode for this rollout runs without
`torch.no_grad()`, with the VAE's own parameters frozen (`requires_grad_(False)`, lazily on first
use) rather than the decode itself being no-grad, so gradient reaches the input latents while the
VAE's weights accumulate no `.grad`.

That grad-enabled VAE decode has a consequence for VAE placement: the base trainer normally keeps
the VAE on CPU between uses and moves it to the training device and back within a single call (see
`load_vae`/`on_before_sample_images`). That round-trip pattern is unsafe here — `nn.Module.to()`
rebinds parameter storage in place, and the diversity decode's output feeds a live autograd graph
that `accelerator.backward()` (called by the outer training loop, *after* `process_batch` returns)
still needs to walk. Moving the VAE back to CPU inside `process_batch` would corrupt that graph with
a device mismatch mid-backward. Instead `process_batch` moves the VAE to the training device and
leaves it there, recording the fact via `self._vae_ref`/`self._vae_needs_cpu_return`; a new
`on_post_optimizer_step` override moves it back to CPU, timed to run strictly after that step's
`accelerator.backward()` and optimizer step have both completed and the graph is dead. This is the
only extension in this repo where a hook other than `process_batch` itself has to manage VAE
device placement.
