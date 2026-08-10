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
