# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Training extensions built on top of [musubi-tuner](https://github.com/kohya-ss/musubi-tuner), consumed as a
library dependency (not a fork). Each extension attaches to musubi-tuner's `NetworkTrainer` extension seams
(`call_dit`, `compute_loss`, `process_batch`, `on_train_start`, etc.) via subclassing or mixins — no edits to
musubi-tuner's own model/trainer files.

Extensions:
- `boo_musubi_tuner.self_flow` — Self-Flow trainers implementing arXiv:2603.06507 (dual-timestep scheduling +
  representation-alignment distillation) via forward hooks, for FLUX.2 (`flux_2_train_network_self_flow.py`) and
  Krea 2 (`krea2_train_network_self_flow.py`) — see [docs/self-flow.md](docs/self-flow.md) for details.
- `boo_musubi_tuner.wavelet_loss` — pluggable frequency-domain auxiliary loss FLUX.2 trainer
  (`flux_2_train_network_wavelet_loss.py`), optional dependency on the separate `wavelet_loss` package.
- `boo_musubi_tuner.explorative_modeling` — best-of-K candidate selection mixin (`explorative_modeling.py`,
  `ExplorativeModelingMixin`) plus FLUX.2 and Krea2 trainer entry points.

## Adding new features

See [docs/agents/features.md](docs/agents/features.md) for the checklist of files to create/update when adding
a new extension or feature.

## Setup

`musubi-tuner` is wired via `[tool.uv.sources]` in `pyproject.toml`, pointed at an editable local checkout
(`../others/musubi-tuner` by default). There's a commented-out pinned-git alternative for portable/CI setups.

`explorative_modeling` requires a musubi-tuner core change: `trainer_base.py`'s `compute_loss` needs a
`reduction="none"` mode (plus a broadcast-shape fix) for best-of-K scoring — [PR #1042](https://github.com/kohya-ss/musubi-tuner/pull/1042)
upstream (open, not merged). It's available on the `compute-loss-reduction` branch of the rockerBOO fork.
Whichever musubi-tuner source is active must have that branch checked out or merged in.

```bash
uv sync
```

## Commands

```bash
uv run pytest                                    # full test suite
uv run pytest tests/self_flow/                   # one extension's tests
uv run pytest tests/self_flow/test_self_flow_loss.py::test_name   # single test
uv run ruff check .                              # lint (line-length 132, configured in pyproject.toml)
uv run ruff format .                             # format
```

Always run `uv run ruff check .` and `uv run ruff format .` before committing — there's no pre-commit hook enforcing this, so it's on the person/agent making the commit.

Tests are CPU-only and use tiny Flux2 configs (see `tests/self_flow/conftest.py`'s `tiny_params`/`tiny_model`
fixtures — `hidden_size=16`, 2+2 blocks) rather than real weights, so the suite runs fast without a GPU.

For running a trainer, see README.md's "Running trainers" section.

## Architecture

### Extension pattern

Every trainer subclasses (or mixes into) a musubi-tuner `*NetworkTrainer` base class (e.g. `Flux2NetworkTrainer`)
and only overrides the extension seams musubi-tuner exposes:

- `handle_model_specific_args` — CLI arg validation, run early (before any model-version logic).
- `extra_trainable_params` — add extra `nn.Module` params (e.g. a projection head) into the optimizer's first
  param group (needed because optimizers like Prodigy don't support per-group LRs).
- `on_transformer_loaded` — register forward hooks on the *raw* transformer, before `accelerator.prepare` /
  block-swap rewrapping, so hook-captured tensors align with user-supplied unwrapped block indices.
- `on_train_start` — finish setup once everything is on-device (e.g. `accelerator.prepare` extra modules,
  snapshot EMA state).
- `call_dit` — wraps the base `call_dit`; extensions stash extra info in `DiTOutput.extra` (a dict) or accept
  extra kwargs (e.g. `hidden_features`, `per_token_timesteps`) that are popped before delegating to `super()`.
- `process_batch` — the main per-step override point; when the extension's flag is off, every override falls
  straight through to `super().process_batch(...)` unchanged, so these trainers work as plain FLUX.2/Krea2
  trainers when their feature flag is disabled.
- `on_post_optimizer_step`, `on_before_sample_images`/`on_after_sample_images`, `on_post_save`, `extra_metadata`,
  `extra_step_logs` — lifecycle hooks for EMA updates, weight swapping for sampling, companion checkpoint files,
  and safetensors metadata / W&B-style step logging.

Every extension follows the "vanilla passthrough when flag is off" rule: check `args.<flag>` first thing in each
overridden method and delegate to `super()` if it's unset. This lets the same trainer class serve as both the
plain base trainer and the extended one.

**Per-token conditioning without model edits**: Self-Flow needs different timesteps for masked vs. unmasked
tokens, and both self_flow and self_flow's feature distillation need internal block hidden states — accomplished
entirely with `register_forward_hook`/`register_forward_pre_hook` on the unmodified Flux2 model
(`PerTokenModulationController`, `BlockFeatureExtractor` in `flux_2_train_network_self_flow.py`). Flux2's
`Modulation`/`LastLayer` broadcast 3D modulation vectors natively, so rerouting `time_in` output through a hook
to produce a `(B, N, D)` per-token vector instead of `(B, D)` works with zero model changes. When hooks aren't
staged, they're pass-throughs — the model is bit-identical to vanilla.

**`ExplorativeModelingMixin`** (`explorative_modeling.py`) is architecture-generic: it only overrides
`process_batch`/`extra_metadata`/`on_train_start`, and is meant to be mixed in ahead of a concrete trainer
(`class Flux2XMNetworkTrainer(ExplorativeModelingMixin, Flux2NetworkTrainer)`). It has two documented
composability gaps (see the class docstring): it reimplements the noising math for candidates 2..K rather than
re-calling `get_noisy_model_input_and_timesteps` (breaks for architectures overriding that method with
non-trivial resampling), and it always calls `compute_loss(..., reduction="none")` (breaks for architectures
whose `compute_loss` override doesn't accept `reduction`). Check these before mixing XM into a new architecture.

For module/test layout conventions, see [docs/agents/features.md](docs/agents/features.md).

### Internal API stability

Both `self_flow` and `explorative_modeling` docstrings explicitly flag themselves as "internal extension
point — no API stability guarantees" against musubi-tuner's seams; expect breakage on musubi-tuner updates.
