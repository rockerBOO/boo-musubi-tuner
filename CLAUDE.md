# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Training extensions built on top of [musubi-tuner](https://github.com/kohya-ss/musubi-tuner), consumed as a
library dependency (not a fork). Each extension attaches to musubi-tuner's `NetworkTrainer` extension seams
(`call_dit`, `compute_loss`, `process_batch`, `on_train_start`, etc.) via subclassing or mixins — no edits to
musubi-tuner's own model/trainer files.

See README.md for the current list of extensions and links to their docs.

## Adding new features

See [docs/agents/features.md](docs/agents/features.md) for the checklist of files to create/update when adding
a new extension or feature.

## Setup

`musubi-tuner` is wired via `[tool.uv.sources]` in `pyproject.toml`. Some extensions require a specific
musubi-tuner branch or core patch to run — check that extension's doc under `docs/` (linked from README.md)
before running it.

```bash
uv sync --extra cu124   # or cu128 / cu130 / cu132 / cpu, matching the CUDA driver
```

Plain `uv sync` with no `--extra` resolves a CPU-only `torch` silently (no error) — training then runs on
CPU with no warning, just very slowly. Always pass a `--extra` matching the target machine's CUDA driver
(or `cpu` for CPU-only machines).

## Commands

```bash
uv run pytest                                    # full test suite
uv run pytest tests/<extension>/                 # one extension's tests
uv run ruff check .                              # lint (line-length 132, configured in pyproject.toml)
uv run ruff format .                             # format
```

Always run `uv run ruff check .` and `uv run ruff format .` before committing — there's no pre-commit hook enforcing this, so it's on the person/agent making the commit.

Tests are CPU-only and use tiny model configs (no real weights), so the suite runs fast without a GPU.

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
  straight through to `super().process_batch(...)` unchanged, so these trainers work as plain base trainers when
  their feature flag is disabled.
- `on_post_optimizer_step`, `on_before_sample_images`/`on_after_sample_images`, `on_post_save`, `extra_metadata`,
  `extra_step_logs` — lifecycle hooks for EMA updates, weight swapping for sampling, companion checkpoint files,
  and safetensors metadata / W&B-style step logging.

Every extension follows the "vanilla passthrough when flag is off" rule: check `args.<flag>` first thing in each
overridden method and delegate to `super()` if it's unset. This lets the same trainer class serve as both the
plain base trainer and the extended one.

For module/test layout conventions, see [docs/agents/features.md](docs/agents/features.md). For
implementation-level internals of a specific extension (hook mechanics, composability gaps, etc.), see
[docs/agents/extension-internals.md](docs/agents/extension-internals.md).

### Internal API stability

Extensions' docstrings explicitly flag themselves as "internal extension point — no API stability guarantees"
against musubi-tuner's seams; expect breakage on musubi-tuner updates.
