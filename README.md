# boo-musubi-tuner

Training extensions built on top of [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) as a library
dependency, rather than as forked branches of musubi-tuner itself.

## Extensions

- `boo_musubi_tuner.self_flow` — Self-Flow FLUX.2 trainer (`flux_2_train_network_self_flow.py`)
- `boo_musubi_tuner.wavelet_loss` — pluggable wavelet loss FLUX.2 trainer (`flux_2_train_network_wavelet_loss.py`)
- `boo_musubi_tuner.explorative_modeling` — best-of-K candidate selection mixin (`explorative_modeling.py`) plus
  FLUX.2 and Krea2 trainer entry points

## Setup

`musubi-tuner` is wired as a dependency via `[tool.uv.sources]` in `pyproject.toml`, pointed at an editable
local checkout (`../others/musubi-tuner` by default). Swap to the commented-out pinned git source for a
portable/CI setup once the extensions' prerequisite core changes have landed there.

`explorative_modeling` depends on a musubi-tuner core change: `trainer_base.py`'s `compute_loss` needs a
`reduction="none"` mode (plus a broadcast-shape fix) for best-of-K candidate scoring. That's
[PR #1042](https://github.com/kohya-ss/musubi-tuner/pull/1042) upstream — open, not yet merged to `main`. It's
available now on the `compute-loss-reduction` branch of the rockerBOO fork. Make sure whichever musubi-tuner
source this repo points at (the local editable checkout or the pinned git dependency) has that branch checked
out or merged in, until #1042 lands on upstream `main`.

```bash
uv sync
uv run pytest
```

## Running trainers

Each trainer is run the same way as musubi-tuner's own scripts, e.g.:

```bash
python src/boo_musubi_tuner/self_flow/flux_2_train_network_self_flow.py ...
python src/boo_musubi_tuner/wavelet_loss/flux_2_train_network_wavelet_loss.py ...
python src/boo_musubi_tuner/explorative_modeling/flux_2_train_network_xm.py ...
python src/boo_musubi_tuner/explorative_modeling/krea2_train_network_xm.py ...
```
