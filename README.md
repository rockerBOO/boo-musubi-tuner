# boo-musubi-tuner

Training extensions for [musubi-tuner](https://github.com/kohya-ss/musubi-tuner).

Repo: [github.com/rockerBOO/boo-musubi-tuner](https://github.com/rockerBOO/boo-musubi-tuner)

## Extensions

- [Self-Flow](docs/self-flow.md) — self-distillation flow-matching training for FLUX.2 and Krea 2
- [Explorative Modeling](docs/explorative-modeling.md) — best-of-K candidate training for FLUX.2 and Krea 2
- [Wavelet Loss](docs/wavelet-loss.md) — frequency-domain auxiliary loss for FLUX.2
- [TDM Diversity Distillation](docs/tdm-distill.md) — experimental K2 guidance+step distillation (TDM) with a DINOv3 diversity term folded into the student loss
- [DINOv3 Standalone Diversity Fine-Tuning](docs/dino_diversity.md) — experimental K2 diversity-only fine-tuning, decoupled from TDM

## Setup

```bash
git clone https://github.com/rockerBOO/boo-musubi-tuner.git
cd boo-musubi-tuner
uv sync --extra cu124   # or cu128 / cu130 / cu132 / cpu, matching your CUDA driver
```

Plain `uv sync` (no `--extra`) does not error, but resolves `torch`/`torchvision` from PyPI's default
index without any of the CUDA-specific wheels — you get a CPU-only build silently. Training will then run
on CPU with no error or warning, just extremely slowly. Pick the `cuXXX` extra matching your installed
CUDA driver version, or `cpu` if you don't have a GPU.

`musubi-tuner` is pulled in as a dependency, pinned to a branch on the
[rockerBOO fork](https://github.com/rockerBOO/musubi-tuner) (needed for Explorative Modeling's
`compute_loss(reduction="none")` support — see
[docs/explorative-modeling.md](docs/explorative-modeling.md#requirements)).

## Running trainers

Each trainer is a drop-in replacement for its base musubi-tuner script — see each extension's doc for its
specific flags.

```bash
accelerate launch src/boo_musubi_tuner/self_flow/flux_2_train_network_self_flow.py ...
accelerate launch src/boo_musubi_tuner/self_flow/krea2_train_network_self_flow.py ...
accelerate launch src/boo_musubi_tuner/wavelet_loss/flux_2_train_network_wavelet_loss.py ...
accelerate launch src/boo_musubi_tuner/explorative_modeling/flux_2_train_network_xm.py ...
accelerate launch src/boo_musubi_tuner/explorative_modeling/krea2_train_network_xm.py ...
```

Explorative Modeling additionally requires a musubi-tuner core patch — see
[docs/explorative-modeling.md](docs/explorative-modeling.md#requirements).

## Development

```bash
uv run pytest                                    # full test suite
uv run pytest tests/self_flow/                   # one extension's tests
uv run pytest tests/self_flow/test_self_flow_loss.py::test_name   # single test
uv run ruff check .                              # lint (line-length 132, configured in pyproject.toml)
uv run ruff format .                             # format
```

Always run `uv run ruff check .` and `uv run ruff format .` before committing — there's no pre-commit hook
enforcing this, so it's on the person/agent making the commit.
