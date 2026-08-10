# boo-musubi-tuner

Training extensions for [musubi-tuner](https://github.com/kohya-ss/musubi-tuner).

Repo: [github.com/rockerBOO/boo-musubi-tuner](https://github.com/rockerBOO/boo-musubi-tuner)

## Extensions

- [Self-Flow](docs/self-flow.md) — self-distillation flow-matching training for FLUX.2 and Krea 2
- [Explorative Modeling](docs/explorative-modeling.md) — best-of-K candidate training for FLUX.2 and Krea 2
- [Wavelet Loss](docs/wavelet-loss.md) — frequency-domain auxiliary loss for FLUX.2

## Setup

`musubi-tuner` is a required dependency; `pyproject.toml` needs to point it at a musubi-tuner checkout (local
path or git) before this will install. See the `[tool.uv.sources]` section in `pyproject.toml` for both options.

```bash
uv sync
```

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
