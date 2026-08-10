# Explorative Modeling (XM)

Best-of-K training: for each training example, sample K noise candidates, score all K, and backward only the
lowest-loss candidate. Trades extra compute per step for better sample/FLOP efficiency and models that commit to
a single mode instead of averaging over several.

- Source: [explorative-modeling.github.io](https://explorative-modeling.github.io/)
- Paper: Gladstone, Ji, Du. *Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation*.
  [arXiv:2607.27372](https://arxiv.org/abs/2607.27372)

## Trainers

- `src/boo_musubi_tuner/explorative_modeling/flux_2_train_network_xm.py` — FLUX.2
- `src/boo_musubi_tuner/explorative_modeling/krea2_train_network_xm.py` — Krea 2

Each is a drop-in replacement for its base musubi-tuner script. XM itself is opt-in behind
`--explorative_modeling`.

## Requirements

This extension needs a musubi-tuner core change: `trainer_base.py`'s `compute_loss` must support a
`reduction="none"` mode (plus a broadcast-shape fix) for scoring K candidates per example. That's
[PR #1042](https://github.com/kohya-ss/musubi-tuner/pull/1042) upstream (open, not yet merged to `main`), and is
available on the `compute-loss-reduction` branch of the [rockerBOO fork](https://github.com/rockerBOO/musubi-tuner).
Whichever musubi-tuner source `pyproject.toml` points at needs that branch checked out or merged in until
PR #1042 lands upstream.

## Usage

```bash
accelerate launch src/boo_musubi_tuner/explorative_modeling/flux_2_train_network_xm.py \
    --explorative_modeling \
    --explorative_modeling_k 4 \
    <...normal FLUX.2 training args...>
```

## Key flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--explorative_modeling` | off | enable best-of-K training |
| `--explorative_modeling_k` | `4` | number of noise candidates per example |
| `--explorative_modeling_memory_efficient` | on | score all K candidates under `no_grad`, then run one more forward on the per-example winner for the actual backward pass (near-constant memory regardless of K). Pass `--no-explorative_modeling_memory_efficient` to instead keep all K forward graphs live and backward through the gathered winner directly (matches the paper's literal pseudocode, ~Kx peak activation memory) |

## Compatibility

XM works with any musubi-tuner architecture trainer whose `compute_loss` accepts a `reduction` argument and
whose `get_noisy_model_input_and_timesteps` does plain flow-matching noising. It is not compatible out of the
box with architectures that do non-trivial resampling in `get_noisy_model_input_and_timesteps` (e.g. Wan's
high/low-noise resampling) or that override `compute_loss` without a `reduction` parameter (e.g. Ideogram 4,
HiDream-O1).
