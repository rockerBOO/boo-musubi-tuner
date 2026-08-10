# Wavelet Loss

A frequency-domain auxiliary loss that penalises structure in the wavelet decomposition of the model's
predicted vs. target output, on top of the standard flow-matching MSE loss. Useful for sharpening
high-frequency detail that plain MSE tends to under-weight.

- Source: [github.com/rockerBOO/wavelet-loss](https://github.com/rockerBOO/wavelet-loss)

The wavelet transform itself lives in the separate `wavelet-loss` package, an optional dependency:

```bash
pip install wavelet-loss
```

## Trainer

- `src/boo_musubi_tuner/wavelet_loss/flux_2_train_network_wavelet_loss.py` — FLUX.2

A drop-in replacement for the base FLUX.2 musubi-tuner script. The wavelet term is opt-in behind
`--wavelet_loss`.

## Usage

```bash
accelerate launch src/boo_musubi_tuner/wavelet_loss/flux_2_train_network_wavelet_loss.py \
    --wavelet_loss \
    --wavelet_loss_alpha 0.1 \
    --wavelet_loss_transform swt \
    --wavelet_loss_level 2 \
    <...normal FLUX.2 training args...>
```

Total loss is `mse.mean() + alpha * wavelet_loss`.

## Key flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--wavelet_loss` | off | enable the wavelet auxiliary loss |
| `--wavelet_loss_alpha` | `0.1` | weight of the wavelet term |
| `--wavelet_loss_transform` | `swt` | `dwt` (discrete), `swt` (stationary), or `qwt` (quaternion) |
| `--wavelet_loss_wavelet` | `sym7` | wavelet family, e.g. `sym7`, `db4` |
| `--wavelet_loss_level` | `1` | decomposition levels; higher levels add finer detail |
| `--wavelet_loss_rectified_flow` | on | run the wavelet term on the reconstructed clean latents (x0) rather than raw velocity space; pass `--no-wavelet_loss_rectified_flow` for the raw-velocity (AWWL-style) variant |
| `--wavelet_loss_timestep_cutoff` / `--wavelet_loss_timestep_transition_width` | `0.7` / `0.4` | fade the wavelet term out at high noise levels |
| `--wavelet_loss_band_weights` / `--wavelet_loss_band_level_weights` | none | per-band or per-band-per-level weighting, e.g. `ll=0.1,lh=0.01,hl=0.01,hh=0.05` |
| `--wavelet_loss_type` | `--loss_type` | loss function for wavelet bands: `l1`/`mae`, `huber`/`smooth_l1`, or MSE (default) |
| `--wavelet_loss_metrics` | off | log detailed per-band wavelet metrics each step (adds overhead) |

## Limitations

- FLUX.2 only — no Krea 2 trainer for this extension yet.
- Requires the separate `wavelet-loss` package; passing `--wavelet_loss` without it installed raises an
  `ImportError` at startup.
- Timesteps are expected on the 1..1000 footing (`--wavelet_loss_max_timestep`, default `1000.0`); values outside
  `[0, max]` raise an error.
