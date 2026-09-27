# Loss Context

Plain FLUX.2 / Krea 2 training, except that each step also passes `noisy_model_input`, `latents` and `noise`
to the loss. The loss itself still comes from musubi-tuner's `--loss_fn` / `--loss_fn_args`. Use these
trainers with a `--loss_fn` that needs the model's clean-latent estimate
`x0_hat = noisy_model_input - sigma * pred`, for example wavelet-loss's `rectified_flow=True`,
`energy_beta` (local high-frequency energy matching against flat-region mottle) or `mottle_metrics`.

- Source: [github.com/rockerBOO/wavelet-loss](https://github.com/rockerBOO/wavelet-loss) (`wavelet_loss.musubi` adapters)

## Trainers

- `src/boo_musubi_tuner/loss_context/flux_2_train_network_loss_context.py`: FLUX.2
- `src/boo_musubi_tuner/loss_context/krea2_train_network_loss_context.py`: Krea 2

## Usage

```bash
accelerate launch src/boo_musubi_tuner/loss_context/flux_2_train_network_loss_context.py \
    --loss_fn wavelet_loss.musubi.WaveletPlusX0Huber \
    --loss_fn_args alpha=1.0 "loss_type='x0_huber'" energy_beta=0.1 mottle_metrics=True \
        "transform_type='swt'" "wavelet='sym7'" level=2 \
    <...normal FLUX.2 training args...>
```

## Flags

None. The trainers add no CLI flags; they accept the base trainer's arguments, including `--loss_fn` /
`--loss_fn_args`.

Stashed `DiTOutput.extra` keys (references to the tensors `call_dit` received, not copies):

| key | meaning |
|---|---|
| `noisy_model_input` | `(1 - sigma) * latents + sigma * noise` |
| `latents` | clean latents |
| `noise` | sampled noise |

## Limitations

- There is no feature flag. Stashing references costs nothing, and a `--loss_fn` that ignores `extra` behaves
  exactly as on the base trainer.
- Doesn't compose with Explorative Modeling yet. XM scores candidates with `compute_loss(reduction="none")`,
  and the wavelet-loss adapters only return a scalar.
