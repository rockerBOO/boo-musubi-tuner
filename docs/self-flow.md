# Self-Flow

Self-distillation training for flow-matching diffusion transformers: a deeper, cleaner-timestep "teacher" view
of the model distills into a shallower, noisier-timestep "student" view during the same training step — no
external teacher model or extra training run required.

- Source: [bfl.ai/research/self-flow](https://bfl.ai/research/self-flow)
- Paper: Chefer, Esser, Lorenz, Podell, Raja, Tong, Torralba, Rombach. *Self-Supervised Flow Matching for
  Scalable Multi-Modal Synthesis*. [arXiv:2603.06507](https://arxiv.org/abs/2603.06507)

## Trainers

- `src/boo_musubi_tuner/self_flow/flux_2_train_network_self_flow.py` — FLUX.2
- `src/boo_musubi_tuner/self_flow/krea2_train_network_self_flow.py` — Krea 2

Each is a drop-in replacement for its base musubi-tuner script — every normal FLUX.2/Krea2 training flag still
works. Self-Flow itself is opt-in behind `--self_flow`.

Krea 2's final transformer layer can't take a per-token timestep via a forward hook alone — its modulation only
broadcasts a token axis of size 1 or 2 and raises before a hook could intervene — so the K2 trainer patches that
one layer's `forward` at the instance level instead. Every other layer, and the whole FLUX.2 trainer, use plain
forward hooks with no model code changes.

## Usage

```bash
accelerate launch src/boo_musubi_tuner/self_flow/flux_2_train_network_self_flow.py \
    --self_flow \
    --mask_ratio 0.25 \
    --self_flow_gamma 0.8 \
    --ema_decay 0.999 \
    <...normal FLUX.2 training args...>
```

## Key flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--self_flow` | off | enable Self-Flow training |
| `--mask_ratio` | `0.25` | fraction of tokens the student takes from the teacher's cleaner noise (paper `R_M`, must be `<= 0.5`) |
| `--self_flow_gamma` | `0.8` | weight of the representation-alignment loss `L_rep` |
| `--self_flow_gamma_warmup_steps` | `0` | linearly ramp `gamma` from 0 over this many steps |
| `--ema_decay` | `0.999` | EMA decay rate for the teacher weights |
| `--student_feature_layer` / `--teacher_feature_layer` | auto (~30%/~70% of blocks) | which transformer blocks feed `L_rep` |
| `--self_flow_teacher_coupling_prob` | `0.0` | per-step gate probability for deliberately mismatching a masked token's timestep; `0.0` disables it (see Limitations) |
| `--self_flow_teacher_mismatch_ratio` | `1.0` | when the coupling gate fires, fraction of masked tokens that get the mismatch |
| `--network_weights_ema` / `--network_weights_proj` | none | resume the EMA teacher / projection head from a previous run's companion checkpoints |

Training saves two companion files alongside each checkpoint: `<name>-ema.safetensors` (the distilled teacher
weights — usually the better checkpoint to actually use) and `<name>-proj.safetensors` (the projection head, only
needed to resume training).

## Limitations

- The teacher-coupling mismatch mechanism is off by default (`--self_flow_teacher_coupling_prob` `0.0`); when
  enabled, its decay schedules (`--self_flow_teacher_coupling_decay`) accept `cosine`/`linear`/`rex` but only
  `constant` is implemented so far.
- Patch-locality mask modes from the paper are not ported.
- FLUX.2 control images (`latents_control_*`) are not supported with `--self_flow`.
