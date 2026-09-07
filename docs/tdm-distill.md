# TDM Diversity Distillation (K2)

Experimental Krea 2 (K2) extension implementing Trajectory Distribution Matching (TDM) for
guidance+step distillation, with a DINOv3 group-diversity term folded into the student's loss to
counteract the diversity loss guidance distillation otherwise causes. Warm-starts the student and
an auxiliary fake-score critic from an existing K2 Turbo LoRA rather than training from scratch,
matching the frozen K2 raw model's score at a sampled point along the student's own trajectory
(the teacher itself is never rolled out — it is evaluated once, or twice under CFG, per training
step, at a single `tau`).

**Source material**: [Krea 2 technical report](https://www.krea.ai/blog/krea-2-technical-report);
[Trajectory Distribution Matching (TDM), arXiv:2503.06674](https://arxiv.org/abs/2503.06674).

**Trainer**: `src/boo_musubi_tuner/tdm_distill/krea2_train_network_tdm_distill.py`
(`Krea2TdmDistillNetworkTrainer`).

## Usage

```bash
accelerate launch src/boo_musubi_tuner/tdm_distill/krea2_train_network_tdm_distill.py \
  --dit /path/to/krea2_raw.safetensors \
  --vae /path/to/qwen_image_vae.safetensors \
  --text_encoder /path/to/qwen3_vl_text_encoder.safetensors \
  --sample_prompts /path/to/prompts.txt \
  --tdm_distill \
  --tdm_turbo_lora_init /mnt/900/lora/krea2/krea2_turbo_lora_rank_64_bf16.safetensors \
  --tdm_step_counts 1,2,4,8 \
  --tdm_diversity_weight 0.1 \
  --tdm_diversity_group_size 4 \
  --tdm_guidance_scale 3.5 \
  ... (usual musubi-tuner LoRA training flags)
```

`--sample_prompts` is required whenever `--tdm_diversity_weight > 0` (the base trainer only loads
the VAE when sampling is configured, and the diversity term needs the VAE to decode latents to
pixels) or whenever `--tdm_guidance_scale > 1.0` (the teacher's unconditional CFG embedding is
computed in `process_sample_prompts`, reusing the text encoder loaded there — see below). Any
prompt file works, even a minimal one.

`--network_dim`/`--network_alpha`/`--network_weights`/`--dim_from_weights` are not supported with
`--tdm_distill` and raise an error if passed: the student/fake-score LoRA's per-module rank and
alpha are always read directly from `--tdm_turbo_lora_init`'s own weights (confirmed on a real
checkpoint to be non-uniform across modules — a single `--network_dim` can't represent it). The
turbo LoRA file must use sd-scripts-style key names (`lora_unet_...`); a ComfyUI/diffusers-style
file (`diffusion_model...`) matches no real module's per-module dim lookup, so every module is
silently skipped and the LoRA network ends up with zero modules for the transformer — no error,
no effective LoRA at all. Convert with musubi-tuner's own `convert_lora.py --target default` first
if needed.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--tdm_distill` | off | Enable TDM diversity distillation |
| `--tdm_turbo_lora_init` | required when on | Turbo LoRA path used to warm-start student + fake-score |
| `--tdm_step_counts` | `1,2,4,8` | Comma-separated student step counts K, sampled per iteration |
| `--tdm_diversity_group_size` | 4 | Same-prompt samples per training step for the diversity term |
| `--tdm_diversity_weight` | 0.1 | Constant weight on the diversity loss (not annealed — see below) |
| `--tdm_guidance_scale` | required, no default | CFG scale for the teacher's real-score forward (`uncond + scale*(cond-uncond)`); `<= 1.0` disables CFG |
| `--fake_score_learning_rate` | 10x `--learning_rate` | Fake-score critic's own optimizer LR |
| `--fake_score_optimizer_type` | mirrors `--optimizer_type` | Fake-score critic's optimizer |

## The diversity term is a real training signal, not a metric

The DINOv3 group-diversity term is fully differentiable end-to-end: gradient flows from the
DINOv3 CLS-token embeddings, back through a grad-enabled VAE decode, back through a fully
grad-enabled student rollout, into the student's LoRA weights. This is implemented via
`Dinov3ImageEmbedder.embed_differentiable` (`tdm_distill.py`), which deliberately avoids
`self.processor`'s PIL/numpy pipeline (not differentiable) and instead resizes/normalizes with
plain tensor ops, and has no `@torch.no_grad()` decorator so gradient can reach the input pixels.
When `--tdm_diversity_weight > 0`, this term actively shapes what
the student learns — it is not a passive/logging-only metric computed on the side.

## Known limitations

- **Data-free**: ignores the dataset's cached image latents entirely — only prompts are used. Every
  other extension in this repo trains on real cached latents; this one does not.
- **Two-optimizer training loop**: the fake-score critic is trained by an extension-owned second
  optimizer, stepped manually inside `process_batch`, not through the base trainer's normal
  optimizer lifecycle.
- **No gradient accumulation**: `--gradient_accumulation_steps` must be 1. The student and the
  fake-score critic share one physical LoRA network across two separately `prepare`d optimizers, so
  the critic's mid-accumulation `zero_grad()` would wipe the student's accumulating gradients.
- **Single-GPU bf16 only**: two `accelerator.prepare`d optimizers sharing one `GradScaler` under fp16
  mixed precision, or running under multi-GPU/DDP, is untested and likely broken (the extra backward
  pass per step confuses DDP's gradient reducer, and fp16 grad scaling gets double-updated).
- **Block swap (`--blocks_to_swap > 0`) is implemented but untested**: `call_dit` resets block
  placement to its canonical layout before every forward (see the docstring there) to work around
  `ModelOffloader`'s single forward/backward-per-step assumption, which this trainer's multiple
  forwards per step (student rollout, fake-score, teacher, updated fake-score) violate. No test
  exercises `blocks_to_swap > 0` in this extension.
- **Teacher CFG requires `--text_encoder` and `--sample_prompts`**: when `--tdm_guidance_scale >
  1.0`, `process_sample_prompts` builds a cached unconditional (empty-prompt) embedding in the same
  text-encoder session it uses for sample-prompt caching, before the DiT loads. This is deliberate:
  a second full text-encoder load in `on_train_start`, after the (possibly quantized, possibly
  block-swapped) DiT is already resident, was confirmed to OOM on a 16GB card in real testing.
  `--tdm_guidance_scale <= 1.0` disables CFG and skips this entirely (single conditional teacher
  forward, no `--text_encoder`/`--sample_prompts` requirement from CFG specifically).
- **omega_tau is min-SNR only, no importance-sampling term**: the fake-score critic's denoising
  loss (Eq. 7) is weighted by `min_snr_weight` (gamma=5, fixed), re-derived for this module's
  velocity-space critic target from the standard min-SNR weighting strategy (Hang et al. 2023).
  The official TDM implementation also applies an importance-sampling correction on top of this,
  but that term corrects for a bias specific to *their* two-hop, model-dependent noise
  construction. We believe no such correction applies here because this module's `x_tau` is
  sampled in a single hop with a fresh random Gaussian draw, which is unbiased *given* `tau` — but
  this is a qualified claim, not a proven one: `tau` here is not drawn from the continuous
  distribution the paper's derivation assumes. It is deterministically the midpoint of a
  uniformly-drawn interval on a `mu=1.15`-shifted `K`-step grid, i.e. a `K`-dependent discrete comb
  whose shape changes every time `sample_step_count` redraws `K`. This interaction between the
  discrete-comb `tau` distribution and the omitted importance-sampling term has not been
  empirically validated. `tdm/tau` and `tdm/omega_tau` are logged per step (see
  `process_batch`'s `loss_metrics`) so the interaction is observable in a real run.
- **Not annealed**: `--tdm_diversity_weight` should be left constant throughout training — Krea's
  own report found annealing this weight toward zero caused diversity to collapse quickly.
- **VAE stays resident on the training device for the whole step when the diversity term runs**:
  because the diversity term needs a grad-enabled VAE decode (`vae.decode_to_pixels` under an
  active autograd graph), `process_batch` moves the VAE onto the training device and cannot move
  it back to CPU immediately afterward — `nn.Module.to()` rebinds parameter storage in place, and
  the outer training loop's `accelerator.backward()` runs *after* `process_batch` returns, so an
  early CPU round-trip would crash with a device mismatch. Instead the VAE is only returned to CPU
  in a dedicated `on_post_optimizer_step` override, once backward has consumed the graph for that
  step. This holds the full VAE resident on-device for an extra portion of every step, on top of
  the resident transformer — a real, additional VRAM cost while `--tdm_diversity_weight > 0`
  (separate from and in addition to the generic "one resident transformer" cost below).
- **Extra grad-enabled forward pass per training step, and it is the dominant VRAM cost**: the
  student rollout used for the diversity term must be fully grad-enabled from step 0
  (`grad_from_step=0`) so gradient reaches every step of the trajectory, unlike the main TDM
  rollout (which only needs its last step differentiated — see `_student_rollout`'s
  `grad_from_step` parameter and the version-counter hazard noted there). This rollout runs at
  batch size `--tdm_diversity_group_size` (default 4) for up to `max(--tdm_step_counts)` steps
  (default 8) — up to 32 sample-forwards with retained activations held simultaneously, not "one
  extra rollout" in any small sense. Peak VRAM from this term scales multiplicatively with
  `K * group_size`, where `K` is redrawn every iteration from `--tdm_step_counts`; it is the
  single largest activation consumer in the step whenever the diversity term is enabled. A
  startup warning fires when `max(--tdm_step_counts) * --tdm_diversity_group_size` exceeds a
  heuristic threshold (16) so this is visible before training starts.
- **VRAM**: one resident transformer plus a VAE decode + DINOv3 forward pass added to every
  training step for the diversity term (see the bullet above for the dominant cost).
- **`--tdm_guidance_scale > 1.0` together with `--tdm_diversity_weight > 0` needs more VRAM than
  either alone**: confirmed on a 16GB card (ConvRot INT8, `--blocks_to_swap 26`) that CFG's extra
  teacher forward and the diversity term's grad-enabled rollout each work fine on their own, but
  together they OOM mid-step during `accelerator.backward()`. This is a real combined-cost ceiling,
  not a code bug. If you hit this, reduce `--tdm_diversity_group_size` (2 is the practical minimum),
  cap `--tdm_step_counts` at a single small value, or use a card with more headroom.
- **Experimental**: no correctness guarantee against the TDM paper's own results; K2 is not an
  architecture the paper evaluates.
- **DINOv3 gate**: `facebook/dinov3-vitb16-pretrain-lvd1689m` requires one-time Hugging Face license
  acceptance plus `huggingface-cli login` (or `hf auth login`) before first use.
- Not tested in composition with `self_flow` or `explorative_modeling` — combining them is out of
  scope for this extension.
