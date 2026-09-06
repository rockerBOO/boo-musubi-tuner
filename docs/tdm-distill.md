# TDM Diversity Distillation (K2)

Experimental Krea 2 (K2) extension implementing Trajectory Distribution Matching (TDM) for
guidance+step distillation, with a DINOv3 group-diversity term folded into the student's loss to
counteract the diversity loss guidance distillation otherwise causes. Warm-starts the student and
an auxiliary fake-score critic from an existing K2 Turbo LoRA rather than training from scratch,
distilling against the frozen K2 raw model's true multi-step trajectory.

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

`--sample_prompts` is required whenever `--tdm_diversity_weight > 0`: the base trainer only loads
the VAE when sampling is configured, and the diversity term needs the VAE to decode latents to
pixels. Any prompt file works, even a minimal one.

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
plain tensor ops, and avoids `@torch.no_grad()` (used only by the separate `embed()` convenience
method for non-training use). When `--tdm_diversity_weight > 0`, this term actively shapes what
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
- **Teacher CFG requires `--text_encoder`**: when `--tdm_guidance_scale > 1.0`, `on_train_start`
  loads the Qwen3-VL encoder once to build a cached unconditional (empty-prompt) embedding, then
  frees it — same pattern as `--sample_prompts` encoding. `--tdm_guidance_scale <= 1.0` disables
  CFG and skips this entirely (single conditional teacher forward, no `--text_encoder` needed).
- **Simplified TDM math**: `fake_score_denoising_loss` implements a plain weighted MSE, not the
  paper's full importance-sampling-reweighted Eq. 7. Treat this as an approximation.
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
- **Extra grad-enabled forward pass per training step**: the student rollout used for the
  diversity term must be fully grad-enabled from step 0 (`grad_from_step=0`) so gradient reaches
  every step of the trajectory, unlike the main TDM rollout (which only needs its last step
  differentiated — see `_student_rollout`'s `grad_from_step` parameter and the version-counter
  hazard noted there). This adds roughly one extra full-trajectory transformer rollout per
  training step beyond what a single-pass TDM step needs, when the diversity term is enabled.
- **VRAM**: one resident transformer plus a VAE decode + DINOv3 forward pass added to every
  training step for the diversity term (see the two bullets above for the specific costs).
- **Experimental**: no correctness guarantee against the TDM paper's own results; K2 is not an
  architecture the paper evaluates.
- **DINOv3 gate**: `facebook/dinov3-vitb16-pretrain-lvd1689m` requires one-time Hugging Face license
  acceptance plus `huggingface-cli login` (or `hf auth login`) before first use.
- Not tested in composition with `self_flow` or `explorative_modeling` — combining them is out of
  scope for this extension.
