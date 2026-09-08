# DINOv3 Standalone Diversity Fine-Tuning (K2)

Experimental Krea 2 (K2) extension that fine-tunes an existing LoRA (the Turbo LoRA, or a
checkpoint from a completed `tdm_distill` run) purely to increase same-prompt,
different-seed sample diversity. No distillation loss is involved — this is decoupled
entirely from `tdm_distill`, so it can be run as its own phase before or after a TDM
guidance-distillation pass, instead of fusing both objectives into one gradient step (which
was found to destabilize training when tried).

**Source material**: [Krea 2 technical report](https://www.krea.ai/blog/krea-2-technical-report)
(DINOv3 group-diversity technique: same-prompt/different-seed groups, deliberately unannealed
weight — Krea found annealing caused rapid diversity collapse). Note: Krea applies this
technique to their prompt-expander RL stage, not as a direct fine-tuning loss on the image
generator's own LoRA weights — this extension is a novel application of the same tool, not a
reproduction of a validated recipe.

**Trainer**: `src/boo_musubi_tuner/dino_diversity/krea2_train_network_dino_diversity.py`
(`Krea2DinoDiversityNetworkTrainer`).

## Usage

```bash
accelerate launch src/boo_musubi_tuner/dino_diversity/krea2_train_network_dino_diversity.py \
  --dit /path/to/krea2_raw.safetensors \
  --vae /path/to/qwen_image_vae.safetensors \
  --text_encoder /path/to/qwen3_vl_text_encoder.safetensors \
  --dataset_config /path/to/dataset_config.toml \
  --dino_diversity \
  --dino_diversity_lora_init /mnt/900/lora/krea2/krea2_turbo_lora_rank_64_bf16.safetensors \
  --dino_diversity_group_size 4 \
  --dino_diversity_step_count 8 \
  --dino_diversity_memory_efficient \
  ... (usual musubi-tuner LoRA training flags)
```

`--network_dim`/`--network_alpha`/`--network_weights`/`--dim_from_weights` are not supported —
the LoRA's per-module rank/alpha are always inferred directly from `--dino_diversity_lora_init`'s
own weights (same reasoning as `tdm_distill`: warm-start LoRAs are not guaranteed to be
uniform-rank across modules).

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--dino_diversity` | off | Enable standalone DINOv3 group-diversity fine-tuning |
| `--dino_diversity_lora_init` | required when on | Any LoRA checkpoint to warm-start from (Turbo LoRA or a `tdm_distill` checkpoint) |
| `--dino_diversity_group_size` | 4 | Same-prompt, different-seed samples per training step |
| `--dino_diversity_step_count` | 8 | Euler rollout steps per sample (full inference-length by default — no distillation critic competing for VRAM here) |
| `--dino_diversity_memory_efficient` | off | Two-pass per-sample gradient accumulation; same math, lower peak VRAM, slower |
| `--dino_diversity_pass1_chunk_size` | `--dino_diversity_group_size` | Sub-batch size for the memory-efficient path's pass 1 forward. Lower this independently of `--dino_diversity_group_size` to raise group size (more diverse comparison set) without raising pass 1's peak VRAM |
| `--dino_diversity_debug_save_images` | off | Save the exact post-VAE-decode, pre-DINOv3 pixel batch to `<output_dir>/dino_diversity_debug/` as PNGs — debugging aid to confirm the rollout/decode is producing correct images before trusting the diversity loss |
| `--dino_diversity_debug_save_every_n_steps` | 1 | Throttle for the above — only save every N steps |

## Known limitations

- **Requires `--dataset_config` despite being fully data-free**: real cached latents are never
  used as training content, only for shape/prompt cycling and to keep the base trainer's
  sample-image machinery working during training. Removing this requirement is future work.
- **Block swap (`--blocks_to_swap > 0`) needs `--block_swap_h2d_only`**: this trainer's rollout
  calls `call_dit` multiple times per step (once per Euler step, up to
  `--dino_diversity_step_count`), which the classic `ModelOffloader` block-swap path handles
  very poorly — see `docs/tdm-distill.md`'s block-swap section for measured per-call costs on
  real hardware. Always pair `--blocks_to_swap` with `--block_swap_h2d_only`,
  `--block_swap_ring_size`, and `--use_pinned_memory_for_block_swap`.
- **VRAM**: full-step-count (default 8), full-gradient rollouts at `group_size=4` build a
  substantially larger graph than `tdm_distill`'s diversity term ever does by default (which
  uses `step_count=1`, `group_size=2`) — use `--dino_diversity_memory_efficient` on
  VRAM-constrained hardware. Note that `--dino_diversity_memory_efficient`'s pass 1 (which gets
  every sample's embedding before the per-sample backward pass) still runs the whole group
  through one no_grad forward by default — no backward graph is retained, but forward-pass
  activation memory (attention, VAE decode) still scales with batch size, so a large `group_size`
  can still OOM pass 1 even with `--dino_diversity_memory_efficient` on. Set
  `--dino_diversity_pass1_chunk_size` below `group_size` to bound pass 1's peak VRAM
  independently of how large a group you want for the diversity comparison itself (e.g.
  `--dino_diversity_group_size 4 --dino_diversity_pass1_chunk_size 2` on a 16GB card that OOMs
  at `group_size=4` but fits `group_size=2`).
- **Experimental / unvalidated combination**: Krea's report validates this diversity technique
  on their prompt-expander RL setup, not as a direct fine-tuning loss on a distilled image
  generator's own LoRA weights.
- **DINOv3 gate**: `facebook/dinov3-vitb16-pretrain-lvd1689m` requires one-time Hugging Face
  license acceptance plus `huggingface-cli login` (or `hf auth login`) before first use.
