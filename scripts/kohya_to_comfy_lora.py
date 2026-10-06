"""Convert a TDM (kohya-named) LoRA checkpoint to ComfyUI names, using a ComfyUI LoRA as the name template.

Usage: python scripts/kohya_to_comfy_lora.py TEMPLATE_COMFY.safetensors IN_KOHYA.safetensors OUT.safetensors

Names and shapes come from the template (e.g. the original ComfyUI turbo LoRA). Keys the trainer does not
train (such as bias entries) are copied from the template unchanged. The result replaces the turbo LoRA in
ComfyUI; do not stack it on top of the turbo LoRA.
"""

import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PREFIX = "diffusion_model."


def convert(template_path: str, trained_path: str, out_path: str) -> tuple[int, int]:
    with safe_open(template_path, "pt") as s:
        template = {k: s.get_tensor(k) for k in s.keys()}  # noqa: SIM118 (safe_open is not iterable)
    with safe_open(trained_path, "pt") as s:
        trained = {k: s.get_tensor(k) for k in s.keys()}  # noqa: SIM118 (safe_open is not iterable)

    out, converted = {}, 0
    for key, ref in template.items():
        if key.endswith((".lora_down.weight", ".lora_up.weight")):
            module, part = key.rsplit(".", 2)[0], key.rsplit(".", 2)[1]
            kohya_key = "lora_unet_" + module[len(PREFIX) :].replace(".", "_") + "." + part + ".weight"
            out[key] = trained[kohya_key].to(torch.bfloat16)
            if out[key].shape != ref.shape:
                raise ValueError(f"{key}: shape {tuple(out[key].shape)} != template {tuple(ref.shape)}")
            converted += 1
        else:
            out[key] = ref
    save_file(out, out_path)
    return converted, len(out) - converted


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    converted, copied = convert(*sys.argv[1:])
    print(f"{converted} LoRA tensors converted, {copied} copied from the template")


if __name__ == "__main__":
    main()
