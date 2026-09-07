"""Build a --dataset_config-ready dataset directory from the 300-prompt eval set.

Reuses smoke_test_data's 4 fixture images (content is discarded by the TDM distill trainer,
only latent shape matters) and, where possible, its already-computed 256x256 latent caches, to
avoid re-encoding duplicate image content 300 times. Text-encoder caching still runs for real
since every caption is unique.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RESOLUTION = 256
PROMPTS_PATH = Path("notes/tdm-distill-eval-prompts.txt")
SMOKE_TEST_IMAGES = Path("smoke_test_data/images")
SMOKE_TEST_CACHE = Path("smoke_test_data/cache")
NUM_PLACEHOLDER_IMAGES = 4

OUTPUT_ROOT = Path("notes/tdm-distill-eval-dataset")
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_CACHE = OUTPUT_ROOT / "cache"
DATASET_CONFIG_PATH = OUTPUT_ROOT / "dataset_config.toml"

DATASET_CONFIG_TEMPLATE = """\
[general]
caption_extension = '.txt'
enable_bucket = true
resolution = {resolution}

[[datasets]]
image_directory = '{image_dir}'
cache_directory = '{cache_dir}'
"""


def build_images_and_captions(prompts: list[str]) -> None:
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    for i, prompt in enumerate(prompts):
        placeholder_idx = i % NUM_PLACEHOLDER_IMAGES
        src_image = SMOKE_TEST_IMAGES / f"{placeholder_idx:02d}.png"
        dst_image = OUTPUT_IMAGES / f"{i:03d}.png"
        shutil.copyfile(src_image, dst_image)
        (OUTPUT_IMAGES / f"{i:03d}.txt").write_text(prompt + "\n", encoding="utf-8")


def copy_or_queue_latent_cache(num_prompts: int) -> list[int]:
    OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    needs_compute = []
    for i in range(num_prompts):
        placeholder_idx = i % NUM_PLACEHOLDER_IMAGES
        cache_name = f"{placeholder_idx:02d}_{RESOLUTION:04d}x{RESOLUTION:04d}_kr2.safetensors"
        src_cache = SMOKE_TEST_CACHE / cache_name
        dst_cache = OUTPUT_CACHE / f"{i:03d}_{RESOLUTION:04d}x{RESOLUTION:04d}_kr2.safetensors"
        if src_cache.exists():
            shutil.copyfile(src_cache, dst_cache)
        else:
            needs_compute.append(i)
    return needs_compute


def write_dataset_config() -> None:
    DATASET_CONFIG_PATH.write_text(
        DATASET_CONFIG_TEMPLATE.format(
            resolution=RESOLUTION,
            image_dir=str(OUTPUT_IMAGES.resolve()),
            cache_dir=str(OUTPUT_CACHE.resolve()),
        ),
        encoding="utf-8",
    )


def run_cache_latents_for_missing(needs_compute: list[int], vae_path: str) -> None:
    if not needs_compute:
        return
    print(
        f"{len(needs_compute)} item(s) have no matching smoke-test latent cache to copy; "
        "running krea2_cache_latents for the full dataset to fill them in.",
        file=sys.stderr,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "musubi_tuner.krea2_cache_latents",
            "--dataset_config",
            str(DATASET_CONFIG_PATH),
            "--vae",
            vae_path,
        ],
        check=True,
    )


def run_cache_text_encoder_outputs(text_encoder_path: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "musubi_tuner.krea2_cache_text_encoder_outputs",
            "--dataset_config",
            str(DATASET_CONFIG_PATH),
            "--text_encoder",
            text_encoder_path,
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vae",
        required=True,
        help="Path to the Qwen-Image VAE checkpoint (used only if a placeholder image's latent cache can't be copied from smoke_test_data).",
    )
    parser.add_argument(
        "--text_encoder",
        required=True,
        help="Path to the Qwen3-VL text encoder checkpoint (used for every item, since captions are unique).",
    )
    args = parser.parse_args()

    prompts = [line.strip() for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    build_images_and_captions(prompts)
    write_dataset_config()
    needs_compute = copy_or_queue_latent_cache(len(prompts))
    run_cache_latents_for_missing(needs_compute, args.vae)
    run_cache_text_encoder_outputs(args.text_encoder)
    print(f"Dataset ready at {OUTPUT_ROOT} ({len(prompts)} items).")


if __name__ == "__main__":
    main()
