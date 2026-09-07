# /// script
# dependencies = ["datasets"]
# ///
"""Sample a diverse, reproducible prompt set from AIML-TUDA/t2i-diversity-evalprompts.

Produces the training-conditioning prompt pool and a small --sample_prompts subset for
TDM distillation's 500-step diversity evaluation run. See
docs/superpowers/specs/2026-09-07-tdm-diversity-eval-dataset-design.md for the full design.

The dataset is long-format: columns are `prompt`, `type`, `prompt_id` (4340 rows = 1085
unique prompt_id values x 4 type values). We group by prompt_id, sample 300 unique
prompt_ids, and for each pick one type via positional rotation.
"""

import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 42
SAMPLE_SIZE = 300
TYPE_ROTATION = ["original", "short_gpt4o", "medium_gpt4o", "long_gpt4o"]
SAMPLE_SUBSET_STRIDE = 25

FULL_PROMPTS_PATH = Path("notes/tdm-distill-eval-prompts.txt")
SAMPLE_PROMPTS_PATH = Path("notes/tdm-distill-sample-prompts.txt")


def main() -> None:
    dataset = load_dataset("AIML-TUDA/t2i-diversity-evalprompts", split="train")

    required_cols = {"prompt", "type", "prompt_id"}
    missing = required_cols - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required_cols)} on AIML-TUDA/t2i-diversity-evalprompts, "
            f"missing {sorted(missing)}. Actual columns: {dataset.column_names}"
        )

    by_prompt_id: dict[str, dict[str, str]] = defaultdict(dict)
    for row in dataset:
        by_prompt_id[row["prompt_id"]][row["type"]] = row["prompt"]

    missing_types = [pid for pid, variants in by_prompt_id.items() if not set(TYPE_ROTATION) <= set(variants)]
    if missing_types:
        raise ValueError(
            f"Expected every prompt_id to have all of {TYPE_ROTATION}; "
            f"{len(missing_types)} prompt_id(s) don't, e.g. {missing_types[0]!r} has {sorted(by_prompt_id[missing_types[0]])}"
        )

    rng = random.Random(SEED)
    prompt_ids = rng.sample(sorted(by_prompt_id), SAMPLE_SIZE)

    prompts = []
    for i, pid in enumerate(prompt_ids):
        variant = TYPE_ROTATION[i % len(TYPE_ROTATION)]
        prompt = by_prompt_id[pid][variant]
        if not prompt or not prompt.strip():
            raise ValueError(f"prompt_id {pid!r}, type {variant!r} is empty; cannot use as a prompt.")
        prompts.append(prompt.strip())

    Path("notes").mkdir(exist_ok=True)

    FULL_PROMPTS_PATH.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    subset = prompts[0::SAMPLE_SUBSET_STRIDE]
    SAMPLE_PROMPTS_PATH.write_text("\n".join(subset) + "\n", encoding="utf-8")

    print(f"Wrote {len(prompts)} prompts to {FULL_PROMPTS_PATH}")
    print(f"Wrote {len(subset)} prompts to {SAMPLE_PROMPTS_PATH}")


if __name__ == "__main__":
    main()
