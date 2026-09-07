# /// script
# dependencies = ["sentence-transformers"]
# ///
"""Sanity-check that the 300 sampled diversity-eval prompts aren't accidentally clustered.

Standalone check, not wired into training. See
docs/superpowers/specs/2026-09-07-tdm-diversity-eval-dataset-design.md for context.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer, util

PROMPTS_PATH = Path("notes/tdm-distill-eval-prompts.txt")
MODEL_NAME = "all-MiniLM-L6-v2"
HISTOGRAM_BUCKETS = 10


def main() -> None:
    prompts = [line.strip() for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(prompts, convert_to_tensor=True, show_progress_bar=True)

    cosine_sim = util.cos_sim(embeddings, embeddings)
    n = len(prompts)
    pairwise_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_distances.append(1.0 - cosine_sim[i][j].item())

    pairwise_distances.sort()
    mean_dist = sum(pairwise_distances) / len(pairwise_distances)
    print(f"{n} prompts, {len(pairwise_distances)} pairs")
    print(f"cosine distance: mean={mean_dist:.4f} min={pairwise_distances[0]:.4f} max={pairwise_distances[-1]:.4f}")

    bucket_width = 1.0 / HISTOGRAM_BUCKETS
    counts = [0] * HISTOGRAM_BUCKETS
    for d in pairwise_distances:
        bucket = min(int(d / bucket_width), HISTOGRAM_BUCKETS - 1)
        counts[bucket] += 1
    print("distance histogram (0.0 -> 1.0):")
    for b, count in enumerate(counts):
        lo, hi = b * bucket_width, (b + 1) * bucket_width
        print(f"  [{lo:.1f}, {hi:.1f}): {count}")


if __name__ == "__main__":
    main()
