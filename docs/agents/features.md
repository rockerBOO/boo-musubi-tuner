# Adding a new feature/extension

Checklist for adding a new extension (e.g. a new training technique, or support for a new base architecture like
FLUX.2/Krea 2 on an existing technique). Follow the pattern already established by `self_flow`, `wavelet_loss`,
and `explorative_modeling` — see CLAUDE.md's "Architecture" section for the extension-seam mechanics.

## 1. Source module(s)

Under `src/boo_musubi_tuner/<extension>/`:

- The core logic as free functions (pure tensor ops, testable without a model) plus the trainer subclass/mixin
  that wires them into musubi-tuner's `NetworkTrainer` seams (`call_dit`, `process_batch`, `on_train_start`,
  etc.) — no edits to musubi-tuner's own model/trainer files. One file per base architecture (e.g.
  `flux_2_train_network_<extension>.py`, `krea2_train_network_<extension>.py`), or a single
  architecture-generic mixin module if the feature needs no model-specific hooks (see `explorative_modeling.py`).
- Every overridden seam must check the feature's flag first and fall through to `super()` unchanged when it's
  off — the "vanilla passthrough when flag is off" rule. This lets the same trainer class serve as a plain base
  trainer and the extended one.
- An `<extension>_setup_parser(parser)` function adding the feature's CLI args.
- A `main()` entry point: `setup_parser_common()` + the base model's `*_setup_parser()` + the extension's
  `*_setup_parser()`, then construct and run the trainer.
- Module paths and logger names must live under `boo_musubi_tuner.<extension>.*` — nothing should reference a
  path or package that only exists outside this repo.

## 2. Tests

Under `tests/<extension>/`, mirroring the source package layout, with one test file per concern (loss math,
masking, timestep helpers, arg validation, lifecycle hooks, save/load, call_dit wiring, etc. — see
`tests/self_flow/` for the fullest example). If the extension needs shared fixtures (tiny CPU-only model
configs — no real weights, so the suite stays fast without a GPU), add a `conftest.py` in that directory; helpers
that aren't fixtures (e.g. batch builders) can live in a plain module imported directly by the tests that need
them. Run the new tests before moving on:

```bash
uv run pytest tests/<extension>/
```

## 3. Docs

Add `docs/<extension>.md` with:

- A one-paragraph description of what the feature does, in end-user terms (not implementation rationale).
- **Source material**: a link to the paper/website/repo this is based on. If there's a paper, cite title,
  authors, and arXiv link. Don't guess at URLs — use ones the user gives you, or ones you've fetched and
  confirmed.
- The trainer file(s) it applies to.
- A usage example (`accelerate launch ...` invocation).
- A flags table (flag, default, meaning).
- Any known limitations or compatibility gaps.

Then add one line to README.md's Extensions list linking to the new doc, and one bullet to CLAUDE.md's
Extensions list (short — implementation detail belongs in the doc, not CLAUDE.md).

## 4. Before committing

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```
