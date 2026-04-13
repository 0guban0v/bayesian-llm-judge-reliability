# Workflow

## Setup

```bash
uv sync
make pre-commit-install
make recommend-models
make verify-models
make setup-models
```

`make verify-models` verifies the configured `judges[*].model` entries from `$(CONFIG)` by default. Use `MODELS="..."` only for ad hoc candidate checks.

`google/gemma-2-9b-it` is a gated Hugging Face repo. Request access on Hugging Face first, then authenticate locally before `make verify-models` or `make setup-models`:

```bash
uv run python -m huggingface_hub.commands.huggingface_cli login
```

## Full Pipeline

```bash
make full
```

`make judge` resumes from existing judge logs by skipping already recorded `(item_id, prompt_order)` pairs. `make matrix`, `make validate`, `make infer`, `make diagnostics`, and `make plots` rebuild derived artifacts from current logs and should be run only after all intended judges finish.

`make infer` now fails fast on incomplete judge coverage. If any configured judge column is missing or partially observed, inference exits with a coverage error instead of fitting a posterior on an invalid matrix.

`make plots` always regenerates the global ridge, posterior predictive, and item-parameter figures. If the posterior archive includes `theta_source` and `source_ids` from the `source_hier` model variant, it also writes `figures/judge_reliability_by_source.png` as a synchronized small-multiples comparison over the most data-rich sources.

## Stepwise Run

```bash
make items-refresh
make judge
make matrix
make validate
make infer
make diagnostics
make plots
```

## Quality

```bash
make pre-commit-run
make test
```

Current local-judge path assumes Apple Silicon with MLX. Sequential multi-model runs are sensitive to Metal unified-memory pressure, and `py-spy` tracing on macOS requires elevated privileges. Avoid running artifact-producing targets with `sudo`; it can leave root-owned Hugging Face cache or repo artifacts behind and break later normal runs.
