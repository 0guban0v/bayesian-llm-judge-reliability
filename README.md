# Bayesian LLM Judge Reliability

Bayesian Item Response Theory (IRT) for measuring how reliable different LLM-as-judge setups are on JudgeBench. The project runs local judges through MLX, turns judge decisions into a binary judge-by-item matrix, and fits Bayesian 1PL or 2PL IRT models in NumPyro.

> Public viewing only. All rights reserved. No reuse, copying, modification, or
> redistribution is permitted without prior written permission.

## Pipeline

```text
configs/experiment.yaml
  -> src/schemas.py
  -> src/data/loader.py
  -> src/judges/runner.py
  -> src/models/irt_numpyro.py
  -> src/analysis/*.py
  -> notebooks/results.py
```

## Quick Start

```bash
uv sync
make pre-commit-install
make recommend-models
make verify-models MODELS="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B Qwen/Qwen2.5-7B-Instruct"
make setup-models
uv run python -m src.data.loader --config configs/experiment.yaml
uv run python -m src.judges.runner --config configs/experiment.yaml
uv run python -m src.data.validate --config configs/experiment.yaml
uv run python -m src.models.irt_numpyro --config configs/experiment.yaml
uv run python -m src.analysis.plots --config configs/experiment.yaml
uv run marimo edit notebooks/results.py
```

Set `inference.backend` in `configs/experiment.yaml` to choose which posterior artifact downstream diagnostics, plots, and posterior queries read. `src.models.irt_numpyro` writes the NumPyro artifact, and `src.models.irt_blackjax` writes the BlackJAX artifact.

## Guardrails

Install repository hooks once after `uv sync`:

```bash
make pre-commit-install
```

Run the full local hook suite on demand:

```bash
make pre-commit-run
```

`pre-commit` runs YAML hygiene, Ruff, banned-pattern checks, and `vulture` dead-code detection before commit. The full unit suite runs on `pre-push`.

Verify that candidate models satisfy this repo's MLX constrained-decoding path before pinning them in config:

```bash
make verify-models MODELS="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3"
```

## Repository Layout

- `configs/experiment.yaml`: single source of truth for data, judges, and inference settings
- `src/data/`: JudgeBench loading, subset preparation, matrix construction, validation
- `src/judges/`: prompt templates, parsing, and MLX-based judge execution
- `src/models/`: NumPyro and BlackJAX inference paths
- `src/analysis/`: posterior diagnostics, figures, and posterior comparisons
- `notebooks/results.py`: marimo notebook for the final presentation layer

## Hero Figure

Save the main comparison figure as `figures/hero.png` once model runs are complete.

## License

This repository is public for viewing only. All rights are reserved.

No use, copying, modification, distribution, or derivative works are permitted
without prior written permission from the author.
