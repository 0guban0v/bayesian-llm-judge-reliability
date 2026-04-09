# Bayesian LLM Judge Reliability

Bayesian Item Response Theory (IRT) for measuring how reliable different LLM-as-judge setups are on JudgeBench. The project runs local judges through MLX, turns judge decisions into a binary judge-by-item matrix, and fits Bayesian 1PL or 2PL IRT models in NumPyro.

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
make recommend-models
make setup-models
uv run python -m src.data.loader --config configs/experiment.yaml
uv run python -m src.judges.runner --config configs/experiment.yaml
uv run python -m src.data.validate --config configs/experiment.yaml
uv run python -m src.models.irt_numpyro --config configs/experiment.yaml
uv run python -m src.analysis.plots --config configs/experiment.yaml
uv run marimo edit notebooks/results.py
```

Set `inference.backend` in `configs/experiment.yaml` to choose which posterior artifact downstream diagnostics, plots, and posterior queries read. `src.models.irt_numpyro` writes the NumPyro artifact, and `src.models.irt_blackjax` writes the BlackJAX artifact.

## Repository Layout

- `configs/experiment.yaml`: single source of truth for data, judges, and inference settings
- `src/data/`: JudgeBench loading, subset preparation, matrix construction, validation
- `src/judges/`: prompt templates, parsing, and MLX-based judge execution
- `src/models/`: NumPyro and BlackJAX inference paths
- `src/analysis/`: posterior diagnostics, figures, and posterior comparisons
- `notebooks/results.py`: marimo notebook for the final presentation layer

## Hero Figure

Save the main comparison figure as `figures/hero.png` once model runs are complete.
