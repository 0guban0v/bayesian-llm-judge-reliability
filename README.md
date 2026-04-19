# Bayesian LLM Judge Reliability

Measure how reliable local LLM judges are on JudgeBench with Bayesian Item Response Theory. This repo runs local MLX judges, builds an item-by-judge correctness matrix, and fits Bayesian 1PL/2PL IRT models in PyMC.

![Judge Reliability Posterior](figures/judge_reliability_ridge.png)

## What It Does

- runs local judge models with a fixed constrained `FINAL VERDICT: A|B` protocol
- rebuilds a binary judge-by-item matrix from append-only JSONL logs
- fits Bayesian IRT to separate judge reliability from item difficulty
- produces posterior diagnostics, global judge figures, and source-aware comparison figures
- logs tracked study runs to local MLflow with configs, metrics, posterior artifacts, and figures

## Current Setup

- dataset: JudgeBench via Hugging Face
- judges: local MLX-compatible models
- protocol: one fixed verdict-only pointwise comparison prompt
- inference: PyMC NUTS with reproducible config-driven settings
- outputs: parquet items/matrix, posterior `.npz`, diagnostics, and figures

## Quick Start

```bash
uv sync
make pre-commit-install
make setup-models
make full
```

## Tracked Study Runs

Use the tracked workflow when you want experiment-result material in MLflow rather than repo-local comparison summaries.

Run one tracked config:

```bash
make tracked-analysis CONFIG=configs/experiment_gpt_global.yaml
```

`tracked-analysis` is end-to-end: it samples or reloads items, resumes judge collection into JSONL logs,
rebuilds the matrix from current logs, runs inference, and uploads artifacts to MLflow.

Run the baseline plus the four split-by-variant study configs sequentially:

```bash
make tracked-study-all
```

`tracked-study-all` finishes with a pooled `report-exports` pass that generates the cross-run
`model_comparison.tex` table. That table is a bounded research output: matched split-wise
PSIS-LOO/WAIC comparison for `global` versus `source_hier` within the `2PL` family only.

Tracked runs log parameters, diagnostics, posterior summaries, figures, and generated report snippets to local
MLflow with SQLite metadata in `mlflow.db` and file artifacts in `mlruns/`.

## Docs

- [Workflow](docs/workflow.md): setup, model verification, and pipeline commands
- [Profiling](docs/profiling.md): full-run metrics and stage profiling
- [Structure](docs/structure.md): repo layout and artifact flow
- [Assumptions](docs/assumptions.md): what the current experiment treats as true
- [Limitations](docs/limitations.md): what the current results do not justify

## License

This repository is public for viewing only. All rights are reserved.

No use, copying, modification, distribution, or derivative works are permitted
without prior written permission from the author.
