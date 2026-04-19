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

`make judge` resumes from existing judge logs by skipping already recorded `(item_key, prompt_order)` pairs, where `item_key` is the split-qualified JudgeBench item identifier. New JSONL records embed the prompt protocol and judge metadata needed for resumable runs. Older logs that predate embedded metadata are unsupported and must be deleted before rerunning under the current protocol. `make matrix`, `make validate`, `make infer`, `make diagnostics`, and `make plots` rebuild derived artifacts from current logs and should be run only after all intended judges finish.

`make infer` now fails fast on incomplete judge coverage. If any configured judge column is missing or partially observed, inference exits with a coverage error instead of fitting a posterior on an invalid matrix.

`make diagnostics` writes `figures/diagnostics_summary.png`.

If a compatible posterior archive is present, `make plots` regenerates `figures/judge_accuracy_ppc.png`,
`figures/judge_reliability_ridge.png`, `figures/trace_theta_tau_theta.png`, and
`figures/separation_by_judge.png`. Legacy posterior archives are unsupported and must be regenerated with
current inference code.

If the posterior archive includes `theta_source` and `source_ids` from the `source_hier` model variant, `make plots` also writes `figures/judge_reliability_by_source.png`.

Checked-in source-plot and report-export policies now live under `analysis.*` in the experiment YAML. In particular, `analysis.plots.max_sources` controls how many sources appear in source-aware outputs, and `analysis.report.*` controls standout-case selection and response synopsis length.

## Tracked Study Workflow

Use the tracked path when experiment-result material should live in MLflow rather than in repo-local comparison summaries.

Run one tracked config:

```bash
make tracked-analysis CONFIG=configs/experiment_gpt_global.yaml
```

`tracked-analysis` is the tracked equivalent of the full pipeline. It samples or reloads items for the requested
config, resumes judge collection against those items, rebuilds the matrix from the resulting logs, runs inference,
and logs outputs to MLflow.

Run the baseline plus all four study configs sequentially:

```bash
make tracked-study-all
```

Because tracked runs invoke judge collection, they can take substantially longer than figure-only or inference-only
commands. They should be treated as experiment executions, not lightweight report refreshes.

Tracked runs use local MLflow storage in `mlruns/` and log:

- resolved config snapshot
- item and matrix hashes
- posterior artifacts
- figures
- generated report snippets
- diagnostics and pairwise comparison metrics

MLflow is the canonical store for tracked study results. The repo report does not auto-build a robustness table from tracked runs.

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
