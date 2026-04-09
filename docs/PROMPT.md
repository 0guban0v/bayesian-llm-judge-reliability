# Project Prompt: Bayesian IRT for LLM-as-Judge Reliability

## Context

I am building a portfolio project for a Bayesian statistics course that doubles as a showcase for frontier AI lab roles (research engineer / applied scientist). The project applies **Bayesian Item Response Theory (IRT)** to measure LLM-as-judge reliability — quantifying how much to trust different automated evaluation setups.

## Problem Statement

Frontier labs rely on LLM-as-judge for model evaluation, RLHF data filtering, and benchmark scoring. Judges disagree with each other and with humans. There is no principled uncertainty quantification over judge reliability. This project treats LLM judges as "test-takers" in a psychometric framework and estimates latent reliability parameters with full Bayesian posteriors.

## The Model

**2-Parameter Logistic IRT (2PL):**

For judge `j` evaluating item `i`:

```
P(correct_{ij} = 1) = sigmoid(a_i * (θ_j - b_i))
```

- `θ_j` ~ Normal(0, 1) — latent reliability of judge j
- `b_i` ~ Normal(0, 2) — item difficulty
- `a_i` ~ LogNormal(0, 0.5) — item discrimination (how well item separates good/bad judges)

Inference via NUTS (No-U-Turn Sampler) in NumPyro (JAX-based).
An alternative BlackJAX NUTS path should produce backend-specific posterior artifacts while still honoring the same configured priors and chain count.

## Data Pipeline

1. **Source:** JudgeBench dataset (~620 response pairs with ground-truth correctness labels across reasoning, math, coding, knowledge)
2. **Judge matrix construction:** Run 8-12 judge configurations against a 200-item subset:
   - 2-3 local models via MLX (selected with `llmfit` for hardware fit, then pinned in config)
   - Each model × 2-3 prompt variants (pointwise vs pairwise, with/without CoT, with/without rubric)
3. **Output:** Binary matrix `judges × items → correct/incorrect`, stored as Parquet

## Stack

| Layer | Tool | Rationale |
|-------|------|-----------|
| Package management | `uv` | Fast, lockfile-based, reproducible |
| Linting/formatting | `ruff` | Single tool, fast |
| Data | `polars` | No pandas. Lazy eval, type-safe, fast |
| Notebooks | `marimo` | Reactive, git-friendly (.py files), no Jupyter |
| Validation | `pydantic` | Config and data schemas |
| Inference | `numpyro` | JAX-native PPL, GPU-capable, used at frontier labs |
| Advanced inference | `blackjax` | Composable MCMC kernels in JAX (for prior sensitivity section) |
| Local LLMs | `mlx-lm` | Apple-silicon-native local inference with broader access to high-quality HF model variants |
| Model selection | `llmfit` | Hardware-aware model recommendation |

**Explicit exclusions:** No PyMC, no pandas, no Jupyter, no notebooks that aren't marimo.

## Repo Structure

```
bayesian-llm-judge-reliability/
├── README.md
├── CLAUDE.md                    # Claude Code instructions
├── AGENTS.md                    # Codex agent instructions
├── pyproject.toml
├── configs/
│   └── experiment.yaml          # All hyperparams, judge configs, priors
├── src/
│   ├── __init__.py
│   ├── schemas.py               # Pydantic models for config, data, results
│   ├── judges/
│   │   ├── __init__.py
│   │   ├── runner.py            # Orchestrates judge evaluation runs
│   │   ├── prompts.py           # Prompt templates (pointwise, pairwise, CoT variants)
│   │   └── parsers.py           # Extract verdicts from raw LLM responses
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Load JudgeBench, construct binary matrix
│   │   └── validate.py          # Data quality checks
│   ├── models/
│   │   ├── __init__.py
│   │   ├── irt_numpyro.py       # 1PL and 2PL IRT models in NumPyro
│   │   └── irt_blackjax.py      # Same model with manual BlackJAX inference loop
│   └── analysis/
│       ├── __init__.py
│       ├── diagnostics.py       # R-hat, ESS, trace plots, rank plots
│       ├── plots.py             # Posterior ridge plots, PPC, comparison plots
│       └── posterior_queries.py  # P(θ_a > θ_b | data), credible intervals
├── notebooks/
│   └── results.py               # Marimo notebook — results and figures only
├── data/
│   ├── raw/                     # JudgeBench source files (gitignored if large)
│   ├── processed/               # Judge response matrix as parquet
│   ├── logs/                    # JSONL logs of every judge API call
│   └── README.md                # Data provenance and license info
├── figures/                     # Publication-quality plots (PNG + PDF)
├── report/
│   └── report.pdf               # Course deliverable (4-6 pages)
└── scripts/
    ├── recommend_models.sh      # llmfit recommendation artifact for current hardware
    ├── setup_models.sh          # prefetch pinned MLX models from configs/experiment.yaml
    └── run_experiment.sh        # End-to-end: collect → infer → plot
```

## Key Design Decisions to Discuss

1. **1PL vs 2PL model selection:** Should we include item discrimination? What does model comparison (WAIC/LOO) tell us?
2. **Prior sensitivity:** How much do results change with vague vs informative priors on θ, b, a?
3. **Judge grouping:** Should we add hierarchical structure (e.g., group judges by base model, with model-level hyperpriors)?
4. **Prompt template design:** What makes a good pointwise vs pairwise judge prompt? Should we include a rubric?
5. **Sample size:** Is 200 items × 10 judges enough for stable IRT estimation? What does posterior predictive check tell us?
6. **Convergence:** What to do if NUTS struggles with the discrimination parameter `a`?

## Deliverables

1. **Git repo** with clean commit history, typed Python, reproducible via `uv run`
2. **Marimo notebook** with results, figures, and interpretation
3. **PDF report** (4-6 pages) for course submission
4. **README** with one hero figure, one key finding, and reproduction instructions

## What I Need Help With

When asking for assistance, I will specify which aspect:
- `[DATA]` — JudgeBench loading, judge matrix construction, data validation
- `[MODEL]` — NumPyro/BlackJAX model specification, prior choices, parameterization
- `[INFERENCE]` — MCMC diagnostics, convergence issues, sampler tuning
- `[ANALYSIS]` — Posterior queries, plots, interpretation of results
- `[INFRA]` — Project setup, configs, CI, repo structure
- `[WRITING]` — Report sections, README, figure captions
- `[JUDGES]` — Prompt engineering for judge configurations, MLX setup, local model selection
