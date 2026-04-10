# Workflow

## Setup

```bash
uv sync
make pre-commit-install
make recommend-models
make verify-models MODELS="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 google/gemma-2-9b-it"
make setup-models
```

## Full Pipeline

```bash
make run
```

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
