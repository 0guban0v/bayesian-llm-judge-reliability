# Workflow

## Setup

```bash
uv sync
make pre-commit-install
make recommend-models
make verify-models MODELS="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3 google/gemma-2-9b-it"
make setup-models
```

`google/gemma-2-9b-it` is a gated Hugging Face repo. Request access on Hugging Face first, then authenticate locally before `make verify-models` or `make setup-models`:

```bash
uv run python -m huggingface_hub.commands.huggingface_cli login
```

## Full Pipeline

```bash
make full
```

`make judge` resumes from existing judge logs by skipping already recorded `(item_id, prompt_order)` pairs. `make matrix`, `make validate`, `make infer`, `make diagnostics`, and `make plots` rebuild derived artifacts from current logs and should be read only after all intended judges finish. `make smoke` is intentionally different: it runs short compatibility checks, deletes those smoke JSONL files, and then starts the full judge phase from a clean slate.

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
