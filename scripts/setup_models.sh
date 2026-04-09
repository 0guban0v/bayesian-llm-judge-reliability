#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiment.yaml}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"

preflight_mlx() {
  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "mlx setup requires macOS on Apple Silicon." >&2
    exit 1
  fi

  if ! command -v system_profiler >/dev/null 2>&1; then
    echo "system_profiler is unavailable; cannot verify Metal support." >&2
    exit 1
  fi

  if ! system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
    echo "No Metal-capable GPU was detected by system_profiler." >&2
    echo "MLX requires a local macOS session with Metal available." >&2
    exit 1
  fi
}

prefetch_models() {
  uv run python - "$CONFIG_PATH" <<'PY'
from __future__ import annotations

import os
import sys
import logging

from huggingface_hub import login
from mlx_lm import load

from src.schemas import ExperimentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [setup_models] %(message)s",
)
logger = logging.getLogger("setup_models")

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)

config = ExperimentConfig.from_yaml(sys.argv[1])
seen: set[tuple[str, bool]] = set()
failures: list[str] = []

for judge in config.judges:
    cache_key = (judge.model, judge.trust_remote_code)
    if cache_key in seen:
        continue
    seen.add(cache_key)
    logger.info("loading %s", judge.model)
    tokenizer_config = {"trust_remote_code": True} if judge.trust_remote_code else None
    try:
        if tokenizer_config is None:
            load(judge.model)
        else:
            load(judge.model, tokenizer_config=tokenizer_config)
    except Exception as exc:
        failures.append(f"{judge.model}: {exc}")
        logger.exception("failed %s", judge.model)
        continue
    logger.info("loaded %s", judge.model)

if failures:
    logger.error("setup summary: failures detected")
    for failure in failures:
        logger.error("- %s", failure)
    raise SystemExit(1)

logger.info("setup summary: all pinned models loaded")
PY
}

preflight_mlx
if ! prefetch_models; then
  echo "MLX model setup failed." >&2
  echo "This usually means Metal could not be initialized from the current process." >&2
  echo "If you are on macOS with Apple Silicon, retry from a normal local Terminal session." >&2
  exit 1
fi
