#!/usr/bin/env bash
set -euo pipefail

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

uv run llmfit recommend --use-case reasoning --json
