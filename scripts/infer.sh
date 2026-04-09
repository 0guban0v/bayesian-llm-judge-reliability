#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

CONFIG_PATH="${1:-configs/experiment.yaml}"

backend=$(uv run python -c "
import sys, logging
logging.disable(logging.CRITICAL)
from src.schemas import ExperimentConfig
config = ExperimentConfig.from_yaml(sys.argv[1])
sys.stdout.write(config.inference.backend)
" "${CONFIG_PATH}")

if [ "${backend}" = "blackjax" ]; then
  uv run python -m src.models.irt_blackjax --config "${CONFIG_PATH}"
else
  uv run python -m src.models.irt_numpyro --config "${CONFIG_PATH}"
fi
