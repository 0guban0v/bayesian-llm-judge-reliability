"""Run Bayesian IRT inference for the configured experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.logging_utils import configure_logging
from src.models.irt_pymc import run_and_save_posterior
from src.schemas import ExperimentConfig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for inference."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    run_and_save_posterior(config)


if __name__ == "__main__":
    main()
