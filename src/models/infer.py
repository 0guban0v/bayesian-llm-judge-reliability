"""Run Bayesian IRT inference for the configured experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.data.loader import load_judge_logs
from src.data.validate import assert_complete_judge_coverage, assert_complete_prompt_order_coverage, validate_items
from src.logging_utils import configure_logging, format_table_for_log
from src.models.irt_common import load_matrix_observations, save_posterior, summarize_item_parameters, summarize_judges
from src.models.irt_pymc import run_mcmc
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def run_and_save_posterior(
    config: ExperimentConfig,
    matrix: pl.DataFrame | None = None,
    items: pl.DataFrame | None = None,
    logs: pl.DataFrame | None = None,
) -> None:
    """Run inference and persist posterior samples."""

    config.ensure_directories()
    prepared_matrix = matrix if matrix is not None else pl.read_parquet(config.data.matrix_path)
    assert_complete_judge_coverage(prepared_matrix, [judge.id for judge in config.judges])
    prepared_items = items if items is not None else pl.read_parquet(config.data.item_path)
    validate_items(prepared_items)
    prepared_logs = logs if logs is not None else load_judge_logs(config.data.logs_dir)
    assert_complete_prompt_order_coverage(prepared_items, prepared_logs, config.judges)
    observations = load_matrix_observations(prepared_matrix)
    idata, samples, ppc_summary = run_mcmc(config, observations)
    output_path = config.inference.posterior_path
    inferencedata_path = config.inference.inferencedata_path
    inferencedata_path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(inferencedata_path)
    logger.info("saved_inferencedata=%s", inferencedata_path)
    save_posterior(
        output_path,
        samples,
        observations,
        config.model.type,
        metadata={
            "backend": np.asarray("pymc"),
            "experiment_seed": np.asarray(config.experiment.seed),
            "num_chains": np.asarray(samples["theta"].shape[0]),
            **ppc_summary,
        },
    )
    summary = summarize_judges(samples, observations["judge_ids"])
    item_summary = summarize_item_parameters(samples)
    logger.info("saved_posterior=%s", output_path)
    if logger.isEnabledFor(logging.INFO):
        logger.info("judge summary\n%s", format_table_for_log(summary))
        logger.info("item parameter summary\n%s", format_table_for_log(item_summary))


def main() -> None:
    """CLI entrypoint for inference."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    run_and_save_posterior(config)


if __name__ == "__main__":
    main()
