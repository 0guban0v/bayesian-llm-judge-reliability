"""Run Bayesian IRT inference for the configured experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.data.loader import build_analysis_table, load_judge_logs
from src.data.validate import (
    assert_analysis_matches_current_items,
    assert_complete_original_choice_coverage,
    assert_complete_prompt_order_coverage,
    validate_items,
)
from src.logging_utils import configure_logging, format_table_for_log
from src.models.irt_common import (
    load_analysis_observations,
    save_posterior,
    summarize_item_parameters,
    summarize_judges,
)
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
    analysis: pl.DataFrame | None = None,
    items: pl.DataFrame | None = None,
    logs: pl.DataFrame | None = None,
) -> None:
    """Run inference and persist posterior samples."""

    config.ensure_directories()
    prepared_items = items if items is not None else pl.read_parquet(config.data.item_path)
    validate_items(prepared_items)
    prepared_logs = logs if logs is not None else load_judge_logs(config.data.logs_dir)
    assert_complete_prompt_order_coverage(prepared_items, prepared_logs, config.judges)
    loaded_analysis_artifact = analysis is None
    prepared_analysis = pl.read_parquet(config.data.analysis_path) if loaded_analysis_artifact else analysis
    assert prepared_analysis is not None
    assert_analysis_matches_current_items(prepared_items, prepared_analysis, config.judges)
    if loaded_analysis_artifact:
        expected_analysis = build_analysis_table(
            prepared_items,
            prepared_logs,
            repeat_policy=config.data.repeat_policy,
        )
        task_order = ["item_key", "judge_id", "prompt_order"]
        if not prepared_analysis.sort(task_order).equals(expected_analysis.sort(task_order)):
            raise ValueError(
                "Analysis table does not match current resolved judge logs. "
                "Rebuild the analysis artifacts before running inference."
            )
    assert_complete_original_choice_coverage(prepared_analysis, config.judges)
    observations = load_analysis_observations(
        prepared_analysis,
        judge_ids=[judge.id for judge in config.judges],
        item_ids=prepared_items.get_column("item_key").to_list(),
    )
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
