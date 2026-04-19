"""Run the analysis pipeline and log outputs to local MLflow."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.analysis.diagnostics import (
    diagnostic_parameter_rows,
    plot_diagnostics_summary_rows,
    summarize_diagnostic_rows,
)
from src.analysis.diagnostics import (
    save_figure as save_diagnostics_figure,
)
from src.analysis.figure_paths import (
    DIAGNOSTICS_SUMMARY_STEM,
    figure_base_path,
)
from src.analysis.plots import (
    JUDGE_ACCURACY_PPC_STEM,
    JUDGE_RELIABILITY_BY_SOURCE_STEM,
    JUDGE_RELIABILITY_RIDGE_STEM,
    SEPARATION_STEM,
    TRACE_THETA_TAU_STEM,
    cleanup_posterior_figure_outputs,
    plot_judge_accuracy_ppc,
    plot_judge_reliability_by_source,
    plot_judge_reliability_ridge,
    plot_separation_by_judge,
    plot_trace_theta_tau_theta,
)
from src.analysis.plots import (
    save_figure as save_plot_figure,
)
from src.analysis.posterior_archive import load_posterior
from src.analysis.posterior_utils import (
    has_source_reliability,
    validate_posterior_plot_inputs,
)
from src.analysis.report_exports import (
    write_diagnostics_exports,
    write_results_exports,
)
from src.data.loader import build_and_write_matrix, load_or_prepare_items
from src.data.validate import assert_complete_judge_coverage, validate_items, validate_matrix
from src.judges.runner import run_all as run_judges
from src.logging_utils import configure_logging
from src.models.infer import run_and_save_posterior
from src.schemas import ExperimentConfig
from src.tracking import (
    log_artifact,
    log_config,
    log_data_artifacts,
    log_metrics,
    log_posterior_metrics,
    process_telemetry_metrics,
    tracked_run,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tracked analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    parser.add_argument(
        "--refresh-items",
        action="store_true",
        help="Force a fresh sample from JudgeBench instead of reusing cached parquet.",
    )
    return parser.parse_args()


def _save_figures(config: ExperimentConfig, matrix: pl.DataFrame, posterior: dict[str, np.ndarray]) -> None:
    """Generate all current analysis figures for a tracked run."""

    figures_dir = config.tracked_figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_rows = diagnostic_parameter_rows(posterior)
    divergence_count = int(np.asarray(posterior.get("diverging", np.array([]))).sum())
    save_diagnostics_figure(
        plot_diagnostics_summary_rows(diagnostic_rows, divergence_count),
        figure_base_path(figures_dir, DIAGNOSTICS_SUMMARY_STEM),
    )
    save_plot_figure(
        plot_judge_accuracy_ppc(matrix, posterior),
        figure_base_path(figures_dir, JUDGE_ACCURACY_PPC_STEM),
    )
    save_plot_figure(
        plot_judge_reliability_ridge(posterior),
        figure_base_path(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM),
    )
    save_plot_figure(
        plot_trace_theta_tau_theta(posterior),
        figure_base_path(figures_dir, TRACE_THETA_TAU_STEM),
    )
    save_plot_figure(
        plot_separation_by_judge(matrix, posterior),
        figure_base_path(figures_dir, SEPARATION_STEM),
    )
    if has_source_reliability(posterior):
        save_plot_figure(
            plot_judge_reliability_by_source(matrix, posterior, max_sources=config.analysis.plots.max_sources),
            figure_base_path(figures_dir, JUDGE_RELIABILITY_BY_SOURCE_STEM),
        )
    else:
        cleanup_posterior_figure_outputs(
            figures_dir,
            keep_ridge=True,
            keep_source=False,
            keep_ppc=True,
            keep_trace=True,
            keep_separation=True,
        )


def _log_diagnostic_metrics(posterior: dict[str, np.ndarray]) -> None:
    """Log compact diagnostics to MLflow."""

    rows = diagnostic_parameter_rows(posterior)
    divergence_count = int(np.asarray(posterior.get("diverging", np.array([]))).sum())
    summary = summarize_diagnostic_rows(rows, divergence_count)
    log_metrics(
        {
            "max_rhat": float(summary.get_column("rhat_max").max()),
            "min_ess": float(summary.get_column("ess_min").min()),
            "divergences": float(divergence_count),
        }
    )


def main() -> None:
    """CLI entrypoint for the tracked analysis pipeline."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    config.ensure_directories()
    started_at = time.perf_counter()
    with tracked_run(config):
        log_config(config)
        items = load_or_prepare_items(config, refresh=args.refresh_items)
        validate_items(items)
        run_judges(config, judge_id=None, limit=None)
        matrix = build_and_write_matrix(config, items)
        expected_judges = [judge.id for judge in config.judges]
        validate_matrix(matrix, expected_judges)
        assert_complete_judge_coverage(matrix, expected_judges)
        log_data_artifacts(config)
        run_and_save_posterior(config, matrix)
        posterior = load_posterior(config.inference.posterior_path)
        validate_posterior_plot_inputs(matrix, posterior)
        _save_figures(config, matrix, posterior)
        output_dir = config.tracked_report_generated_dir
        write_results_exports(config, matrix, posterior, output_dir=output_dir)
        write_diagnostics_exports(config, posterior, output_dir=output_dir)
        _log_diagnostic_metrics(posterior)
        log_posterior_metrics(config, posterior)
        for artifact_name in (
            "judge_summary.tex",
            "pairwise_summary.tex",
            "diagnostics_summary.tex",
        ):
            log_artifact(output_dir / artifact_name, "report_generated")
        figure_names = [
            "diagnostics_summary.png",
            "judge_accuracy_ppc.png",
            "judge_reliability_ridge.png",
            "trace_theta_tau_theta.png",
            "separation_by_judge.png",
        ]
        if has_source_reliability(posterior):
            figure_names.append("judge_reliability_by_source.png")
        for figure_name in figure_names:
            log_artifact(config.tracked_figures_dir / figure_name, "figures")
        log_metrics(
            {
                "tracked_run_duration_seconds": time.perf_counter() - started_at,
                **process_telemetry_metrics(),
            }
        )
        logger.info("completed tracked analysis run for %s", config.experiment.name)


if __name__ == "__main__":
    main()
