"""Regression tests for tracked analysis orchestration."""

from __future__ import annotations

import unittest
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from src.analysis import run_tracked_analysis
from src.schemas import ExperimentConfig


class RunTrackedAnalysisTests(unittest.TestCase):
    """Verify tracked analysis uses one item sample through judge and matrix stages."""

    def test_refresh_items_rebuilds_matrix_from_refreshed_items(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment_gpt_global.yaml")
        items = MagicMock(name="items")
        matrix = MagicMock(name="matrix")
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "theta": np.asarray([[[0.1, -0.1], [0.2, -0.2]]]),
            "theta_source": np.asarray([[[[0.1], [-0.1]], [[0.2], [-0.2]]]]),
            "source_ids": np.asarray(["source-a"]),
            "judge_accuracy_ppc_mean": np.asarray([0.5, 0.5]),
            "judge_accuracy_ppc_p05": np.asarray([0.4, 0.4]),
            "judge_accuracy_ppc_p95": np.asarray([0.6, 0.6]),
            "diverging": np.asarray([0, 0]),
        }
        log_artifact = MagicMock()

        with (
            patch.object(
                run_tracked_analysis,
                "parse_args",
                return_value=Namespace(config=Path("configs/experiment_gpt_global.yaml"), refresh_items=True),
            ),
            patch.object(run_tracked_analysis, "tracked_run", return_value=nullcontext()),
            patch.object(run_tracked_analysis, "log_config"),
            patch.object(run_tracked_analysis, "load_or_prepare_items", return_value=items) as load_items,
            patch.object(run_tracked_analysis, "validate_items") as validate_items,
            patch.object(run_tracked_analysis, "run_judges") as run_judges,
            patch.object(run_tracked_analysis, "build_and_write_matrix", return_value=matrix) as build_matrix,
            patch.object(run_tracked_analysis, "validate_matrix") as validate_matrix,
            patch.object(run_tracked_analysis, "assert_complete_judge_coverage") as assert_complete_judge_coverage,
            patch.object(run_tracked_analysis, "log_data_artifacts"),
            patch.object(run_tracked_analysis, "run_and_save_posterior"),
            patch.object(run_tracked_analysis, "load_posterior", return_value=posterior),
            patch.object(run_tracked_analysis, "validate_posterior_plot_inputs"),
            patch.object(run_tracked_analysis, "_save_figures"),
            patch.object(run_tracked_analysis, "write_results_exports"),
            patch.object(run_tracked_analysis, "write_diagnostics_exports"),
            patch.object(run_tracked_analysis, "write_case_exports"),
            patch.object(run_tracked_analysis, "_log_diagnostic_metrics"),
            patch.object(run_tracked_analysis, "log_posterior_metrics"),
            patch.object(run_tracked_analysis, "log_metrics"),
            patch.object(run_tracked_analysis, "log_artifact", log_artifact),
        ):
            run_tracked_analysis.main()

        load_items.assert_called_once_with(config, refresh=True)
        validate_items.assert_called_once_with(items)
        run_judges.assert_called_once_with(config, judge_id=None, limit=None)
        build_matrix.assert_called_once_with(config, items)
        expected_judges = [judge.id for judge in config.judges]
        validate_matrix.assert_called_once_with(matrix, expected_judges)
        assert_complete_judge_coverage.assert_called_once_with(matrix, expected_judges)
        logged_paths = [call.args[0].name for call in log_artifact.call_args_list]
        self.assertIn("judge_reliability_by_source.png", logged_paths)

    def test_non_source_posterior_does_not_log_source_reliability_figure(self) -> None:
        items = MagicMock(name="items")
        matrix = MagicMock(name="matrix")
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "theta": np.asarray([[[0.1, -0.1], [0.2, -0.2]]]),
            "judge_accuracy_ppc_mean": np.asarray([0.5, 0.5]),
            "judge_accuracy_ppc_p05": np.asarray([0.4, 0.4]),
            "judge_accuracy_ppc_p95": np.asarray([0.6, 0.6]),
            "diverging": np.asarray([0, 0]),
        }
        log_artifact = MagicMock()

        with (
            patch.object(
                run_tracked_analysis,
                "parse_args",
                return_value=Namespace(config=Path("configs/experiment_gpt_global.yaml"), refresh_items=False),
            ),
            patch.object(run_tracked_analysis, "tracked_run", return_value=nullcontext()),
            patch.object(run_tracked_analysis, "log_config"),
            patch.object(run_tracked_analysis, "load_or_prepare_items", return_value=items),
            patch.object(run_tracked_analysis, "validate_items"),
            patch.object(run_tracked_analysis, "run_judges"),
            patch.object(run_tracked_analysis, "build_and_write_matrix", return_value=matrix),
            patch.object(run_tracked_analysis, "validate_matrix"),
            patch.object(run_tracked_analysis, "assert_complete_judge_coverage"),
            patch.object(run_tracked_analysis, "log_data_artifacts"),
            patch.object(run_tracked_analysis, "run_and_save_posterior"),
            patch.object(run_tracked_analysis, "load_posterior", return_value=posterior),
            patch.object(run_tracked_analysis, "validate_posterior_plot_inputs"),
            patch.object(run_tracked_analysis, "_save_figures"),
            patch.object(run_tracked_analysis, "write_results_exports"),
            patch.object(run_tracked_analysis, "write_diagnostics_exports"),
            patch.object(run_tracked_analysis, "write_case_exports"),
            patch.object(run_tracked_analysis, "_log_diagnostic_metrics"),
            patch.object(run_tracked_analysis, "log_posterior_metrics"),
            patch.object(run_tracked_analysis, "log_metrics"),
            patch.object(run_tracked_analysis, "log_artifact", log_artifact),
        ):
            run_tracked_analysis.main()

        logged_paths = [call.args[0].name for call in log_artifact.call_args_list]
        self.assertNotIn("judge_reliability_by_source.png", logged_paths)


if __name__ == "__main__":
    unittest.main()
