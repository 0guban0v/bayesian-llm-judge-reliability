"""Regression tests for posterior diagnostics helpers."""

from __future__ import annotations

import unittest

import arviz as az
import numpy as np
from src.analysis.diagnostics import (
    compute_ess,
    compute_rhat,
    diagnostic_group_rows,
    plot_diagnostics_summary,
)


class ComputeRhatTests(unittest.TestCase):
    """Verify split-R-hat semantics."""

    def test_compute_rhat_matches_arviz_split(self) -> None:
        samples = np.asarray(
            [
                [[0.0], [1.0], [10.0], [11.0]],
                [[0.0], [1.0], [10.0], [11.0]],
            ]
        )

        expected = np.asarray(
            az.rhat(az.convert_to_dataset({"parameter": samples}), method="split")["parameter"]
        ).reshape(-1)

        np.testing.assert_allclose(compute_rhat(samples), expected)
        self.assertGreater(float(compute_rhat(samples)[0]), 1.0)

    def test_compute_rhat_returns_nan_when_split_draws_too_short(self) -> None:
        samples = np.asarray([[[0.0], [1.0]], [[0.0], [1.0]]])

        result = compute_rhat(samples)

        self.assertTrue(np.isnan(result[0]))

    def test_diagnostic_group_rows_orders_named_parameters(self) -> None:
        posterior = {
            "theta": np.ones((2, 4, 5)),
            "b": np.ones((2, 4, 3)),
            "a": np.ones((2, 4, 3)),
            "tau_theta": np.ones((2, 4, 5)),
            "theta_source": np.ones((2, 4, 10)),
        }

        rows = diagnostic_group_rows(posterior)

        self.assertEqual(
            [row["parameter"] for row in rows],
            ["theta", "b", "a", "tau_theta", "theta_source"],
        )
        self.assertEqual(
            [row["label"] for row in rows][:4],
            ["θ\n(5 judges)", "b\n(3 items)", "a\n(3 items)", "τ_θ\n(5 judges)"],
        )

    def test_diagnostic_group_rows_ignores_unexpected_posterior_keys(self) -> None:
        posterior = {
            "theta": np.ones((2, 4, 5)),
            "b": np.ones((2, 4, 3)),
            "log_likelihood": np.ones((2, 4, 3)),
            "diverging_detail": np.ones((2, 4, 3)),
        }

        rows = diagnostic_group_rows(posterior)

        self.assertEqual(
            [row["parameter"] for row in rows],
            ["theta", "b"],
        )

    def test_plot_diagnostics_summary_returns_two_panel_figure(self) -> None:
        posterior = {
            "theta": np.random.default_rng(0).normal(size=(2, 8, 5)),
            "b": np.random.default_rng(1).normal(size=(2, 8, 3)),
            "a": np.abs(np.random.default_rng(2).normal(size=(2, 8, 3))) + 0.5,
            "tau_theta": np.abs(np.random.default_rng(3).normal(size=(2, 8, 5))) + 0.2,
            "diverging": np.zeros(16, dtype=int),
        }

        figure = plot_diagnostics_summary(posterior)

        self.assertEqual(len(figure.axes), 2)
        self.assertEqual(figure.axes[0].get_xlabel(), "R̂")
        self.assertEqual(figure.axes[1].get_xlabel(), "ESS")

    def test_compute_rhat_matches_arviz_for_zero_within_chain_variance(self) -> None:
        samples = np.asarray(
            [
                [[1.0], [1.0], [1.0], [1.0]],
                [[2.0], [2.0], [2.0], [2.0]],
            ]
        )

        expected = np.asarray(
            az.rhat(az.convert_to_dataset({"parameter": samples}), method="split")["parameter"]
        ).reshape(-1)

        np.testing.assert_allclose(compute_rhat(samples), expected)

    def test_compute_ess_matches_arviz_bulk(self) -> None:
        samples = np.random.default_rng(7).normal(size=(2, 16, 3))

        expected = np.asarray(
            az.ess(az.convert_to_dataset({"parameter": samples}), method="bulk")["parameter"]
        ).reshape(-1)

        np.testing.assert_allclose(compute_ess(samples), expected)


if __name__ == "__main__":
    unittest.main()
