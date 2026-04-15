"""Regression tests for posterior diagnostics helpers."""

from __future__ import annotations

import unittest

import numpy as np
from src.analysis.diagnostics import compute_rhat, diagnostic_group_rows, plot_diagnostics_summary


def _standard_rhat(flattened: np.ndarray) -> np.ndarray:
    """Compute standard R-hat for flattened chain draws."""

    chains, draws, features = flattened.shape
    chain_means = flattened.mean(axis=1)
    overall_mean = chain_means.mean(axis=0)
    between = draws * ((chain_means - overall_mean) ** 2).sum(axis=0) / (chains - 1)
    within = flattened.var(axis=1, ddof=1).mean(axis=0)
    variance_estimate = ((draws - 1) / draws) * within + between / draws
    ratio = np.divide(
        variance_estimate,
        within,
        out=np.full(features, np.nan, dtype=float),
        where=within != 0,
    )
    return np.sqrt(ratio)


class ComputeRhatTests(unittest.TestCase):
    """Verify split-R-hat semantics."""

    def test_compute_rhat_matches_manual_split(self) -> None:
        samples = np.asarray(
            [
                [[0.0], [1.0], [10.0], [11.0]],
                [[0.0], [1.0], [10.0], [11.0]],
            ]
        )

        expected = _standard_rhat(
            np.asarray(
                [
                    [[0.0], [1.0]],
                    [[0.0], [1.0]],
                    [[10.0], [11.0]],
                    [[10.0], [11.0]],
                ]
            )
        )

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

    def test_compute_rhat_returns_nan_for_zero_within_chain_variance(self) -> None:
        samples = np.asarray(
            [
                [[1.0], [1.0], [1.0], [1.0]],
                [[2.0], [2.0], [2.0], [2.0]],
            ]
        )

        result = compute_rhat(samples)

        self.assertTrue(np.isnan(result[0]))


if __name__ == "__main__":
    unittest.main()
