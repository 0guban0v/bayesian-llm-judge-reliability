"""Regression tests for posterior diagnostics helpers."""

from __future__ import annotations

import unittest

import numpy as np
from src.analysis.diagnostics import compute_rhat


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
