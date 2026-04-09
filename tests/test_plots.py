"""Regression tests for posterior predictive plot helpers."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl
from src.analysis.plots import (
    plot_posterior_predictive_check,
    posterior_predictive_judge_accuracy,
    validate_posterior_judge_order,
)


class PosteriorPredictiveJudgeAccuracyTests(unittest.TestCase):
    """Verify posterior predictive judge accuracy stays on the probability scale."""

    def test_returns_mean_probabilities_not_inverse_mean(self) -> None:
        posterior = {
            "theta": np.asarray([[[4.0, -4.0]]]),
            "b": np.asarray([[[0.0, 0.0]]]),
            "a": np.asarray([[[1.0, 1.0]]]),
        }

        predicted_mean, lower, upper = posterior_predictive_judge_accuracy(posterior)

        self.assertLess(predicted_mean[0], 1.0)
        self.assertGreater(predicted_mean[1], 0.0)
        np.testing.assert_allclose(predicted_mean, np.asarray([0.98201379, 0.01798621]))
        np.testing.assert_allclose(lower, predicted_mean)
        np.testing.assert_allclose(upper, predicted_mean)

    def test_validate_posterior_judge_order_accepts_matching_order(self) -> None:
        matrix_judge_ids = np.asarray(["judge-a", "judge-b"])
        posterior = {"judge_ids": np.asarray(["judge-a", "judge-b"])}

        validate_posterior_judge_order(matrix_judge_ids, posterior)

    def test_validate_posterior_judge_order_rejects_mismatch(self) -> None:
        matrix_judge_ids = np.asarray(["judge-a", "judge-b"])
        posterior = {"judge_ids": np.asarray(["judge-b", "judge-a"])}

        with self.assertRaisesRegex(
            ValueError,
            "Posterior judge order does not match matrix judge column order",
        ):
            validate_posterior_judge_order(matrix_judge_ids, posterior)

    def test_plot_posterior_predictive_check_rejects_misaligned_judges(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1"],
                "label": ["A>B"],
                "original_id": [1],
                "question": ["q1"],
                "source": ["s1"],
                "split": ["gpt"],
                "judge-a": [1],
                "judge-b": [0],
            }
        )
        posterior = {
            "judge_ids": np.asarray(["judge-b", "judge-a"]),
            "theta": np.asarray([[[1.0, -1.0]]]),
            "b": np.asarray([[[0.0]]]),
            "a": np.asarray([[[1.0]]]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "Posterior judge order does not match matrix judge column order",
        ):
            plot_posterior_predictive_check(matrix, posterior)


if __name__ == "__main__":
    unittest.main()
