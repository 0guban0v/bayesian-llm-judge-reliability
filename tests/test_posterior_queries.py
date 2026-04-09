"""Regression tests for posterior comparison queries."""

from __future__ import annotations

import unittest

import numpy as np
from src.analysis.posterior_queries import (
    credible_interval,
    effect_size,
    probability_judge_a_exceeds_b,
    rank_judges,
    resolve_judge_indices,
)


class PosteriorQueryTests(unittest.TestCase):
    """Verify posterior query helpers provide user-facing errors."""

    def test_credible_interval_returns_equal_tailed_bounds(self) -> None:
        lower, upper = credible_interval(np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]), level=0.8)

        self.assertAlmostEqual(lower, 0.4)
        self.assertAlmostEqual(upper, 3.6)

    def test_probability_judge_a_exceeds_b_uses_strict_greater_than(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b", "judge-c"]),
            "theta": np.asarray(
                [
                    [
                        [2.0, 1.0, -1.0],
                        [0.0, 1.0, -2.0],
                        [3.0, 1.0, -3.0],
                        [1.0, 1.0, -4.0],
                    ]
                ]
            ),
        }

        probability = probability_judge_a_exceeds_b(posterior, "judge-a", "judge-b")

        self.assertEqual(probability, 0.5)

    def test_effect_size_returns_standardized_posterior_difference(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "theta": np.asarray([[[2.0, 1.0], [0.0, 1.0], [3.0, 1.0], [1.0, 1.0]]]),
        }

        standardized = effect_size(posterior, "judge-a", "judge-b")

        self.assertAlmostEqual(standardized, 0.4472135954999579)

    def test_rank_judges_sorts_by_posterior_mean_and_reports_intervals(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b", "judge-c"]),
            "theta": np.asarray(
                [
                    [
                        [2.0, 1.0, -1.0],
                        [0.0, 1.0, -2.0],
                        [3.0, 1.0, -3.0],
                        [1.0, 1.0, -4.0],
                    ]
                ]
            ),
        }

        ranking = rank_judges(posterior)

        self.assertEqual(
            ranking.get_column("judge_id").to_list(),
            ["judge-a", "judge-b", "judge-c"],
        )
        np.testing.assert_allclose(
            ranking.get_column("theta_mean").to_numpy(),
            np.asarray([1.5, 1.0, -2.5]),
        )
        np.testing.assert_allclose(
            ranking.get_column("theta_p05").to_numpy(),
            np.asarray([0.15, 1.0, -3.85]),
        )
        np.testing.assert_allclose(
            ranking.get_column("theta_p95").to_numpy(),
            np.asarray([2.85, 1.0, -1.15]),
        )

    def test_resolve_judge_indices_raises_clear_error_for_unknown_judge(self) -> None:
        posterior = {"judge_ids": np.asarray(["judge-a", "judge-b"])}

        with self.assertRaisesRegex(
            ValueError,
            "Unknown judge ID 'judge-c'. Available judge IDs: judge-a, judge-b",
        ):
            resolve_judge_indices(posterior, "judge-a", "judge-c")

    def test_probability_query_raises_value_error_for_unknown_judge(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "theta": np.asarray([[[0.1, -0.1]]]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "Unknown judge ID 'judge-c'. Available judge IDs: judge-a, judge-b",
        ):
            probability_judge_a_exceeds_b(posterior, "judge-a", "judge-c")


if __name__ == "__main__":
    unittest.main()
