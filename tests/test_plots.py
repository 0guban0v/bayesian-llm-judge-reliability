"""Regression tests for posterior predictive plot helpers."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl
from src.analysis.plots import (
    JUDGE_COLOR_PINS,
    SOURCE_COLOR_PINS,
    judge_color_map,
    plot_posterior_predictive_check,
    posterior_predictive_judge_accuracy,
    source_color_map,
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

    def test_uses_all_available_draws_for_predictive_intervals(self) -> None:
        early_draws = np.full((240, 1), -4.0)
        late_draws = np.full((20, 1), 4.0)
        posterior = {
            "theta": np.asarray([np.vstack([early_draws, late_draws])]),
            "b": np.asarray([np.zeros((260, 1))]),
            "a": np.asarray([np.ones((260, 1))]),
        }

        predicted_mean, _, upper = posterior_predictive_judge_accuracy(posterior)

        self.assertGreater(predicted_mean[0], 0.09)
        self.assertGreater(upper[0], 0.9)

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

    def test_judge_color_map_uses_pinned_colors_for_known_models(self) -> None:
        judge_ids = np.asarray(
            [
                "deepseek-r1-distill-qwen-14b",
                "deepseek-r1-distill-qwen-7b",
                "mistral-7b-instruct-v0-3",
                "qwen2-5-7b-instruct",
                "gemma-2-9b-it",
            ]
        )

        color_map = judge_color_map(judge_ids)

        self.assertEqual(color_map, JUDGE_COLOR_PINS)

    def test_judge_color_map_fallback_is_deterministic_and_does_not_drift(self) -> None:
        baseline = judge_color_map(np.asarray(["deepseek-r1-distill-qwen-14b", "judge-x"]))
        expanded = judge_color_map(
            np.asarray(["deepseek-r1-distill-qwen-14b", "judge-x", "judge-y"])
        )

        self.assertEqual(
            baseline["deepseek-r1-distill-qwen-14b"],
            expanded["deepseek-r1-distill-qwen-14b"],
        )
        self.assertEqual(baseline["judge-x"], expanded["judge-x"])
        self.assertRegex(baseline["judge-x"], r"^#[0-9a-f]{6}$")

    def test_source_color_map_uses_pinned_colors_for_known_sources(self) -> None:
        source_ids = list(SOURCE_COLOR_PINS)

        color_map = source_color_map(source_ids)

        self.assertEqual(color_map, SOURCE_COLOR_PINS)

    def test_source_color_map_fallback_is_deterministic_and_does_not_drift(self) -> None:
        baseline = source_color_map(["livebench-reasoning", "source-x"])
        expanded = source_color_map(["livebench-reasoning", "source-x", "source-y"])

        self.assertEqual(baseline["livebench-reasoning"], expanded["livebench-reasoning"])
        self.assertEqual(baseline["source-x"], expanded["source-x"])
        self.assertRegex(baseline["source-x"], r"^#[0-9a-f]{6}$")


if __name__ == "__main__":
    unittest.main()
