"""Regression tests for posterior predictive plot helpers."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl
from src.analysis.plots import (
    JUDGE_COLOR_PINS,
    SOURCE_COLOR_PINS,
    judge_color_map,
    ordered_source_ids,
    plot_judge_reliability_by_source,
    plot_posterior_predictive_check,
    posterior_predictive_judge_accuracy,
    source_color_map,
    source_reliability_summary,
    top_source_ids,
    validate_posterior_judge_order,
)


class PosteriorPredictiveJudgeAccuracyTests(unittest.TestCase):
    """Verify posterior predictive judge accuracy stays on the probability scale."""

    def test_returns_mean_probabilities_not_inverse_mean(self) -> None:
        matrix = pl.DataFrame({"item_id": ["item-1", "item-2"], "source": ["s1", "s1"]})
        posterior = {
            "item_ids": np.asarray(["item-1", "item-2"]),
            "theta": np.asarray([[[4.0, -4.0]]]),
            "b": np.asarray([[[0.0, 0.0]]]),
            "a": np.asarray([[[1.0, 1.0]]]),
        }

        predicted_mean, lower, upper = posterior_predictive_judge_accuracy(matrix, posterior)

        self.assertLess(predicted_mean[0], 1.0)
        self.assertGreater(predicted_mean[1], 0.0)
        np.testing.assert_allclose(predicted_mean, np.asarray([0.98201379, 0.01798621]))
        np.testing.assert_allclose(lower, predicted_mean)
        np.testing.assert_allclose(upper, predicted_mean)

    def test_uses_all_available_draws_for_predictive_intervals(self) -> None:
        matrix = pl.DataFrame({"item_id": ["item-1"], "source": ["s1"]})
        early_draws = np.full((240, 1), -4.0)
        late_draws = np.full((20, 1), 4.0)
        posterior = {
            "item_ids": np.asarray(["item-1"]),
            "theta": np.asarray([np.vstack([early_draws, late_draws])]),
            "b": np.asarray([np.zeros((260, 1))]),
            "a": np.asarray([np.ones((260, 1))]),
        }

        predicted_mean, _, upper = posterior_predictive_judge_accuracy(matrix, posterior)

        self.assertGreater(predicted_mean[0], 0.09)
        self.assertGreater(upper[0], 0.9)

    def test_uses_theta_source_for_source_hier_predictive_accuracy(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "source": ["source-a", "source-b"],
            }
        )
        posterior = {
            "item_ids": np.asarray(["item-1", "item-2"]),
            "source_ids": np.asarray(["source-a", "source-b"]),
            "theta": np.asarray([[[0.0]]]),
            "theta_source": np.asarray([[[[3.0, -3.0]]]]),
            "b": np.asarray([[[0.0, 0.0]]]),
            "a": np.asarray([[[1.0, 1.0]]]),
        }

        predicted_mean, lower, upper = posterior_predictive_judge_accuracy(matrix, posterior)

        np.testing.assert_allclose(predicted_mean, np.asarray([0.5]), atol=1e-6)
        np.testing.assert_allclose(lower, predicted_mean)
        np.testing.assert_allclose(upper, predicted_mean)

    def test_rejects_item_id_mismatch_between_matrix_and_posterior(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "source": ["s1", "s1"],
            }
        )
        posterior = {
            "item_ids": np.asarray(["item-1", "item-x"]),
            "theta": np.asarray([[[1.0]]]),
            "b": np.asarray([[[0.0, 0.0]]]),
            "a": np.asarray([[[1.0, 1.0]]]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "Posterior item_ids do not match matrix item_id order",
        ):
            posterior_predictive_judge_accuracy(matrix, posterior)

    def test_rejects_missing_source_mapping_for_source_hier_posterior(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1"],
                "source": ["source-missing"],
            }
        )
        posterior = {
            "item_ids": np.asarray(["item-1"]),
            "source_ids": np.asarray(["source-a"]),
            "theta_source": np.asarray([[[[0.0]]]]),
            "theta": np.asarray([[[0.0]]]),
            "b": np.asarray([[[0.0]]]),
            "a": np.asarray([[[1.0]]]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "Matrix sources are missing from posterior source_ids",
        ):
            posterior_predictive_judge_accuracy(matrix, posterior)

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

    def test_plot_posterior_predictive_check_uses_single_judge_legend(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "label": ["A>B", "B>A"],
                "original_id": [1, 2],
                "question": ["q1", "q2"],
                "source": ["s1", "s1"],
                "split": ["gpt", "gpt"],
                "judge-a": [1, 0],
                "judge-b": [0, 1],
            }
        )
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2"]),
            "theta": np.asarray([[[1.0, -1.0]]]),
            "b": np.asarray([[[0.0, 0.0]]]),
            "a": np.asarray([[[1.0, 1.0]]]),
        }

        figure = plot_posterior_predictive_check(matrix, posterior)

        self.assertEqual(len(figure.legends), 1)
        self.assertEqual(figure.legends[0].get_title().get_text(), "Judges")

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

    def test_ordered_source_ids_prefers_high_count_sources(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2", "item-3", "item-4"],
                "label": ["A>B", "A>B", "B>A", "B>A"],
                "original_id": [1, 2, 3, 4],
                "question": ["q1", "q2", "q3", "q4"],
                "source": ["source-b", "source-a", "source-a", "source-c"],
                "split": ["gpt", "gpt", "claude", "claude"],
            }
        )
        posterior = {"source_ids": np.asarray(["source-c", "source-a", "source-b"])}

        ordered = ordered_source_ids(matrix, posterior)

        self.assertEqual(ordered, ["source-a", "source-c", "source-b"])

    def test_top_source_ids_limits_small_multiples_to_eight_sources(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": [f"item-{index}" for index in range(9)],
                "label": ["A>B"] * 9,
                "original_id": list(range(9)),
                "question": [f"q{index}" for index in range(9)],
                "source": [f"source-{index}" for index in range(9)],
                "split": ["gpt"] * 9,
            }
        )
        posterior = {"source_ids": np.asarray([f"source-{index}" for index in range(9)])}

        limited = top_source_ids(matrix, posterior)

        self.assertEqual(len(limited), 8)
        self.assertNotIn("source-8", limited)

    def test_source_reliability_summary_and_plot_use_source_order(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2", "item-3"],
                "label": ["A>B", "B>A", "A>B"],
                "original_id": [1, 2, 3],
                "question": ["q1", "q2", "q3"],
                "source": ["source-b", "source-a", "source-a"],
                "split": ["gpt", "gpt", "claude"],
                "judge-a": [1, 1, 0],
                "judge-b": [0, 1, 1],
            }
        )
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "source_ids": np.asarray(["source-b", "source-a"]),
            "theta": np.asarray([[[0.6, 0.2], [0.8, 0.4]]]),
            "theta_source": np.asarray(
                [
                    [
                        [[0.2, 0.8], [0.0, 0.6]],
                        [[0.4, 1.0], [0.2, 0.8]],
                    ]
                ]
            ),
        }

        ordered = ordered_source_ids(matrix, posterior)
        summary = source_reliability_summary(posterior, ordered)
        figure = plot_judge_reliability_by_source(matrix, posterior)

        self.assertEqual(ordered, ["source-a", "source-b"])
        self.assertEqual(summary.height, 4)
        self.assertEqual(summary.get_column("source").to_list()[:2], ["source-a", "source-a"])
        visible_axes = [axis for axis in figure.axes if axis.get_visible()]
        axes = visible_axes[0]
        self.assertEqual(
            [tick.get_text() for tick in axes.get_yticklabels()],
            ["judge-a", "judge-b"],
        )
        self.assertEqual(axes.get_title(), "source-a (n=2)")
        self.assertEqual(visible_axes[1].get_title(), "source-b (n=1)")
        self.assertEqual(len(figure.legends), 1)


if __name__ == "__main__":
    unittest.main()
