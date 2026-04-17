"""Regression tests for posterior predictive plot helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
from src.analysis.figure_paths import (
    JUDGE_ACCURACY_PPC_STEM,
    JUDGE_RELIABILITY_BY_SOURCE_STEM,
    JUDGE_RELIABILITY_RIDGE_STEM,
)
from src.analysis.plot_config import JUDGE_COLOR_PINS, SOURCE_COLOR_PINS, judge_color_map, source_color_map
from src.analysis.plots import (
    cleanup_posterior_figure_outputs,
    plot_judge_accuracy_ppc,
    plot_judge_reliability_by_source,
    plot_judge_reliability_ridge,
    plot_prior_predictive_probabilities,
    sample_prior_predictive_probabilities,
    stable_sigmoid,
)
from src.analysis.posterior_utils import (
    judge_accuracy_ppc_summaries,
    observed_accuracy,
    ordered_source_ids,
    source_reliability_summary,
    top_source_ids,
    validate_judge_accuracy_ppc_summaries,
    validate_posterior_judge_order,
)


def make_plot_config(*, model_type: str = "2PL", variant: str = "source_hier") -> SimpleNamespace:
    """Return a minimal config namespace for plot helper tests."""

    return SimpleNamespace(
        experiment=SimpleNamespace(seed=7),
        judges=[SimpleNamespace(id="judge-a"), SimpleNamespace(id="judge-b")],
        model=SimpleNamespace(
            type=model_type,
            variant=variant,
            priors=SimpleNamespace(
                theta=SimpleNamespace(dist="normal", loc=0.0, scale=1.0),
                b=SimpleNamespace(dist="normal", loc=0.0, scale=2.0),
                a=SimpleNamespace(dist="lognormal", loc=0.0, scale=0.5),
                tau_theta=SimpleNamespace(dist="lognormal", loc=0.0, scale=0.5),
            ),
        ),
    )


class PosteriorPredictiveJudgeAccuracyTests(unittest.TestCase):
    """Verify posterior predictive plot helpers stay on the probability scale."""

    def test_prior_predictive_sampling_returns_probability_scale_outputs(self) -> None:
        matrix = pl.DataFrame({"item_id": ["item-1", "item-2"], "source": ["s1", "s2"]})
        config = make_plot_config()

        probabilities, judge_means = sample_prior_predictive_probabilities(
            matrix,
            config,
            num_draws=20,
        )

        self.assertEqual(probabilities.shape, (80,))
        self.assertEqual(judge_means.shape, (40,))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        self.assertTrue(np.all(judge_means >= 0.0))
        self.assertTrue(np.all(judge_means <= 1.0))

    def test_plot_prior_predictive_probabilities_returns_single_panel_figure(self) -> None:
        matrix = pl.DataFrame({"item_id": ["item-1", "item-2"], "source": ["s1", "s2"]})
        config = make_plot_config()

        figure = plot_prior_predictive_probabilities(matrix, config, num_draws=20)

        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(figure.axes[0].get_xlabel(), "Prior predictive judge mean accuracy")
        self.assertEqual(len(figure.axes[0].collections), 0)

    def test_prior_predictive_sampling_supports_global_variant(self) -> None:
        matrix = pl.DataFrame({"item_id": ["item-1", "item-2"], "source": ["s1", "s2"]})
        config = make_plot_config(variant="global")

        probabilities, judge_means = sample_prior_predictive_probabilities(
            matrix,
            config,
            num_draws=20,
        )

        self.assertEqual(probabilities.shape, (80,))
        self.assertEqual(judge_means.shape, (40,))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_prior_predictive_sampling_supports_1pl_without_discrimination_draws(self) -> None:
        matrix = pl.DataFrame({"item_id": ["item-1", "item-2"], "source": ["s1", "s2"]})
        config = make_plot_config(model_type="1PL", variant="global")

        probabilities, judge_means = sample_prior_predictive_probabilities(
            matrix,
            config,
            num_draws=20,
        )

        self.assertEqual(probabilities.shape, (80,))
        self.assertEqual(judge_means.shape, (40,))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_prior_predictive_sampling_honors_configured_prior_families(self) -> None:
        matrix = pl.DataFrame({"item_id": ["item-1", "item-2"], "source": ["s1", "s2"]})
        config = make_plot_config(variant="global")
        config.model.priors.theta.dist = "lognormal"
        config.model.priors.b.dist = "lognormal"

        probabilities, judge_means = sample_prior_predictive_probabilities(
            matrix,
            config,
            num_draws=20,
        )

        self.assertEqual(probabilities.shape, (80,))
        self.assertEqual(judge_means.shape, (40,))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all(np.isfinite(judge_means)))

    def test_stable_sigmoid_handles_extreme_logits(self) -> None:
        logits = np.asarray([-1_000.0, -50.0, 0.0, 50.0, 1_000.0])

        probabilities = stable_sigmoid(logits)

        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))
        self.assertLess(probabilities[0], 1e-10)
        self.assertGreater(probabilities[-1], 1.0 - 1e-10)

    def test_reads_saved_judge_accuracy_ppc_summaries(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "judge_accuracy_ppc_mean": np.asarray([0.8, 0.2]),
            "judge_accuracy_ppc_p05": np.asarray([0.7, 0.1]),
            "judge_accuracy_ppc_p95": np.asarray([0.9, 0.3]),
        }

        predicted_mean, lower, upper = judge_accuracy_ppc_summaries(posterior)

        np.testing.assert_allclose(predicted_mean, np.asarray([0.8, 0.2]))
        np.testing.assert_allclose(lower, np.asarray([0.7, 0.1]))
        np.testing.assert_allclose(upper, np.asarray([0.9, 0.3]))

    def test_rejects_missing_saved_judge_accuracy_ppc_summaries(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a"]),
            "judge_accuracy_ppc_mean": np.asarray([0.5]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "unsupported for PPC outputs",
        ):
            validate_judge_accuracy_ppc_summaries(posterior)

    def test_rejects_judge_accuracy_ppc_summary_length_mismatch(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "judge_accuracy_ppc_mean": np.asarray([0.5]),
            "judge_accuracy_ppc_p05": np.asarray([0.4]),
            "judge_accuracy_ppc_p95": np.asarray([0.6]),
        }

        with self.assertRaisesRegex(
            ValueError,
            "length does not match judge_ids length",
        ):
            validate_judge_accuracy_ppc_summaries(posterior)

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

    def test_observed_accuracy_preserves_matrix_column_order(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "label": ["A>B", "B>A"],
                "original_id": [1, 2],
                "question": ["q1", "q2"],
                "source": ["s1", "s1"],
                "split": ["gpt", "gpt"],
                "judge-b": [0, 1],
                "judge-a": [1, 0],
            }
        )

        judge_ids, accuracies = observed_accuracy(matrix)

        np.testing.assert_array_equal(judge_ids, np.asarray(["judge-b", "judge-a"]))
        np.testing.assert_allclose(accuracies, np.asarray([0.5, 0.5]))

    def test_plot_judge_accuracy_ppc_returns_single_axis_accuracy_figure(self) -> None:
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
            "theta": np.asarray([[[1.0, -1.0]]]),
            "judge_accuracy_ppc_mean": np.asarray([0.75, 0.25]),
            "judge_accuracy_ppc_p05": np.asarray([0.6, 0.1]),
            "judge_accuracy_ppc_p95": np.asarray([0.9, 0.4]),
        }

        figure = plot_judge_accuracy_ppc(matrix, posterior)

        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(figure.axes[0].get_xlabel(), "Accuracy")
        self.assertNotEqual(tuple(figure.axes[0].get_xlim()), (0.0, 1.0))

    def test_plot_judge_reliability_ridge_returns_single_axis_density_figure(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "theta": np.asarray([[[1.0, -1.0], [0.8, -0.8]]]),
        }

        figure = plot_judge_reliability_ridge(posterior)

        self.assertEqual(len(figure.axes), 1)
        self.assertEqual(figure.axes[0].get_xlabel(), "Posterior reliability (theta)")
        self.assertEqual(len(figure.legends), 0)

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
        expanded = judge_color_map(np.asarray(["deepseek-r1-distill-qwen-14b", "judge-x", "judge-y"]))

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
        axes = figure.axes[0]
        self.assertEqual(
            [tick.get_text() for tick in axes.get_xticklabels()],
            ["judge-a", "\njudge-b"],
        )
        self.assertEqual(
            [tick.get_text() for tick in axes.get_yticklabels()],
            ["source a", "source b"],
        )
        self.assertEqual(len(figure.axes), 2)

    def test_cleanup_posterior_figure_outputs_removes_stale_source_figure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            figures_dir = Path(temp_dir)
            ppc_path = figures_dir / f"{JUDGE_ACCURACY_PPC_STEM}.png"
            ridge_path = figures_dir / f"{JUDGE_RELIABILITY_RIDGE_STEM}.png"
            source_path = figures_dir / f"{JUDGE_RELIABILITY_BY_SOURCE_STEM}.png"
            ppc_path.write_bytes(b"ppc")
            ridge_path.write_bytes(b"ridge")
            source_path.write_bytes(b"source")

            cleanup_posterior_figure_outputs(
                figures_dir,
                keep_ridge=True,
                keep_source=False,
                keep_ppc=True,
            )

            self.assertTrue(ppc_path.exists())
            self.assertTrue(ridge_path.exists())
            self.assertFalse(source_path.exists())

    def test_plot_judge_reliability_by_source_uses_single_axis_for_one_source(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "label": ["A>B", "B>A"],
                "original_id": [1, 2],
                "question": ["q1", "q2"],
                "source": ["source-a", "source-a"],
                "split": ["gpt", "claude"],
                "judge-a": [1, 0],
            }
        )
        posterior = {
            "judge_ids": np.asarray(["judge-a"]),
            "source_ids": np.asarray(["source-a"]),
            "theta": np.asarray([[[0.6], [0.8]]]),
            "theta_source": np.asarray([[[[0.2], [0.4]]]]),
        }

        figure = plot_judge_reliability_by_source(matrix, posterior)

        axes = figure.axes[0]
        self.assertEqual(
            [tick.get_text() for tick in axes.get_yticklabels()],
            ["source a"],
        )


if __name__ == "__main__":
    unittest.main()
