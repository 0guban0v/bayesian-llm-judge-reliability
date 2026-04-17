"""Regression tests for wide-to-long IRT observation loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from src.models.irt_common import (
    build_model_priors,
    load_matrix_observations,
    summarize_item_parameters,
)
from src.models.irt_pymc import run_mcmc
from src.schemas import ExperimentConfig, PriorConfig


class LoadMatrixObservationsTests(unittest.TestCase):
    """Verify Polars-based observation loading preserves indexing semantics."""

    def test_load_matrix_observations_preserves_item_and_judge_order(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2", "item-3"],
                "label": ["A>B", "B>A", "A>B"],
                "original_id": [1, 2, 3],
                "question": ["q1", "q2", "q3"],
                "source": ["s1", "s2", "s3"],
                "split": ["gpt", "gpt", "claude"],
                "judge-a": [1, None, 0],
                "judge-b": [0, 1, None],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            matrix_path = Path(temp_dir) / "matrix.parquet"
            matrix.write_parquet(matrix_path)
            observations = load_matrix_observations(matrix_path)

        np.testing.assert_array_equal(observations["correct"], np.asarray([1, 0, 1, 0]))
        np.testing.assert_array_equal(observations["judge_idx"], np.asarray([0, 1, 1, 0]))
        np.testing.assert_array_equal(observations["item_idx"], np.asarray([0, 0, 1, 2]))
        np.testing.assert_array_equal(
            observations["judge_ids"],
            np.asarray(["judge-a", "judge-b"]),
        )
        np.testing.assert_array_equal(
            observations["item_ids"],
            np.asarray(["item-1", "item-2", "item-3"]),
        )
        self.assertEqual(observations["n_judges"], 2)
        self.assertEqual(observations["n_items"], 3)
        np.testing.assert_array_equal(observations["source_ids"], np.asarray(["s1", "s2", "s3"]))
        np.testing.assert_array_equal(observations["source_idx"], np.asarray([0, 0, 1, 2]))


class SummarizeItemParametersTests(unittest.TestCase):
    """Verify compact item-parameter summaries for CLI output."""

    def test_summarize_item_parameters_includes_b_and_a_for_2pl(self) -> None:
        samples = {
            "b": np.asarray([[[0.0, 1.0], [1.0, 2.0]]]),
            "a": np.asarray([[[1.0, 2.0], [2.0, 3.0]]]),
        }

        summary = summarize_item_parameters(samples)

        self.assertEqual(summary.get_column("parameter").to_list(), ["b", "a"])

    def test_summarize_item_parameters_includes_only_b_for_1pl(self) -> None:
        samples = {"b": np.asarray([[[0.0, 1.0], [1.0, 2.0]]])}

        summary = summarize_item_parameters(samples)

        self.assertEqual(summary.get_column("parameter").to_list(), ["b"])


class SourceHierModelTests(unittest.TestCase):
    """Verify the source-aware hierarchical PyMC variant runs and returns new parameters."""

    def test_run_mcmc_supports_source_hier_variant(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml").model_copy(deep=True)
        config.inference.num_warmup = 2
        config.inference.num_samples = 4
        config.inference.num_chains = 1
        config.model.variant = "source_hier"
        config.model.priors.tau_theta = PriorConfig(dist="lognormal", loc=0.0, scale=0.5)
        observations = {
            "correct": np.asarray([1, 0, 1, 0], dtype=np.int32),
            "judge_idx": np.asarray([0, 0, 1, 1], dtype=np.int32),
            "item_idx": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "source_idx": np.asarray([0, 1, 0, 1], dtype=np.int32),
            "n_judges": 2,
            "n_items": 2,
            "n_sources": 2,
            "judge_ids": np.asarray(["judge-a", "judge-b"]),
            "item_ids": np.asarray(["item-1", "item-2"]),
            "source_ids": np.asarray(["source-a", "source-b"]),
        }

        _, samples, ppc_summary = run_mcmc(config, observations)

        self.assertIn("theta", samples)
        self.assertIn("tau_theta", samples)
        self.assertIn("theta_source", samples)
        self.assertIn("judge_accuracy_ppc_mean", ppc_summary)


class BuildModelPriorsTests(unittest.TestCase):
    """Verify configured prior distribution families are preserved."""

    def test_build_model_priors_preserves_distribution_names(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        priors = build_model_priors(config.model)

        self.assertEqual(priors.theta.dist, "normal")
        self.assertEqual(priors.b.dist, "normal")
        self.assertEqual(priors.a.dist, "lognormal")
        self.assertIsNotNone(priors.tau_theta)
        self.assertEqual(priors.tau_theta.dist, "lognormal")


if __name__ == "__main__":
    unittest.main()
