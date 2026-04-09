"""Regression tests for wide-to-long IRT observation loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from src.models.irt_numpyro import load_matrix_observations, summarize_item_parameters


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


if __name__ == "__main__":
    unittest.main()
