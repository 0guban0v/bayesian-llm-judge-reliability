"""Regression tests for shared matrix semantics helpers."""

from __future__ import annotations

import unittest

import polars as pl
from src.data.matrix_semantics import (
    first_original_judgments,
    judge_columns,
    observed_accuracy_frame,
    pivot_original_judgments,
    summarize_matrix,
)


class MatrixSemanticsTests(unittest.TestCase):
    """Verify shared log and matrix helper behavior."""

    def test_first_original_judgments_keep_first_duplicate_in_log_order(self) -> None:
        logs = pl.DataFrame(
            {
                "item_id": ["item-1", "item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a", "judge-b"],
                "prompt_order": ["original", "original", "reversed"],
                "correct": [True, False, True],
            }
        )

        first = first_original_judgments(logs)

        self.assertEqual(first.sort(["item_id", "judge_id"])["correct_int"].to_list(), [1])

    def test_pivot_original_judgments_returns_wide_matrix(self) -> None:
        first = pl.DataFrame(
            {
                "item_id": ["item-1", "item-1", "item-2"],
                "judge_id": ["judge-a", "judge-b", "judge-a"],
                "correct_int": [1, 0, 1],
            }
        )

        pivoted = pivot_original_judgments(first).sort("item_id")

        self.assertEqual(pivoted.columns, ["item_id", "judge-a", "judge-b"])
        self.assertEqual(pivoted["judge-a"].to_list(), [1, 1])
        self.assertEqual(pivoted["judge-b"].to_list(), [0, None])

    def test_judge_columns_and_summary_skip_metadata(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "original_id": [1, 2],
                "split": ["gpt", "claude"],
                "source": ["source-a", "source-b"],
                "question": ["question-a", "question-b"],
                "label": ["A>B", "B>A"],
                "judge-a": [1, None],
                "judge-b": [0, 1],
            }
        )

        self.assertEqual(judge_columns(matrix), ["judge-a", "judge-b"])
        summary = summarize_matrix(matrix)
        self.assertEqual(summary.get_column("judge_id").to_list(), ["judge-a", "judge-b"])
        self.assertEqual(summary.get_column("responded_items").to_list(), [1, 2])
        self.assertEqual(summary.get_column("accuracy").to_list(), [1.0, 0.5])

    def test_observed_accuracy_frame_preserves_requested_order(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "original_id": [1, 2],
                "split": ["gpt", "claude"],
                "source": ["source-a", "source-b"],
                "question": ["question-a", "question-b"],
                "label": ["A>B", "B>A"],
                "judge-a": [1, 0],
                "judge-b": [0, 1],
            }
        )

        observed = observed_accuracy_frame(matrix, ["judge-b", "judge-a"])

        self.assertEqual(observed.get_column("judge_id").to_list(), ["judge-b", "judge-a"])
        self.assertEqual(observed.get_column("accuracy").to_list(), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
