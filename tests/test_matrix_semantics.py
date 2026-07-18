"""Regression tests for shared matrix semantics helpers."""

from __future__ import annotations

import unittest

import polars as pl
from src.data.matrix_semantics import (
    judge_columns,
    observed_accuracy_frame,
    pivot_original_judgments,
    resolve_original_judgments,
    summarize_matrix,
)


class MatrixSemanticsTests(unittest.TestCase):
    """Verify shared log and matrix helper behavior."""

    def duplicate_logs(self) -> pl.DataFrame:
        """Return conflicting repeats plus unrelated reversed judgment."""

        return pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1", "gpt:item-1"],
                "item_id": ["item-1", "item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a", "judge-b"],
                "prompt_order": ["original", "original", "reversed"],
                "repeat_index": [0, 1, 0],
                "correct": [True, False, True],
            }
        )

    def test_reject_policy_fails_on_duplicate(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Duplicate judgment: item_key=gpt:item-1 judge_id=judge-a prompt_order=original count=2",
        ):
            resolve_original_judgments(self.duplicate_logs())

    def test_reject_policy_also_fails_on_reversed_duplicate(self) -> None:
        logs = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1"],
                "item_id": ["item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a"],
                "prompt_order": ["reversed", "reversed"],
                "repeat_index": [0, 1],
                "correct": [True, False],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate judgment: item_key=gpt:item-1 judge_id=judge-a prompt_order=reversed count=2",
        ):
            resolve_original_judgments(logs)

    def test_first_policy_keeps_first_duplicate_in_log_order(self) -> None:
        resolved = resolve_original_judgments(self.duplicate_logs(), repeat_policy="first")

        self.assertEqual(resolved.sort(["item_key", "judge_id"])["correct_int"].to_list(), [1])

    def test_latest_policy_keeps_last_duplicate_in_log_order(self) -> None:
        resolved = resolve_original_judgments(self.duplicate_logs(), repeat_policy="latest")

        self.assertEqual(resolved.sort(["item_key", "judge_id"])["correct_int"].to_list(), [0])

    def test_rejects_unsupported_repeat_policy_at_runtime(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported repeat policy: newest"):
            resolve_original_judgments(self.duplicate_logs(), repeat_policy="newest")  # type: ignore[arg-type]

    def test_policy_resolves_repeat_before_excluding_invalid_verdict(self) -> None:
        logs = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1"],
                "item_id": ["item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a"],
                "prompt_order": ["original", "original"],
                "repeat_index": [0, 1],
                "correct": [None, True],
            }
        )

        first = resolve_original_judgments(logs, repeat_policy="first")
        latest = resolve_original_judgments(logs, repeat_policy="latest")

        self.assertEqual(first.height, 0)
        self.assertEqual(latest["correct_int"].to_list(), [1])

    def test_pivot_original_judgments_returns_wide_matrix(self) -> None:
        first = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1", "claude:item-2"],
                "judge_id": ["judge-a", "judge-b", "judge-a"],
                "correct_int": [1, 0, 1],
            }
        )

        pivoted = pivot_original_judgments(first).sort("item_key")

        self.assertEqual(pivoted.columns, ["item_key", "judge-a", "judge-b"])
        self.assertEqual(pivoted["judge-a"].to_list(), [1, 1])
        self.assertEqual(pivoted["judge-b"].to_list(), [None, 0])

    def test_judge_columns_and_summary_skip_metadata(self) -> None:
        matrix = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "claude:item-2"],
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
                "item_key": ["gpt:item-1", "claude:item-2"],
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
