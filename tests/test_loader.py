"""Regression tests for JudgeBench loading and matrix construction."""

from __future__ import annotations

import unittest

import polars as pl
from src.data.loader import build_binary_matrix


class BuildBinaryMatrixTests(unittest.TestCase):
    """Verify matrix construction edge cases."""

    def test_warns_when_duplicate_judgments_exist(self) -> None:
        items = pl.DataFrame(
            {
                "item_id": ["item-1"],
                "original_id": [1],
                "split": ["gpt"],
                "source": ["source"],
                "question": ["question"],
                "label": ["A>B"],
            }
        )
        logs = pl.DataFrame(
            {
                "item_id": ["item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a"],
                "prompt_order": ["original", "original"],
                "correct": [True, False],
            }
        )

        with self.assertLogs("src.data.loader", level="WARNING") as captured:
            matrix = build_binary_matrix(items, logs, ["judge-a"])

        self.assertIn("duplicate original-order judgments detected", captured.output[0])
        self.assertEqual(matrix["judge-a"].to_list(), [1])


if __name__ == "__main__":
    unittest.main()
