"""Regression tests for log formatting helpers."""

from __future__ import annotations

import unittest

import polars as pl
from src.logging_utils import format_table_for_log


class FormatTableForLogTests(unittest.TestCase):
    """Verify compact ASCII rendering for log tables."""

    def test_returns_empty_marker_for_empty_frame(self) -> None:
        frame = pl.DataFrame(schema={"judge_id": pl.String, "accuracy": pl.Float64})

        rendered = format_table_for_log(frame)

        self.assertEqual(rendered, "<empty>")

    def test_renders_ascii_table_without_shape_or_dtype_metadata(self) -> None:
        frame = pl.DataFrame(
            {
                "judge_id": ["judge-a", "judge-b"],
                "accuracy": [0.8, 0.6],
            }
        )

        rendered = format_table_for_log(frame)

        self.assertIn("judge_id", rendered)
        self.assertIn("accuracy", rendered)
        self.assertIn("|", rendered)
        self.assertIn("+", rendered)
        self.assertNotIn("shape:", rendered)
        self.assertNotIn("f64", rendered)
        self.assertNotIn("str", rendered)
        self.assertNotIn("┌", rendered)


if __name__ == "__main__":
    unittest.main()
