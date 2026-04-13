"""Regression tests for JudgeBench loading and matrix construction."""

from __future__ import annotations

import unittest

import polars as pl
from src.data.loader import _matches_categories, build_binary_matrix
from src.data.validate import assert_complete_judge_coverage, validate_items


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


class CategoryMatcherTests(unittest.TestCase):
    """Verify category matching uses token sequences rather than substrings."""

    def test_matches_single_token_category(self) -> None:
        self.assertTrue(_matches_categories("livebench-math", ["math"]))

    def test_matches_multiword_category(self) -> None:
        self.assertTrue(_matches_categories("mmlu-pro-computer science", ["computer science"]))

    def test_does_not_match_substring_inside_larger_word(self) -> None:
        self.assertFalse(_matches_categories("bioinformatics", ["math"]))

    def test_does_not_match_related_but_different_phrase(self) -> None:
        self.assertFalse(_matches_categories("computational mathematics", ["math"]))

    def test_empty_category_list_matches_all_sources(self) -> None:
        self.assertTrue(_matches_categories("any-source", []))

    def test_blank_category_does_not_become_match_all(self) -> None:
        self.assertFalse(_matches_categories("livebench-math", ["   "]))


class ValidateItemsTests(unittest.TestCase):
    """Verify JudgeBench item validation catches malformed subsets."""

    def test_accepts_valid_items(self) -> None:
        items = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "original_id": [1, 2],
                "split": ["gpt", "claude"],
                "source": ["source-a", "source-b"],
                "question": ["question-a", "question-b"],
                "label": ["A>B", "B>A"],
            }
        )

        validate_items(items)

    def test_rejects_duplicate_item_ids(self) -> None:
        items = pl.DataFrame(
            {
                "item_id": ["item-1", "item-1"],
                "original_id": [1, 2],
                "split": ["gpt", "claude"],
                "source": ["source-a", "source-b"],
                "question": ["question-a", "question-b"],
                "label": ["A>B", "B>A"],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Sampled JudgeBench items contain duplicate item IDs.",
        ):
            validate_items(items)

    def test_rejects_invalid_labels(self) -> None:
        items = pl.DataFrame(
            {
                "item_id": ["item-1", "item-2"],
                "original_id": [1, 2],
                "split": ["gpt", "claude"],
                "source": ["source-a", "source-b"],
                "question": ["question-a", "question-b"],
                "label": ["A>B", "TIE"],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Sampled JudgeBench items contain unsupported labels.",
        ):
            validate_items(items)


class ValidateCoverageTests(unittest.TestCase):
    """Verify inference coverage guard rejects partial judge matrices."""

    def test_accepts_complete_judge_coverage(self) -> None:
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

        assert_complete_judge_coverage(matrix, ["judge-a", "judge-b"])

    def test_rejects_incomplete_judge_coverage(self) -> None:
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

        with self.assertRaisesRegex(
            ValueError,
            r"Inference requires complete judge coverage.*judge-a \(1/2\)",
        ):
            assert_complete_judge_coverage(matrix, ["judge-a", "judge-b"])


if __name__ == "__main__":
    unittest.main()
