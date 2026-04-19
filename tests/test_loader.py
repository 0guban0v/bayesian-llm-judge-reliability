"""Regression tests for JudgeBench loading and matrix construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl
from src.data.loader import _matches_categories, build_binary_matrix, load_or_prepare_items
from src.data.validate import assert_complete_judge_coverage, validate_items
from src.schemas import ExperimentConfig


class BuildBinaryMatrixTests(unittest.TestCase):
    """Verify matrix construction edge cases."""

    def test_warns_when_duplicate_judgments_exist(self) -> None:
        items = pl.DataFrame(
            {
                "item_key": ["gpt:item-1"],
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
                "item_key": ["gpt:item-1", "gpt:item-1"],
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

    def test_distinguishes_same_item_id_across_splits_via_item_key(self) -> None:
        items = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "claude:item-1"],
                "item_id": ["item-1", "item-1"],
                "original_id": [1, 1],
                "split": ["gpt", "claude"],
                "source": ["source-a", "source-b"],
                "question": ["question-a", "question-b"],
                "label": ["A>B", "B>A"],
            }
        )
        logs = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "claude:item-1"],
                "item_id": ["item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a"],
                "prompt_order": ["original", "original"],
                "correct": [True, False],
            }
        )

        matrix = build_binary_matrix(items, logs, ["judge-a"]).sort("item_key")

        self.assertEqual(matrix.get_column("item_key").to_list(), ["claude:item-1", "gpt:item-1"])
        self.assertEqual(matrix.get_column("judge-a").to_list(), [0, 1])


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
                "item_key": ["gpt:item-1", "claude:item-1"],
                "item_id": ["item-1", "item-1"],
                "original_id": [1, 2],
                "split": ["gpt", "claude"],
                "source": ["source-a", "source-b"],
                "question": ["question-a", "question-b"],
                "label": ["A>B", "B>A"],
            }
        )

        validate_items(items)

    def test_rejects_duplicate_split_qualified_item_keys(self) -> None:
        items = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1"],
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
            "Sampled JudgeBench items contain duplicate split-qualified item keys.",
        ):
            validate_items(items)

    def test_rejects_invalid_labels(self) -> None:
        items = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "claude:item-2"],
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

        assert_complete_judge_coverage(matrix, ["judge-a", "judge-b"])

    def test_rejects_incomplete_judge_coverage(self) -> None:
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

        with self.assertRaisesRegex(
            ValueError,
            r"Inference requires complete judge coverage.*judge-a \(1/2\)",
        ):
            assert_complete_judge_coverage(matrix, ["judge-a", "judge-b"])


class LoadOrPrepareItemsTests(unittest.TestCase):
    """Verify cached item reuse enforces current schema requirements."""

    def test_rejects_legacy_cached_items_without_item_key(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml")).model_copy(deep=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config.data.output_dir = root / "processed"
            config.data.raw_dir = root / "raw"
            config.data.logs_dir = root / "logs"
            config.inference.output_dir = root / "posteriors"
            legacy_items = pl.DataFrame(
                {
                    "item_id": ["item-1"],
                    "original_id": [1],
                    "split": ["gpt"],
                    "source": ["source-a"],
                    "question": ["question-a"],
                    "response_model": ["model-a"],
                    "response_a": ["answer-a"],
                    "response_b": ["answer-b"],
                    "label": ["A>B"],
                }
            )
            config.ensure_directories()
            legacy_items.write_parquet(config.data.item_path)

            with self.assertRaisesRegex(
                ValueError,
                "predate split-qualified item keys.*--refresh-items",
            ):
                load_or_prepare_items(config)


if __name__ == "__main__":
    unittest.main()
