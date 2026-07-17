"""Regression tests for JudgeBench loading and matrix construction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
from src.data.item_identity import ITEM_CONTENT_FIELDS, item_content_hash
from src.data.loader import _dataset_to_frame, _matches_categories, build_binary_matrix, load_or_prepare_items
from src.data.validate import assert_complete_judge_coverage, validate_items
from src.schemas import ExperimentConfig


def build_item(
    *,
    item_key: str = "gpt:item-1",
    item_id: str = "item-1",
    original_id: int = 1,
    split: str = "gpt",
    source: str = "source-a",
    question: str = "question-a",
    response_a: str = "answer-a",
    response_b: str = "answer-b",
    label: str = "A>B",
) -> dict[str, object]:
    """Return one valid item carrying its canonical content hash."""

    item: dict[str, object] = {
        "item_key": item_key,
        "item_id": item_id,
        "original_id": original_id,
        "split": split,
        "source": source,
        "question": question,
        "response_model": "model-a",
        "response_a": response_a,
        "response_b": response_b,
        "label": label,
    }
    item["item_content_hash"] = item_content_hash(item)
    return item


class ItemContentHashTests(unittest.TestCase):
    """Verify stable hashing over judge-relevant item content."""

    def test_normalized_equivalent_content_has_same_hash(self) -> None:
        composed = build_item(question="  café\r\nline  ")
        decomposed = build_item(question="cafe\u0301\nline")

        self.assertEqual(item_content_hash(composed), item_content_hash(decomposed))

    def test_each_content_field_changes_hash(self) -> None:
        baseline = build_item()
        baseline_hash = item_content_hash(baseline)

        for field in ITEM_CONTENT_FIELDS:
            with self.subTest(field=field):
                mutated = dict(baseline)
                mutated[field] = f"{mutated[field]}-changed"
                self.assertNotEqual(item_content_hash(mutated), baseline_hash)

    def test_grouping_field_whitespace_changes_hash(self) -> None:
        baseline = build_item()

        for field in ("source", "split"):
            with self.subTest(field=field):
                mutated = dict(baseline)
                mutated[field] = f" {mutated[field]}"
                self.assertNotEqual(item_content_hash(mutated), item_content_hash(baseline))

    def test_missing_content_field_is_rejected(self) -> None:
        item = build_item()
        del item["response_b"]

        with self.assertRaisesRegex(ValueError, "requires fields: response_b"):
            item_content_hash(item)


class DatasetToFrameTests(unittest.TestCase):
    """Verify normalized JudgeBench rows receive content identities."""

    def test_adds_item_content_hash(self) -> None:
        table = pl.DataFrame(
            {
                "pair_id": ["item-1"],
                "original_id": [1],
                "source": ["source-a"],
                "question": ["question-a"],
                "response_model": ["model-a"],
                "response_A": ["answer-a"],
                "response_B": ["answer-b"],
                "label": ["A>B"],
            }
        ).to_arrow()
        dataset = SimpleNamespace(data=SimpleNamespace(table=table))

        with patch("src.data.loader.load_dataset", return_value=dataset):
            items = _dataset_to_frame("dataset", "gpt")

        item = items.row(0, named=True)
        self.assertEqual(item["item_content_hash"], item_content_hash(item))


class BuildBinaryMatrixTests(unittest.TestCase):
    """Verify matrix construction edge cases."""

    def test_warns_when_duplicate_judgments_exist(self) -> None:
        item = build_item(source="source", question="question")
        items = pl.DataFrame([item])
        logs = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1"],
                "item_content_hash": [item["item_content_hash"], item["item_content_hash"]],
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
        gpt_item = build_item()
        claude_item = build_item(
            item_key="claude:item-1",
            split="claude",
            source="source-b",
            question="question-b",
            label="B>A",
        )
        items = pl.DataFrame([gpt_item, claude_item])
        logs = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "claude:item-1"],
                "item_content_hash": [
                    gpt_item["item_content_hash"],
                    claude_item["item_content_hash"],
                ],
                "item_id": ["item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a"],
                "prompt_order": ["original", "original"],
                "correct": [True, False],
            }
        )

        matrix = build_binary_matrix(items, logs, ["judge-a"]).sort("item_key")

        self.assertEqual(matrix.get_column("item_key").to_list(), ["claude:item-1", "gpt:item-1"])
        self.assertEqual(matrix.get_column("judge-a").to_list(), [0, 1])

    def test_uses_current_content_result_when_stale_result_appears_first(self) -> None:
        old_item = build_item()
        current_item = build_item(question="changed question")
        items = pl.DataFrame([current_item])
        logs = pl.DataFrame(
            {
                "item_key": [old_item["item_key"], current_item["item_key"]],
                "item_content_hash": [
                    old_item["item_content_hash"],
                    current_item["item_content_hash"],
                ],
                "item_id": ["item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a"],
                "prompt_order": ["original", "original"],
                "correct": [True, False],
            }
        )

        with self.assertLogs("src.data.loader", level="WARNING") as captured:
            matrix = build_binary_matrix(items, logs, ["judge-a"])

        self.assertIn("stale judgments excluded because item content changed", captured.output[0])
        self.assertEqual(matrix["judge-a"].to_list(), [0])

    def test_excludes_stale_result_when_current_content_has_not_been_judged(self) -> None:
        old_item = build_item()
        current_item = build_item(label="B>A")
        logs = pl.DataFrame(
            {
                "item_key": [old_item["item_key"]],
                "item_content_hash": [old_item["item_content_hash"]],
                "item_id": ["item-1"],
                "judge_id": ["judge-a"],
                "prompt_order": ["original"],
                "correct": [True],
            }
        )

        with self.assertLogs("src.data.loader", level="WARNING"):
            matrix = build_binary_matrix(pl.DataFrame([current_item]), logs, ["judge-a"])

        self.assertEqual(matrix["judge-a"].to_list(), [None])


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
            [
                build_item(),
                build_item(
                    item_key="claude:item-1",
                    original_id=2,
                    split="claude",
                    source="source-b",
                    question="question-b",
                    label="B>A",
                ),
            ]
        )

        validate_items(items)

    def test_rejects_duplicate_split_qualified_item_keys(self) -> None:
        items = pl.DataFrame([build_item(), build_item(original_id=2)])

        with self.assertRaisesRegex(
            ValueError,
            "Sampled JudgeBench items contain duplicate split-qualified item keys.",
        ):
            validate_items(items)

    def test_rejects_invalid_labels(self) -> None:
        items = pl.DataFrame([build_item(label="TIE")])

        with self.assertRaisesRegex(
            ValueError,
            "Sampled JudgeBench items contain unsupported labels.",
        ):
            validate_items(items)

    def test_rejects_missing_item_content_hash(self) -> None:
        item = build_item()
        del item["item_content_hash"]

        with self.assertRaisesRegex(ValueError, "missing content-hash columns: item_content_hash"):
            validate_items(pl.DataFrame([item]))

    def test_rejects_mismatched_item_content_hash(self) -> None:
        item = build_item()
        item["question"] = "changed after hashing"

        with self.assertRaisesRegex(ValueError, "Item content hash mismatch for item_key=gpt:item-1"):
            validate_items(pl.DataFrame([item]))


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

    def test_rejects_cached_items_without_item_content_hash(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml")).model_copy(deep=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config.data.output_dir = root / "processed"
            config.data.raw_dir = root / "raw"
            config.data.logs_dir = root / "logs"
            config.inference.output_dir = root / "posteriors"
            legacy_item = build_item()
            del legacy_item["item_content_hash"]
            config.ensure_directories()
            pl.DataFrame([legacy_item]).write_parquet(config.data.item_path)

            with self.assertRaisesRegex(ValueError, "predate item content hashes.*--refresh-items"):
                load_or_prepare_items(config)


if __name__ == "__main__":
    unittest.main()
