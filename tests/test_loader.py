"""Regression tests for JudgeBench loading and matrix construction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
from src.data.item_identity import ITEM_CONTENT_FIELDS, item_content_hash, validate_item_content_hash
from src.data.loader import (
    _dataset_to_frame,
    _matches_categories,
    build_analysis_table,
    build_and_write_analysis_artifacts,
    build_and_write_matrix,
    build_binary_matrix,
    load_judge_logs,
    load_or_prepare_items,
)
from src.data.validate import (
    assert_complete_judge_coverage,
    assert_complete_original_choice_coverage,
    assert_complete_prompt_order_coverage,
    validate_items,
)
from src.judges.prompts import FIXED_PROMPT_VARIANT, PROMPT_PROTOCOL_VERSION
from src.schemas import ExperimentConfig, JudgeConfig


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


def build_log_record(item: dict[str, object], **overrides: object) -> dict[str, object]:
    """Return one valid persisted judge result."""

    record: dict[str, object] = {
        "item_id": item["item_id"],
        "item_key": item["item_key"],
        "item_content_hash": item["item_content_hash"],
        "judge_id": "judge-a",
        "timestamp": "2026-04-16T00:00:00+00:00",
        "source": item["source"],
        "question": item["question"],
        "ground_truth_label": item["label"],
        "prompt_variant": FIXED_PROMPT_VARIANT,
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "prompt_order": "original",
        "repeat_index": 0,
        "model": "model-a",
        "max_tokens": 8,
        "trust_remote_code": False,
        "raw_response": "FINAL VERDICT: A",
        "parsed_verdict": "A",
        "correct": True,
        "latency_ms": 10,
    }
    record.update(overrides)
    return record


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

    def test_missing_stored_hash_is_rejected_as_required(self) -> None:
        item = build_item()
        del item["item_content_hash"]

        with self.assertRaisesRegex(ValueError, "item_content_hash.*required"):
            validate_item_content_hash(item)

    def test_non_string_stored_hash_reports_type(self) -> None:
        item = build_item()
        item["item_content_hash"] = None

        with self.assertRaisesRegex(ValueError, "item_content_hash.*must be a string, found NoneType"):
            validate_item_content_hash(item)


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

    def test_rejects_duplicate_judgments_by_default(self) -> None:
        item = build_item(source="source", question="question")
        items = pl.DataFrame([item])
        logs = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1"],
                "item_content_hash": [item["item_content_hash"], item["item_content_hash"]],
                "item_id": ["item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a"],
                "prompt_order": ["original", "original"],
                "repeat_index": [0, 1],
                "parsed_verdict": ["A", "B"],
                "correct": [True, False],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate judgment: item_key=gpt:item-1 judge_id=judge-a prompt_order=original count=2",
        ):
            build_binary_matrix(items, logs, ["judge-a"])

    def test_first_and_latest_policies_select_by_log_order(self) -> None:
        item = build_item(source="source", question="question")
        items = pl.DataFrame([item])
        logs = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1"],
                "item_content_hash": [item["item_content_hash"], item["item_content_hash"]],
                "item_id": ["item-1", "item-1"],
                "judge_id": ["judge-a", "judge-a"],
                "prompt_order": ["original", "original"],
                "repeat_index": [0, 1],
                "correct": [True, False],
            }
        )

        with self.assertLogs("src.data.loader", level="WARNING"):
            first_matrix = build_binary_matrix(items, logs, ["judge-a"], repeat_policy="first")
        with self.assertLogs("src.data.loader", level="WARNING"):
            latest_matrix = build_binary_matrix(items, logs, ["judge-a"], repeat_policy="latest")

        self.assertEqual(first_matrix["judge-a"].to_list(), [1])
        self.assertEqual(latest_matrix["judge-a"].to_list(), [0])

    def test_configured_repeat_policy_reaches_matrix_build(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml")).model_copy(deep=True)
        config.data.repeat_policy = "latest"
        item = build_item(source="source", question="question")
        items = pl.DataFrame([item])
        logs = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "gpt:item-1"],
                "item_content_hash": [item["item_content_hash"], item["item_content_hash"]],
                "item_id": ["item-1", "item-1"],
                "judge_id": [config.judges[0].id, config.judges[0].id],
                "prompt_order": ["original", "original"],
                "repeat_index": [0, 1],
                "parsed_verdict": ["A", "B"],
                "correct": [True, False],
            }
        )

        with (
            patch("src.data.loader.load_judge_logs", return_value=logs),
            patch("src.data.loader.write_frame"),
            self.assertLogs("src.data.loader", level="WARNING"),
        ):
            matrix = build_and_write_matrix(config, items)

        self.assertEqual(matrix[config.judges[0].id].to_list(), [0])

    def test_analysis_table_preserves_both_orders_and_choice_spaces(self) -> None:
        item = build_item()
        logs = pl.DataFrame(
            [
                build_log_record(item, prompt_order="original", parsed_verdict="A", correct=True),
                build_log_record(
                    item,
                    prompt_order="reversed",
                    raw_response="FINAL VERDICT: A",
                    parsed_verdict="B",
                    correct=False,
                ),
            ]
        )

        analysis = build_analysis_table(pl.DataFrame([item]), logs).sort("prompt_order")

        self.assertEqual(analysis["prompt_order"].to_list(), ["original", "reversed"])
        self.assertEqual(analysis["displayed_choice"].to_list(), ["A", "A"])
        self.assertEqual(analysis["normalized_choice"].to_list(), ["A", "B"])
        self.assertEqual(analysis["gold_choice"].to_list(), ["A", "A"])
        self.assertEqual(analysis["correct"].to_list(), [True, False])
        self.assertEqual(analysis["valid"].to_list(), [True, True])

    def test_analysis_table_keeps_invalid_choice_as_explicit_row(self) -> None:
        item = build_item()
        logs = pl.DataFrame(
            [
                build_log_record(
                    item,
                    prompt_order="reversed",
                    raw_response="unparseable",
                    parsed_verdict=None,
                    correct=None,
                )
            ]
        )

        analysis = build_analysis_table(pl.DataFrame([item]), logs)

        self.assertEqual(analysis.height, 1)
        self.assertIsNone(analysis.item(0, "displayed_choice"))
        self.assertIsNone(analysis.item(0, "normalized_choice"))
        self.assertIsNone(analysis.item(0, "correct"))
        self.assertFalse(analysis.item(0, "valid"))

    def test_analysis_table_rejects_logged_correctness_that_disagrees_with_choice(self) -> None:
        item = build_item()
        logs = pl.DataFrame([build_log_record(item, parsed_verdict="B", correct=True)])

        with self.assertRaisesRegex(
            ValueError,
            "Judge log correctness is inconsistent with its normalized choice.*prompt_order=original",
        ):
            build_analysis_table(pl.DataFrame([item]), logs)

    def test_artifact_rebuild_persists_both_orders_and_original_compatibility_matrix(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml")).model_copy(deep=True)
        config.judges = [config.judges[0].model_copy(update={"id": "judge-a"})]
        item = build_item()
        items = pl.DataFrame([item])
        logs = pl.DataFrame(
            [
                build_log_record(item, prompt_order="original", parsed_verdict="A", correct=True),
                build_log_record(item, prompt_order="reversed", parsed_verdict="B", correct=False),
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config.data.output_dir = Path(temp_dir)
            analysis, matrix = build_and_write_analysis_artifacts(config, items, logs)
            persisted_analysis = pl.read_parquet(config.data.analysis_path)
            persisted_matrix = pl.read_parquet(config.data.matrix_path)

        self.assertEqual(analysis["prompt_order"].to_list(), ["original", "reversed"])
        self.assertEqual(persisted_analysis["prompt_order"].to_list(), ["original", "reversed"])
        self.assertEqual(matrix["judge-a"].to_list(), [1])
        self.assertEqual(persisted_matrix["judge-a"].to_list(), [1])

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

    def test_stale_warning_limits_item_key_sample(self) -> None:
        old_items = [
            build_item(item_key=f"gpt:item-{index}", item_id=f"item-{index}", original_id=index) for index in range(7)
        ]
        current_items = [
            build_item(
                item_key=f"gpt:item-{index}",
                item_id=f"item-{index}",
                original_id=index,
                question="changed question",
            )
            for index in range(7)
        ]
        logs = pl.DataFrame(
            {
                "item_key": [item["item_key"] for item in old_items],
                "item_content_hash": [item["item_content_hash"] for item in old_items],
                "item_id": [item["item_id"] for item in old_items],
                "judge_id": ["judge-a"] * 7,
                "prompt_order": ["original"] * 7,
                "correct": [True] * 7,
            }
        )

        with self.assertLogs("src.data.loader", level="WARNING") as captured:
            build_binary_matrix(pl.DataFrame(current_items), logs, ["judge-a"])

        warning = captured.output[0]
        self.assertIn("rows=7 item_key_count=7", warning)
        self.assertIn("gpt:item-4", warning)
        self.assertNotIn("gpt:item-5", warning)


class LoadJudgeLogsTests(unittest.TestCase):
    """Verify every persisted log row satisfies current result schema."""

    def test_rejects_missing_hash_in_mixed_log(self) -> None:
        item = build_item()
        first_record = build_log_record(item)
        second_record = build_log_record(item, prompt_order="reversed")
        del second_record["item_content_hash"]

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            log_path = logs_dir / "judge-a.jsonl"
            log_path.write_text(
                json.dumps(first_record) + "\n" + json.dumps(second_record) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "line 2.*predates item content hashes"):
                load_judge_logs(logs_dir)

    def test_rejects_malformed_hash(self) -> None:
        item = build_item()
        record = build_log_record(item, item_content_hash="invalid")

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            log_path = logs_dir / "judge-a.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1: item_content_hash: String should match pattern"):
                load_judge_logs(logs_dir)

    def test_rejects_log_without_explicit_repeat_index(self) -> None:
        item = build_item()
        record = build_log_record(item)
        del record["repeat_index"]

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            log_path = logs_dir / "judge-a.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1.*predates explicit repeat indices"):
                load_judge_logs(logs_dir)

    def test_preserves_explicit_repeat_indices(self) -> None:
        item = build_item()
        records = [
            build_log_record(item, repeat_index=0),
            build_log_record(item, repeat_index=2),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            log_path = logs_dir / "judge-a.jsonl"
            log_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            logs = load_judge_logs(logs_dir)

        self.assertEqual(logs["repeat_index"].to_list(), [0, 2])

    def test_migrates_compatible_legacy_reverse_order_field(self) -> None:
        item = build_item()
        record = build_log_record(item, reverse_order=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            log_path = logs_dir / "judge-a.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            logs = load_judge_logs(logs_dir)

        self.assertNotIn("reverse_order", logs.columns)

    def test_rejects_malformed_legacy_reverse_order_field(self) -> None:
        item = build_item()
        record = build_log_record(item, reverse_order="sometimes")

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            log_path = logs_dir / "judge-a.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reverse_order must be a boolean"):
                load_judge_logs(logs_dir)


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
    """Verify inference coverage guards reject partial judge tasks."""

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

    def test_accepts_complete_prompt_order_coverage(self) -> None:
        items = pl.DataFrame(
            [
                build_item(),
                build_item(item_key="claude:item-2", item_id="item-2", split="claude", original_id=2),
            ]
        )
        logs = pl.DataFrame(
            [
                {
                    "item_key": item["item_key"],
                    "item_content_hash": item["item_content_hash"],
                    "judge_id": "judge-a",
                    "prompt_order": prompt_order,
                    "repeat_index": 0,
                }
                for item in items.iter_rows(named=True)
                for prompt_order in ("original", "reversed")
            ]
        )
        judge = JudgeConfig(
            id="judge-a",
            model="model-a",
            prompt_orders=["original", "reversed"],
        )

        assert_complete_prompt_order_coverage(items, logs, [judge])

    def test_rejects_incomplete_prompt_order_coverage(self) -> None:
        first_item = build_item()
        second_item = build_item(item_key="claude:item-2", item_id="item-2", split="claude", original_id=2)
        items = pl.DataFrame([first_item, second_item])
        logs = pl.DataFrame(
            [
                {
                    "item_key": first_item["item_key"],
                    "item_content_hash": first_item["item_content_hash"],
                    "judge_id": "judge-a",
                    "prompt_order": "original",
                    "repeat_index": 0,
                },
                {
                    "item_key": first_item["item_key"],
                    "item_content_hash": first_item["item_content_hash"],
                    "judge_id": "judge-a",
                    "prompt_order": "reversed",
                    "repeat_index": 0,
                },
                {
                    "item_key": second_item["item_key"],
                    "item_content_hash": second_item["item_content_hash"],
                    "judge_id": "judge-a",
                    "prompt_order": "original",
                    "repeat_index": 0,
                },
            ]
        )
        judge = JudgeConfig(
            id="judge-a",
            model="model-a",
            prompt_orders=["original", "reversed"],
        )

        with self.assertRaisesRegex(
            ValueError,
            r"complete prompt-order coverage.*judge-a/reversed \(1/2\).*claude:item-2",
        ):
            assert_complete_prompt_order_coverage(items, logs, [judge])

    def test_prompt_order_coverage_excludes_stale_item_content(self) -> None:
        old_item = build_item()
        current_item = build_item(question="changed question")
        logs = pl.DataFrame(
            {
                "item_key": [old_item["item_key"]],
                "item_content_hash": [old_item["item_content_hash"]],
                "judge_id": ["judge-a"],
                "prompt_order": ["original"],
                "repeat_index": [0],
            }
        )
        judge = JudgeConfig(id="judge-a", model="model-a")

        with (
            self.assertLogs("src.data.loader", level="WARNING"),
            self.assertRaisesRegex(ValueError, r"judge-a/original \(0/1\)"),
        ):
            assert_complete_prompt_order_coverage(pl.DataFrame([current_item]), logs, [judge])

    def test_rejects_invalid_original_choice_coverage_from_long_form_table(self) -> None:
        item = build_item()
        logs = pl.DataFrame([build_log_record(item, parsed_verdict=None, correct=None)])
        analysis = build_analysis_table(pl.DataFrame([item]), logs)
        judge = JudgeConfig(id="judge-a", model="model-a")

        with self.assertRaisesRegex(
            ValueError,
            r"complete valid original-order choice coverage.*judge-a \(0/1\).*gpt:item-1",
        ):
            assert_complete_original_choice_coverage(analysis, [judge])


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
