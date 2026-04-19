"""Regression tests for judge runner orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
from src.judges.prompts import FIXED_PROMPT_VARIANT, PROMPT_PROTOCOL_VERSION
from src.judges.runner import (
    judge_item,
    judge_metadata_fields,
    load_processed_keys,
    run_all,
    validate_log_metadata,
)
from src.schemas import ExperimentConfig


class RunAllTests(unittest.TestCase):
    """Verify runner orchestration keeps one shared item materialization."""

    def test_run_all_materializes_items_once_before_iterating_judges(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        items = pl.DataFrame(
            {
                "item_key": ["gpt:item-1", "claude:item-2"],
                "item_id": ["item-1", "item-2"],
                "original_id": [1, 2],
                "split": ["gpt", "claude"],
                "source": ["s1", "s2"],
                "question": ["q1", "q2"],
                "response_model": ["m1", "m2"],
                "response_a": ["a1", "a2"],
                "response_b": ["b1", "b2"],
                "label": ["A>B", "B>A"],
            }
        )
        expected_items = [
            {
                "item_key": "gpt:item-1",
                "item_id": "item-1",
                "original_id": 1,
                "split": "gpt",
                "source": "s1",
                "question": "q1",
                "response_model": "m1",
                "response_a": "a1",
                "response_b": "b1",
                "label": "A>B",
            }
        ]
        captured_item_lists: list[list[dict[str, object]]] = []

        def fake_run_judge(
            _config: ExperimentConfig,
            _judge: object,
            runner_items: list[dict[str, object]],
        ) -> int:
            captured_item_lists.append(runner_items)
            return 0

        with (
            patch.object(ExperimentConfig, "ensure_directories"),
            patch("src.judges.runner.load_or_prepare_items", return_value=items),
            patch("src.judges.runner.run_judge", side_effect=fake_run_judge),
            patch("src.judges.runner.clear_model_cache"),
            patch.object(
                pl.DataFrame,
                "to_dicts",
                autospec=True,
                wraps=pl.DataFrame.to_dicts,
            ) as to_dicts_spy,
        ):
            run_all(config, judge_id=None, limit=1)

        to_dicts_spy.assert_called_once()
        self.assertEqual(len(to_dicts_spy.call_args.args), 1)
        self.assertIsInstance(to_dicts_spy.call_args.args[0], pl.DataFrame)
        self.assertEqual(len(captured_item_lists), len(config.judges))
        self.assertTrue(all(runner_items == expected_items for runner_items in captured_item_lists))


class JudgeItemTests(unittest.TestCase):
    """Verify logged judge results reflect the fixed prompt protocol."""

    def test_judge_item_records_fixed_prompt_variant(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        item = {
            "item_key": "gpt:item-1",
            "item_id": "item-1",
            "split": "gpt",
            "source": "s1",
            "question": "q1",
            "response_a": "a1",
            "response_b": "b1",
            "label": "A>B",
        }

        with (
            patch("src.judges.runner.generate_text", return_value="FINAL VERDICT: A"),
            patch("src.judges.runner.time.perf_counter", side_effect=[0.0, 0.01]),
            patch("src.judges.runner.datetime") as datetime_mock,
        ):
            datetime_mock.now.return_value = datetime(2026, 4, 16, tzinfo=UTC)
            result = judge_item(judge, item, "original")

        self.assertEqual(result.prompt_variant, FIXED_PROMPT_VARIANT)
        self.assertEqual(result.prompt_protocol_version, PROMPT_PROTOCOL_VERSION)
        self.assertEqual(result.model, judge.model)
        self.assertTrue(result.correct)

    def test_judge_result_rejects_unknown_jsonl_fields(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        item = {
            "item_key": "gpt:item-1",
            "item_id": "item-1",
            "split": "gpt",
            "source": "s1",
            "question": "q1",
            "response_a": "a1",
            "response_b": "b1",
            "label": "A>B",
        }

        with (
            patch("src.judges.runner.generate_text", return_value="FINAL VERDICT: A"),
            patch("src.judges.runner.time.perf_counter", side_effect=[0.0, 0.01]),
            patch("src.judges.runner.datetime") as datetime_mock,
        ):
            datetime_mock.now.return_value = datetime(2026, 4, 16, tzinfo=UTC)
            result = judge_item(judge, item, "original")

        payload = result.to_json_dict()
        payload["unexpected_field"] = "unexpected"

        with self.assertRaisesRegex(ValueError, "Extra inputs are not permitted|unexpected_field"):
            type(result).model_validate(payload)


class LogMetadataTests(unittest.TestCase):
    """Verify embedded log metadata enforces prompt protocol compatibility."""

    def build_log_record(
        self,
        judge_id: str,
        prompt_order: str = "original",
        **overrides: object,
    ) -> dict[str, object]:
        """Return a valid persisted judge-log row for validation tests."""

        record = {
            "item_key": "gpt:item-1",
            "item_id": "item-1",
            "judge_id": judge_id,
            "timestamp": "2026-04-16T00:00:00+00:00",
            "source": "s1",
            "question": "q1",
            "ground_truth_label": "A>B",
            "prompt_variant": FIXED_PROMPT_VARIANT,
            "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
            "prompt_order": prompt_order,
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "max_tokens": 8,
            "trust_remote_code": False,
            "reverse_order": False,
            "raw_response": "FINAL VERDICT: A",
            "parsed_verdict": "A",
            "correct": True,
            "latency_ms": 10,
        }
        record.update(overrides)
        return record

    def test_judge_metadata_fields_include_prompt_protocol_fields(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]

        metadata_fields = judge_metadata_fields(judge)

        self.assertEqual(metadata_fields["prompt_variant"], FIXED_PROMPT_VARIANT)
        self.assertEqual(metadata_fields["prompt_protocol_version"], PROMPT_PROTOCOL_VERSION)

    def test_validate_log_metadata_rejects_legacy_log_without_embedded_metadata(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "predates split-qualified item keys"):
                validate_log_metadata(log_path, judge)

    def test_validate_log_metadata_accepts_matching_embedded_metadata(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        record = self.build_log_record(judge.id, **judge_metadata_fields(judge))

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            validate_log_metadata(log_path, judge)

    def test_validate_log_metadata_rejects_mixed_metadata_later_in_file(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        first_record = self.build_log_record(judge.id, **judge_metadata_fields(judge))
        second_record = self.build_log_record(
            judge.id,
            item_key="gpt:item-2",
            item_id="item-2",
            **judge_metadata_fields(judge),
        )
        second_record["max_tokens"] = judge.max_tokens + 1

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(
                json.dumps(first_record) + "\n" + json.dumps(second_record) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"changed fields: \['max_tokens'\], line: 2"):
                validate_log_metadata(log_path, judge)

    def test_validate_log_metadata_rejects_missing_required_resume_fields(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        record = self.build_log_record(judge.id, **judge_metadata_fields(judge))
        del record["prompt_order"]

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"log is malformed at line 1: Field required"):
                validate_log_metadata(log_path, judge)

    def test_validate_log_metadata_rejects_invalid_prompt_order_value(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        record = self.build_log_record(judge.id, prompt_order="orig", **judge_metadata_fields(judge))

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"log is malformed at line 1"):
                validate_log_metadata(log_path, judge)

    def test_validate_log_metadata_rejects_unknown_extra_field_on_disk(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        record = self.build_log_record(
            judge.id,
            **judge_metadata_fields(judge),
            unexpected_field="unexpected",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"log is malformed at line 1"):
                validate_log_metadata(log_path, judge)

    def test_load_processed_keys_rejects_malformed_record(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        record = self.build_log_record(judge.id, prompt_order="orig", **judge_metadata_fields(judge))

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"Judge log .* is malformed at line 1"):
                load_processed_keys(log_path)


if __name__ == "__main__":
    unittest.main()
