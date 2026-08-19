"""Regression tests for judge runner orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl
from src.data.item_identity import item_content_hash
from src.judges.prompts import FIXED_PROMPT_VARIANT, PROMPT_PROTOCOL_VERSION
from src.judges.runner import (
    judge_item,
    judge_metadata_fields,
    load_processed_keys,
    run_all,
    run_judge,
    validate_log_metadata,
)
from src.schemas import ExperimentConfig


def build_item(
    *,
    item_key: str = "gpt:item-1",
    item_id: str = "item-1",
    split: str = "gpt",
    source: str = "s1",
    question: str = "q1",
    response_a: str = "a1",
    response_b: str = "b1",
    label: str = "A>B",
    original_id: int = 1,
    response_model: str = "m1",
) -> dict[str, object]:
    """Return one item carrying its canonical content hash."""

    item: dict[str, object] = {
        "item_key": item_key,
        "item_id": item_id,
        "original_id": original_id,
        "split": split,
        "source": source,
        "question": question,
        "response_model": response_model,
        "response_a": response_a,
        "response_b": response_b,
        "label": label,
    }
    item["item_content_hash"] = item_content_hash(item)
    return item


class RunAllTests(unittest.TestCase):
    """Verify runner orchestration keeps one shared item materialization."""

    def test_run_all_materializes_items_once_before_iterating_judges(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        first_item = build_item()
        second_item = build_item(
            item_key="claude:item-2",
            item_id="item-2",
            original_id=2,
            split="claude",
            source="s2",
            question="q2",
            response_model="m2",
            response_a="a2",
            response_b="b2",
            label="B>A",
        )
        items = pl.DataFrame([first_item, second_item])
        expected_items = [first_item]
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
        item = build_item()

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
        self.assertEqual(result.item_content_hash, item["item_content_hash"])
        self.assertEqual(result.repeat_index, 0)
        self.assertTrue(result.correct)

    def test_judge_result_rejects_unknown_jsonl_fields(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        item = build_item()

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

    def test_judge_item_preserves_requested_repeat_index(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        item = build_item()

        with (
            patch("src.judges.runner.generate_text", return_value="FINAL VERDICT: A"),
            patch("src.judges.runner.time.perf_counter", side_effect=[0.0, 0.01]),
        ):
            result = judge_item(judge, item, "original", repeat_index=3)

        self.assertEqual(result.repeat_index, 3)


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
            "item_content_hash": "0" * 64,
            "item_id": "item-1",
            "judge_id": judge_id,
            "timestamp": "2026-04-16T00:00:00+00:00",
            "source": "s1",
            "question": "q1",
            "ground_truth_label": "A>B",
            "prompt_variant": FIXED_PROMPT_VARIANT,
            "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
            "prompt_order": prompt_order,
            "repeat_index": 0,
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "max_tokens": 8,
            "trust_remote_code": False,
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
        self.assertNotIn("num_repeats", metadata_fields)
        self.assertNotIn("prompt_orders", metadata_fields)

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

    def test_validate_log_metadata_rejects_legacy_log_without_repeat_index(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        record = self.build_log_record(judge.id, **judge_metadata_fields(judge))
        del record["repeat_index"]

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "predates explicit repeat indices"):
                validate_log_metadata(log_path, judge)

    def test_validate_log_metadata_accepts_compatible_legacy_reverse_order_field(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        record = self.build_log_record(judge.id, reverse_order=False, **judge_metadata_fields(judge))

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

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

    def test_load_processed_keys_qualifies_item_by_content_hash(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        record = self.build_log_record(judge.id, repeat_index=3, **judge_metadata_fields(judge))

        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            self.assertEqual(
                load_processed_keys(log_path),
                {("gpt:item-1", "0" * 64, "original", 3)},
            )

    def test_run_judge_reprocesses_item_when_content_hash_changes(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0].model_copy(update={"prompt_orders": ["original"]})
        old_item = build_item()
        changed_item = build_item(question="changed question")
        old_record = self.build_log_record(
            judge.id,
            item_content_hash=old_item["item_content_hash"],
            **judge_metadata_fields(judge),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            config = config.model_copy(update={"data": config.data.model_copy(update={"logs_dir": logs_dir})})
            log_path = logs_dir / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(old_record) + "\n", encoding="utf-8")

            with (
                patch("src.judges.runner.generate_text", return_value="FINAL VERDICT: A"),
                patch("src.judges.runner.time.perf_counter", side_effect=[0.0, 0.01]),
            ):
                completed = run_judge(config, judge, [changed_item])

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(completed, 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]["item_content_hash"], changed_item["item_content_hash"])
        self.assertEqual(records[-1]["repeat_index"], 0)

    def test_run_judge_schedules_and_resumes_each_repeat_index(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0].model_copy(update={"num_repeats": 3, "prompt_orders": ["original"]})
        item = build_item()
        existing_record = self.build_log_record(
            judge.id,
            item_content_hash=item["item_content_hash"],
            repeat_index=0,
            **judge_metadata_fields(judge),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            config = config.model_copy(update={"data": config.data.model_copy(update={"logs_dir": logs_dir})})
            log_path = logs_dir / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")

            with (
                patch(
                    "src.judges.runner.generate_text",
                    side_effect=["FINAL VERDICT: A", "FINAL VERDICT: B"],
                ),
                patch(
                    "src.judges.runner.time.perf_counter",
                    side_effect=[0.0, 0.01, 0.02, 0.03],
                ),
            ):
                completed = run_judge(config, judge, [item])

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(completed, 2)
        self.assertEqual([record["repeat_index"] for record in records], [0, 1, 2])

    def test_run_judge_schedules_and_resumes_each_prompt_order(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        judge = config.judges[0]
        item = build_item()
        existing_record = self.build_log_record(
            judge.id,
            item_content_hash=item["item_content_hash"],
            prompt_order="original",
            reverse_order=False,
            **judge_metadata_fields(judge),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = Path(temp_dir)
            config = config.model_copy(update={"data": config.data.model_copy(update={"logs_dir": logs_dir})})
            log_path = logs_dir / f"{judge.id}.jsonl"
            log_path.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")

            with (
                patch("src.judges.runner.generate_text", return_value="FINAL VERDICT: A"),
                patch("src.judges.runner.time.perf_counter", side_effect=[0.0, 0.01]),
            ):
                completed = run_judge(config, judge, [item])

            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(completed, 1)
        self.assertEqual([record["prompt_order"] for record in records], ["original", "reversed"])


if __name__ == "__main__":
    unittest.main()
