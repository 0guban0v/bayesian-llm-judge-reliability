"""Regression tests for judge runner orchestration."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl
from src.judges.runner import run_all
from src.schemas import ExperimentConfig


class RunAllTests(unittest.TestCase):
    """Verify runner orchestration keeps one shared item materialization."""

    def test_run_all_materializes_items_once_before_iterating_judges(self) -> None:
        config = ExperimentConfig.from_yaml(Path("configs/experiment.yaml"))
        items = pl.DataFrame(
            {
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
        materialized_items = items.to_dicts()
        captured_item_lists: list[list[dict[str, object]]] = []

        def fake_run_judge(
            _config: ExperimentConfig,
            _judge: object,
            runner_items: list[dict[str, object]],
        ) -> int:
            captured_item_lists.append(runner_items)
            return 0

        with (
            patch("src.judges.runner.load_or_prepare_items", return_value=items),
            patch("src.judges.runner.run_judge", side_effect=fake_run_judge),
            patch("src.judges.runner.clear_model_cache"),
        ):
            run_all(config, judge_id=None, limit=1)

        self.assertEqual(len(captured_item_lists), len(config.judges))
        self.assertTrue(
            all(runner_items == materialized_items[:1] for runner_items in captured_item_lists)
        )
        first_id = id(captured_item_lists[0])
        self.assertTrue(all(id(runner_items) == first_id for runner_items in captured_item_lists))


if __name__ == "__main__":
    unittest.main()
