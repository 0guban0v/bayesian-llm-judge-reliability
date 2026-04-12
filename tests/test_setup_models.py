"""Regression tests for model setup helpers."""

from __future__ import annotations

import unittest

from scripts.setup_models import unique_judge_models
from src.schemas import JudgeConfig


class UniqueJudgeModelsTests(unittest.TestCase):
    """Verify setup deduplicates model loads while preserving order."""

    def test_unique_judge_models_deduplicates_model_and_trust_flag_pairs(self) -> None:
        judges = [
            JudgeConfig(
                id="judge-a",
                model="model-1",
                prompt_template="pointwise",
                trust_remote_code=False,
            ),
            JudgeConfig(
                id="judge-b",
                model="model-1",
                prompt_template="pointwise",
                trust_remote_code=False,
            ),
            JudgeConfig(
                id="judge-c",
                model="model-1",
                prompt_template="pointwise",
                trust_remote_code=True,
            ),
            JudgeConfig(
                id="judge-d",
                model="model-2",
                prompt_template="pointwise",
                trust_remote_code=False,
            ),
        ]

        self.assertEqual(
            unique_judge_models(judges),
            [("model-1", False), ("model-1", True), ("model-2", False)],
        )


if __name__ == "__main__":
    unittest.main()
