"""Tests for model verification request handling."""

from __future__ import annotations

import unittest

from src.schemas import JudgeConfig, unique_model_requests


class VerifyModelsTests(unittest.TestCase):
    """Verify config-driven verification requests are deduplicated correctly."""

    def test_unique_model_requests_deduplicates_model_and_trust_flag_pairs(self) -> None:
        judges = [
            JudgeConfig(
                id="judge-a",
                model="model-a",
                trust_remote_code=False,
            ),
            JudgeConfig(
                id="judge-b",
                model="model-a",
                trust_remote_code=False,
            ),
            JudgeConfig(
                id="judge-c",
                model="model-a",
                trust_remote_code=True,
            ),
            JudgeConfig(
                id="judge-d",
                model="model-b",
                trust_remote_code=False,
            ),
        ]

        self.assertEqual(
            unique_model_requests(judges),
            [
                ("model-a", False),
                ("model-a", True),
                ("model-b", False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
