"""Regression tests for model setup helpers."""

from __future__ import annotations

import unittest
from unittest import mock

from scripts.setup_models import ensure_mlx_platform
from src.schemas import JudgeConfig, unique_model_requests


class UniqueModelRequestsTests(unittest.TestCase):
    """Verify setup deduplicates model loads while preserving order."""

    def test_unique_judge_models_deduplicates_model_and_trust_flag_pairs(self) -> None:
        judges = [
            JudgeConfig(
                id="judge-a",
                model="model-1",
                trust_remote_code=False,
            ),
            JudgeConfig(
                id="judge-b",
                model="model-1",
                trust_remote_code=False,
            ),
            JudgeConfig(
                id="judge-c",
                model="model-1",
                trust_remote_code=True,
            ),
            JudgeConfig(
                id="judge-d",
                model="model-2",
                trust_remote_code=False,
            ),
        ]

        self.assertEqual(
            unique_model_requests(judges),
            [("model-1", False), ("model-1", True), ("model-2", False)],
        )


class EnsureMlxPlatformTests(unittest.TestCase):
    """Verify platform gating fails early on unsupported machines."""

    @mock.patch("scripts.setup_models.platform.processor", return_value="i386")
    @mock.patch("scripts.setup_models.platform.machine", return_value="x86_64")
    @mock.patch("scripts.setup_models.platform.system", return_value="Darwin")
    def test_rejects_intel_macos(
        self,
        _system: mock.Mock,
        _machine: mock.Mock,
        _processor: mock.Mock,
    ) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "mlx setup requires Apple Silicon \\(arm64\\); Intel macOS is not supported.",
        ):
            ensure_mlx_platform()

    @mock.patch("scripts.setup_models.subprocess.run")
    @mock.patch("scripts.setup_models.platform.processor", return_value="arm")
    @mock.patch("scripts.setup_models.platform.machine", return_value="arm64")
    @mock.patch("scripts.setup_models.platform.system", return_value="Darwin")
    def test_allows_apple_silicon_to_reach_metal_check(
        self,
        _system: mock.Mock,
        _machine: mock.Mock,
        _processor: mock.Mock,
        run_mock: mock.Mock,
    ) -> None:
        run_mock.return_value = mock.Mock(returncode=0, stdout="Metal: Supported")

        ensure_mlx_platform()

        run_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
