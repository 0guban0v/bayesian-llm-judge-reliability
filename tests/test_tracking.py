"""Regression tests for pure MLflow tracking helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from src.schemas import ExperimentConfig
from src.tracking import (
    cpu_name,
    inferred_accelerator,
    normalize_process_ru_maxrss_bytes,
    rank_order_string,
    resolved_pairwise_count,
    run_name,
    system_telemetry_params,
    total_ram_bytes,
)


class TrackingHelperTests(unittest.TestCase):
    """Verify pure tracking helper behavior without importing MLflow."""

    def test_run_name_uses_experiment_split_type_and_variant(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment_gpt_source_hier.yaml")

        self.assertEqual(
            run_name(config),
            "bayesian-llm-judge-reliability-gpt-source-hier-gpt-2PL-source_hier",
        )

    def test_rank_order_string_uses_posterior_mean_order(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b", "judge-c"]),
            "theta": np.asarray(
                [
                    [
                        [2.0, 1.0, -1.0],
                        [0.0, 1.0, -2.0],
                        [3.0, 1.0, -3.0],
                        [1.0, 1.0, -4.0],
                    ]
                ]
            ),
        }

        self.assertEqual(rank_order_string(posterior), "judge-a>judge-b>judge-c")

    def test_resolved_pairwise_count_uses_support_threshold_in_either_direction(self) -> None:
        posterior = {
            "judge_ids": np.asarray(["judge-a", "judge-b", "judge-c"]),
            "theta": np.asarray(
                [
                    [
                        [4.0, 1.0, -4.0],
                        [4.0, 1.0, -4.0],
                        [4.0, 1.0, -4.0],
                        [4.0, 1.0, -4.0],
                    ]
                ]
            ),
        }

        self.assertEqual(resolved_pairwise_count(posterior), 3)

    def test_normalize_process_ru_maxrss_bytes_interprets_darwin_as_bytes(self) -> None:
        self.assertEqual(normalize_process_ru_maxrss_bytes(512, "darwin"), 512)

    def test_normalize_process_ru_maxrss_bytes_interprets_linux_as_kibibytes(self) -> None:
        self.assertEqual(normalize_process_ru_maxrss_bytes(512, "linux"), 512 * 1024)

    def test_total_ram_bytes_returns_none_when_sysconf_fails(self) -> None:
        with patch("src.tracking.os.sysconf", side_effect=ValueError):
            self.assertIsNone(total_ram_bytes())

    def test_inferred_accelerator_reports_apple_metal_on_arm64_darwin(self) -> None:
        with patch("src.tracking.sys.platform", "darwin"), patch("src.tracking.platform.machine", return_value="arm64"):
            self.assertEqual(inferred_accelerator(), "apple_metal")

    def test_cpu_name_uses_environment_fallback_when_platform_reports_nothing(self) -> None:
        with (
            patch("src.tracking.platform.processor", return_value=""),
            patch("src.tracking.platform.uname") as uname_mock,
            patch.dict("src.tracking.os.environ", {"PROCESSOR_IDENTIFIER": "Intel64 Family 6 Model 191"}),
        ):
            uname_mock.return_value.processor = ""
            self.assertEqual(cpu_name(), "Intel64 Family 6 Model 191")

    def test_system_telemetry_params_adds_apple_soc_when_available(self) -> None:
        with (
            patch("src.tracking.sys.platform", "darwin"),
            patch("src.tracking.platform.machine", return_value="arm64"),
            patch("src.tracking.platform.python_version", return_value="3.12.0"),
            patch("src.tracking.os.cpu_count", return_value=16),
            patch("src.tracking.total_ram_bytes", return_value=137438953472),
            patch("src.tracking.cpu_name", return_value="Apple M4 Max"),
        ):
            params = system_telemetry_params()

        self.assertEqual(params["apple_soc"], "Apple M4 Max")


if __name__ == "__main__":
    unittest.main()
