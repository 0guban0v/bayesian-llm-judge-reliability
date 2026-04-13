"""Regression tests for configuration schema helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.schemas import ExperimentConfig, InferenceConfig


class InferenceConfigTests(unittest.TestCase):
    """Verify inference path resolution."""

    def test_posterior_path_uses_output_dir_and_file_name(self) -> None:
        config = InferenceConfig(
            sampler="NUTS",
            num_warmup=10,
            num_samples=10,
            num_chains=2,
            target_accept_prob=0.8,
            output_dir=Path("data/processed/posteriors"),
            file_name="irt_posterior.npz",
        )

        resolved = config.posterior_path

        self.assertEqual(
            resolved,
            Path("data/processed/posteriors/irt_posterior.npz"),
        )

    def test_from_yaml_resolves_repo_relative_paths(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        self.assertTrue(config.data.output_dir.is_absolute())
        self.assertTrue(config.data.raw_dir.is_absolute())
        self.assertTrue(config.data.logs_dir.is_absolute())
        self.assertTrue(config.inference.output_dir.is_absolute())
        self.assertEqual(config.figures_dir, Path.cwd() / "figures")
        self.assertEqual(config.report_dir, Path.cwd() / "report")


if __name__ == "__main__":
    unittest.main()
