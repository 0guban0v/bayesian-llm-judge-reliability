"""Regression tests for configuration schema helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.schemas import ExperimentConfig, InferenceConfig


class InferenceConfigTests(unittest.TestCase):
    """Verify backend-specific posterior path resolution."""

    def test_posterior_path_for_blackjax_adds_suffix_once(self) -> None:
        config = InferenceConfig(
            sampler="NUTS",
            num_warmup=10,
            num_samples=10,
            num_chains=2,
            target_accept_prob=0.8,
            output_dir=Path("data/processed/posteriors"),
            file_name="irt_posterior.npz",
        )

        resolved = config.posterior_path_for_backend("blackjax")

        self.assertEqual(
            resolved,
            Path("data/processed/posteriors/irt_posterior_blackjax.npz"),
        )

    def test_posterior_path_for_blackjax_preserves_existing_suffix(self) -> None:
        config = InferenceConfig(
            sampler="NUTS",
            num_warmup=10,
            num_samples=10,
            num_chains=2,
            target_accept_prob=0.8,
            output_dir=Path("data/processed/posteriors"),
            file_name="irt_2pl_blackjax.npz",
        )

        resolved = config.posterior_path_for_backend("blackjax")

        self.assertEqual(
            resolved,
            Path("data/processed/posteriors/irt_2pl_blackjax.npz"),
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
