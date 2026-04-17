"""Regression tests for configuration schema helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.schemas import AnalysisConfig, ExperimentConfig, InferenceConfig, PriorConfig


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

    def test_from_yaml_loads_analysis_defaults(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        self.assertEqual(config.analysis.plots.max_sources, 8)
        self.assertEqual(config.analysis.report.standout_judge_id, "deepseek-r1-distill-qwen-14b")
        self.assertEqual(config.analysis.report.standout_case_limit, 3)
        self.assertEqual(config.analysis.report.response_synopsis_chars, 120)

    def test_analysis_config_defaults_are_available_without_yaml_block(self) -> None:
        analysis = AnalysisConfig()

        self.assertEqual(analysis.plots.max_sources, 8)
        self.assertEqual(analysis.report.standout_judge_id, "deepseek-r1-distill-qwen-14b")
        self.assertEqual(analysis.report.standout_case_limit, 3)
        self.assertEqual(analysis.report.response_synopsis_chars, 120)

    def test_experiment_config_requires_at_least_one_judge(self) -> None:
        payload = {
            "experiment": {"name": "demo", "seed": 7, "date": "2026-04-07"},
            "data": {
                "source": "judgebench",
                "subset_size": 1,
            },
            "judges": [],
            "inference": {
                "sampler": "NUTS",
                "num_warmup": 10,
                "num_samples": 10,
                "num_chains": 2,
                "target_accept_prob": 0.8,
            },
            "model": {
                "type": "1PL",
                "variant": "global",
                "priors": {
                    "theta": {"dist": "normal", "loc": 0.0, "scale": 1.0},
                    "b": {"dist": "normal", "loc": 0.0, "scale": 1.0},
                    "a": {"dist": "lognormal", "loc": 0.0, "scale": 1.0},
                },
            },
        }

        with self.assertRaisesRegex(ValueError, "At least one judge must be configured"):
            ExperimentConfig.model_validate(payload)

    def test_prior_config_requires_declared_distribution(self) -> None:
        config = PriorConfig(dist="normal", loc=0.0, scale=1.0)

        self.assertEqual(config.dist, "normal")

    def test_prior_config_rejects_unknown_distribution(self) -> None:
        with self.assertRaisesRegex(ValueError, "Input should be 'normal' or 'lognormal'"):
            PriorConfig(dist="gamma", loc=0.0, scale=1.0)


if __name__ == "__main__":
    unittest.main()
