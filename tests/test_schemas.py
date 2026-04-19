"""Regression tests for configuration schema helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
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
            save_log_likelihood=False,
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
        self.assertFalse(config.inference.save_log_likelihood)

    def test_from_yaml_loads_tracking_defaults(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        self.assertEqual(config.tracking.backend, "mlflow")
        self.assertEqual(config.tracking.experiment_name, "bayesian-llm-judge-reliability")
        self.assertTrue(config.tracking.tracking_db.is_absolute())
        self.assertTrue(config.tracking.artifact_dir.is_absolute())
        self.assertEqual(config.tracking_uri, f"sqlite:///{(Path.cwd() / 'mlflow.db').resolve()}")
        self.assertEqual(config.tracking_artifact_uri, (Path.cwd() / "mlruns").resolve().as_uri())
        self.assertEqual(
            config.tracked_output_dir,
            Path.cwd() / ".tracked_runs" / config.experiment.name,
        )
        self.assertFalse(config.tracked_output_dir.is_relative_to(config.tracking.artifact_dir))

    def test_analysis_config_defaults_are_available_without_yaml_block(self) -> None:
        analysis = AnalysisConfig()

        self.assertEqual(analysis.plots.max_sources, 8)

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
                "save_log_likelihood": False,
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

    def test_split_variant_study_configs_resolve_tracking_and_model_fields(self) -> None:
        config_paths = [
            "configs/experiment_gpt_global.yaml",
            "configs/experiment_gpt_source_hier.yaml",
            "configs/experiment_claude_global.yaml",
            "configs/experiment_claude_source_hier.yaml",
        ]

        configs = [ExperimentConfig.from_yaml(path) for path in config_paths]

        self.assertEqual([config.data.splits for config in configs], [["gpt"], ["gpt"], ["claude"], ["claude"]])
        self.assertEqual(
            [config.model.variant for config in configs],
            ["global", "source_hier", "global", "source_hier"],
        )
        self.assertTrue(all(config.model.type == "2PL" for config in configs))
        self.assertTrue(all(config.inference.save_log_likelihood for config in configs))
        self.assertTrue(all(config.tracking.tracking_db == Path.cwd() / "mlflow.db" for config in configs))
        self.assertTrue(all(config.tracking.artifact_dir == Path.cwd() / "mlruns" for config in configs))

    def test_from_yaml_rejects_legacy_unknown_tracking_field(self) -> None:
        payload = yaml.safe_load(Path("configs/experiment_gpt_global.yaml").read_text(encoding="utf-8"))
        payload["tracking"]["tracking_dir"] = "mlruns"

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "legacy_tracking.yaml"
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Extra inputs are not permitted|tracking_dir"):
                ExperimentConfig.from_yaml(config_path)


if __name__ == "__main__":
    unittest.main()
