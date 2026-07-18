"""Regression tests for configuration schema helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml
from src.schemas import (
    AnalysisConfig,
    DataConfig,
    ExperimentConfig,
    InferenceConfig,
    JudgeConfig,
    JudgeResult,
    PriorConfig,
)


def judge_result_payload() -> dict[str, object]:
    """Return minimal valid persisted judge result content."""

    return {
        "item_id": "item-1",
        "item_key": "gpt:item-1",
        "item_content_hash": "0" * 64,
        "judge_id": "judge-1",
        "timestamp": "2026-04-16T00:00:00+00:00",
        "source": "source-1",
        "question": "question-1",
        "ground_truth_label": "A>B",
        "prompt_variant": "fixed_verdict_only",
        "prompt_protocol_version": "v1",
        "prompt_order": "original",
        "repeat_index": 0,
        "model": "model-1",
        "max_tokens": 8,
        "trust_remote_code": False,
        "reverse_order": False,
        "raw_response": "FINAL VERDICT: A",
        "parsed_verdict": "A",
        "correct": True,
        "latency_ms": 10,
    }


class JudgeResultSchemaTests(unittest.TestCase):
    """Verify judge logs require canonical item content hashes."""

    def test_requires_item_content_hash(self) -> None:
        payload = judge_result_payload()
        del payload["item_content_hash"]

        with self.assertRaisesRegex(ValueError, "(?s)item_content_hash.*Field required"):
            JudgeResult.model_validate(payload)

    def test_rejects_malformed_item_content_hash(self) -> None:
        payload = judge_result_payload()
        payload["item_content_hash"] = "not-a-sha256"

        with self.assertRaisesRegex(ValueError, "(?s)item_content_hash.*String should match pattern"):
            JudgeResult.model_validate(payload)

    def test_requires_repeat_index(self) -> None:
        payload = judge_result_payload()
        del payload["repeat_index"]

        with self.assertRaisesRegex(ValueError, "(?s)repeat_index.*Field required"):
            JudgeResult.model_validate(payload)

    def test_rejects_negative_repeat_index(self) -> None:
        payload = judge_result_payload()
        payload["repeat_index"] = -1

        with self.assertRaisesRegex(ValueError, "(?s)repeat_index.*greater than or equal to 0"):
            JudgeResult.model_validate(payload)


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


class DataConfigTests(unittest.TestCase):
    """Verify duplicate-judgment policy configuration."""

    def test_repeat_policy_defaults_to_reject(self) -> None:
        config = DataConfig(source="judgebench", subset_size=1)

        self.assertEqual(config.repeat_policy, "reject")

    def test_rejects_unknown_repeat_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "Input should be 'reject', 'first' or 'latest'"):
            DataConfig(source="judgebench", subset_size=1, repeat_policy="newest")


class JudgeConfigTests(unittest.TestCase):
    """Verify repeated-inference scheduling configuration."""

    def test_num_repeats_defaults_to_one(self) -> None:
        config = JudgeConfig(id="judge-a", model="model-a")

        self.assertEqual(config.num_repeats, 1)

    def test_num_repeats_requires_positive_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "(?s)num_repeats.*greater than or equal to 1"):
            JudgeConfig(id="judge-a", model="model-a", num_repeats=0)


class ExperimentConfigTests(unittest.TestCase):
    """Verify resolved experiment configuration."""

    def test_from_yaml_resolves_repo_relative_paths(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        self.assertTrue(config.data.output_dir.is_absolute())
        self.assertTrue(config.data.raw_dir.is_absolute())
        self.assertTrue(config.data.logs_dir.is_absolute())
        self.assertTrue(config.inference.output_dir.is_absolute())
        self.assertEqual(config.figures_dir, Path.cwd() / "figures")
        self.assertEqual(config.report_dir, Path.cwd() / "report")
        self.assertEqual(config.data.repeat_policy, "reject")

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
