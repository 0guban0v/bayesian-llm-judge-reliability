"""Regression tests for report export helpers."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
from src.analysis.report_exports import (
    _study_inferencedata_paths,
    write_cross_run_summary_exports,
    write_model_comparison_exports,
)
from src.schemas import ExperimentConfig


class ReportExportsTests(unittest.TestCase):
    """Verify report exports fail safely when comparison artifacts are unavailable."""

    def test_study_inferencedata_paths_are_derived_from_study_configs(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        paths = _study_inferencedata_paths(config)

        self.assertEqual(set(paths.keys()), {"gpt", "claude"})
        self.assertEqual(set(paths["gpt"].keys()), {"global", "source_hier"})
        self.assertEqual(set(paths["claude"].keys()), {"global", "source_hier"})
        self.assertEqual(
            paths["gpt"]["global"],
            ExperimentConfig.from_yaml("configs/experiment_gpt_global.yaml").inference.inferencedata_path,
        )
        self.assertEqual(
            paths["gpt"]["source_hier"],
            ExperimentConfig.from_yaml("configs/experiment_gpt_source_hier.yaml").inference.inferencedata_path,
        )
        self.assertEqual(
            paths["claude"]["global"],
            ExperimentConfig.from_yaml("configs/experiment_claude_global.yaml").inference.inferencedata_path,
        )
        self.assertEqual(
            paths["claude"]["source_hier"],
            ExperimentConfig.from_yaml("configs/experiment_claude_source_hier.yaml").inference.inferencedata_path,
        )

    def test_study_inferencedata_paths_ignore_unrelated_matching_yaml(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            for file_name in (
                "experiment_gpt_global.yaml",
                "experiment_gpt_source_hier.yaml",
                "experiment_claude_global.yaml",
                "experiment_claude_source_hier.yaml",
            ):
                shutil.copy(Path("configs") / file_name, config_dir / file_name)
            shutil.copy(Path("configs/experiment_gpt_global.yaml"), config_dir / "experiment_gpt_extra.yaml")

            with patch("src.analysis.report_exports.project_root", return_value=root):
                paths = _study_inferencedata_paths(config)

        self.assertEqual(set(paths.keys()), {"gpt", "claude"})
        self.assertEqual(set(paths["gpt"].keys()), {"global", "source_hier"})
        self.assertEqual(set(paths["claude"].keys()), {"global", "source_hier"})

    def test_model_comparison_export_writes_safe_note_when_required_study_config_is_missing(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "configs").mkdir()
            output_dir = root / "generated"

            with patch("src.analysis.report_exports.project_root", return_value=root):
                write_model_comparison_exports(config, output_dir=output_dir)

            content = (output_dir / "model_comparison.tex").read_text(encoding="utf-8")

        self.assertIn("Missing required study config for model comparison", content)

    def test_model_comparison_export_writes_safe_note_when_split_discovery_is_incomplete(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            with patch(
                "src.analysis.report_exports._study_inferencedata_paths",
                return_value={"gpt": {"global": Path("dummy.nc")}, "claude": {"source_hier": Path("dummy.nc")}},
            ):
                write_model_comparison_exports(config, output_dir=output_dir)

            content = (output_dir / "model_comparison.tex").read_text(encoding="utf-8")

        self.assertIn("PSIS-LOO/WAIC comparison is unavailable", content)

    def test_model_comparison_export_writes_safe_note_when_study_files_missing(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            with patch(
                "src.analysis.report_exports._study_inferencedata_paths",
                return_value={
                    "gpt": {"global": Path("missing-global.nc"), "source_hier": Path("missing-source.nc")},
                    "claude": {"global": Path("missing-global.nc"), "source_hier": Path("missing-source.nc")},
                },
            ):
                write_model_comparison_exports(config, output_dir=output_dir)

            content = (output_dir / "model_comparison.tex").read_text(encoding="utf-8")

        self.assertIn("PSIS-LOO/WAIC comparison is unavailable", content)

    def test_model_comparison_export_propagates_unexpected_config_failure(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)

            with patch(
                "src.analysis.report_exports._study_inferencedata_paths",
                side_effect=ValueError("legacy tracking_dir key"),
            ):
                with self.assertRaisesRegex(ValueError, "legacy tracking_dir key"):
                    write_model_comparison_exports(config, output_dir=output_dir)

    def test_cross_run_summary_export_writes_table_with_expected_label(self) -> None:
        config = ExperimentConfig.from_yaml("configs/experiment.yaml")

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            root = Path(tmp_dir)
            run_specs = {
                "configs/experiment.yaml": ("gpt + claude", "source_hier", "pooled"),
                "configs/experiment_gpt_global.yaml": ("gpt", "global", "gpt_global"),
                "configs/experiment_gpt_source_hier.yaml": ("gpt", "source_hier", "gpt_source"),
                "configs/experiment_claude_global.yaml": ("claude", "global", "claude_global"),
                "configs/experiment_claude_source_hier.yaml": ("claude", "source_hier", "claude_source"),
            }
            run_configs: dict[str, SimpleNamespace] = {}
            for relative_path, (split_label, variant, stem) in run_specs.items():
                matrix_path = root / f"{stem}.parquet"
                posterior_path = root / f"{stem}.npz"
                matrix_path.touch()
                posterior_path.touch()
                run_configs[str(root / relative_path)] = SimpleNamespace(
                    data=SimpleNamespace(matrix_path=matrix_path, splits=split_label.split(" + ")),
                    inference=SimpleNamespace(posterior_path=posterior_path),
                    model=SimpleNamespace(variant=variant),
                )

            def fake_load_run_config(_project_root: Path, config_path: Path) -> SimpleNamespace:
                return run_configs[str(root / config_path)]

            def fake_read_parquet(path: Path) -> pl.DataFrame:
                item_count = {
                    "pooled.parquet": 500,
                    "gpt_global.parquet": 350,
                    "gpt_source.parquet": 350,
                    "claude_global.parquet": 270,
                    "claude_source.parquet": 270,
                }[path.name]
                return pl.DataFrame({"item_id": list(range(item_count))})

            def fake_rank_judges(posterior_token: str) -> pl.DataFrame:
                top_judge = (
                    "deepseek-r1-distill-qwen-7b"
                    if posterior_token == "claude_source"
                    else "deepseek-r1-distill-qwen-14b"
                )
                return pl.DataFrame(
                    {
                        "judge_id": [top_judge, "mistral-7b-instruct-v0-3"],
                        "theta_mean": [0.5, 0.2],
                        "theta_p05": [0.1, -0.1],
                        "theta_p95": [0.9, 0.5],
                    }
                )

            def fake_load_posterior(path: Path) -> str:
                return path.stem

            with (
                patch("src.analysis.report_exports._load_run_config", side_effect=fake_load_run_config),
                patch("src.analysis.report_exports.pl.read_parquet", side_effect=fake_read_parquet),
                patch("src.analysis.report_exports.load_posterior", side_effect=fake_load_posterior),
                patch("src.analysis.report_exports.rank_judges", side_effect=fake_rank_judges),
                patch("src.analysis.report_exports.probability_judge_a_exceeds_b", return_value=0.846),
            ):
                write_cross_run_summary_exports(config, output_dir=output_dir)
            content = (output_dir / "cross_run_summary.tex").read_text(encoding="utf-8")

        self.assertIn(r"\label{tab:cross-run-summary}", content)
        self.assertIn("Pooled baseline", content)
        self.assertIn("GPT global", content)
        self.assertIn("Claude source-hier", content)


if __name__ == "__main__":
    unittest.main()
