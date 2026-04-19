"""Regression tests for report export helpers."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
            write_cross_run_summary_exports(config, output_dir=output_dir)
            content = (output_dir / "cross_run_summary.tex").read_text(encoding="utf-8")

        self.assertIn(r"\label{tab:cross-run-summary}", content)
        self.assertIn("Pooled baseline", content)
        self.assertIn("GPT global", content)
        self.assertIn("Claude source-hier", content)


if __name__ == "__main__":
    unittest.main()
