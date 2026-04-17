"""Shared figure output names and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PRIOR_PREDICTIVE_STEM: Final[str] = "prior_predictive_probabilities"
JUDGE_ACCURACY_PPC_STEM: Final[str] = "judge_accuracy_ppc"
JUDGE_RELIABILITY_RIDGE_STEM: Final[str] = "judge_reliability_ridge"
JUDGE_RELIABILITY_BY_SOURCE_STEM: Final[str] = "judge_reliability_by_source"
DIAGNOSTICS_SUMMARY_STEM: Final[str] = "diagnostics_summary"


def figure_base_path(figures_dir: Path, stem: str) -> Path:
    """Return the output path stem for a named figure artifact."""

    return figures_dir / stem


def figure_output_path(figures_dir: Path, stem: str) -> Path:
    """Return the PNG output path for a named figure artifact."""

    return figure_base_path(figures_dir, stem).with_suffix(".png")


def analysis_figure_paths(figures_dir: Path) -> dict[str, Path]:
    """Return canonical PNG paths for analysis figures keyed by stem."""

    return {
        PRIOR_PREDICTIVE_STEM: figure_output_path(figures_dir, PRIOR_PREDICTIVE_STEM),
        JUDGE_ACCURACY_PPC_STEM: figure_output_path(figures_dir, JUDGE_ACCURACY_PPC_STEM),
        JUDGE_RELIABILITY_RIDGE_STEM: figure_output_path(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM),
        JUDGE_RELIABILITY_BY_SOURCE_STEM: figure_output_path(
            figures_dir,
            JUDGE_RELIABILITY_BY_SOURCE_STEM,
        ),
        DIAGNOSTICS_SUMMARY_STEM: figure_output_path(figures_dir, DIAGNOSTICS_SUMMARY_STEM),
    }


def remove_figure_output(figures_dir: Path, stem: str) -> Path:
    """Remove a named figure PNG if it exists and return its path."""

    path = figure_output_path(figures_dir, stem)
    path.unlink(missing_ok=True)
    return path
