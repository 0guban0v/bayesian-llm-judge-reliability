"""Shared figure output names and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PRIOR_PREDICTIVE_STEM: Final[str] = "prior_predictive_probabilities"
JUDGE_RELIABILITY_RIDGE_STEM: Final[str] = "judge_reliability_ridge"
ITEM_PARAMETER_SCATTER_STEM: Final[str] = "item_parameter_scatter"
POSTERIOR_PREDICTIVE_STEM: Final[str] = "posterior_predictive"
JUDGE_RELIABILITY_BY_SOURCE_STEM: Final[str] = "judge_reliability_by_source"


def figure_base_path(figures_dir: Path, stem: str) -> Path:
    """Return the output path stem for a named figure artifact."""

    return figures_dir / stem


def figure_png_path(figures_dir: Path, stem: str) -> Path:
    """Return the PNG output path for a named figure artifact."""

    return figure_base_path(figures_dir, stem).with_suffix(".png")
