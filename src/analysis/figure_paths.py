"""Shared figure output names and path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PRIOR_PREDICTIVE_STEM: Final[str] = "prior_predictive_probabilities"
JUDGE_RELIABILITY_RIDGE_STEM: Final[str] = "judge_reliability_ridge"
JUDGE_RELIABILITY_BY_SOURCE_STEM: Final[str] = "judge_reliability_by_source"


def figure_base_path(figures_dir: Path, stem: str) -> Path:
    """Return the output path stem for a named figure artifact."""

    return figures_dir / stem
