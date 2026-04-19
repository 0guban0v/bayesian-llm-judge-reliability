"""Whitelist framework-driven symbols that Vulture cannot resolve statically."""

from __future__ import annotations

from src.analysis.figure_paths import analysis_figure_paths
from src.analysis.plot_config import source_color_map
from src.judges.parsers import JudgeOutput
from src.judges.runner import judge_metadata_fields
from src.models.irt_common import sample_prior_values
from src.schemas import ExperimentConfig, InferenceConfig, JudgeResult

_ = (
    analysis_figure_paths,
    source_color_map,
    judge_metadata_fields,
    JudgeOutput.extract_verdict,
    InferenceConfig.sampler,
    ExperimentConfig.ensure_unique_judge_ids,
    JudgeResult.ground_truth_label,
    JudgeResult.item_key,
    JudgeResult.prompt_variant,
    sample_prior_values,
)
