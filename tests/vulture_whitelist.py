"""Whitelist framework-driven symbols that Vulture cannot resolve statically."""

from __future__ import annotations

from src.analysis.figure_paths import analysis_figure_paths
from src.judges.parsers import JudgeOutput
from src.schemas import ExperimentConfig, InferenceConfig, JudgeConfig, JudgeResult

_ = (
    analysis_figure_paths,
    JudgeOutput.extract_verdict,
    JudgeConfig.validate_prompt_template,
    InferenceConfig.sampler,
    ExperimentConfig.ensure_unique_judge_ids,
    JudgeResult.ground_truth_label,
    JudgeResult.prompt_variant,
)
