"""Whitelist framework-driven symbols that Vulture cannot resolve statically."""

from __future__ import annotations

from src.schemas import ExperimentConfig, InferenceConfig, JudgeConfig, JudgeResult

_ = (
    JudgeConfig.validate_prompt_template,
    InferenceConfig.sampler,
    ExperimentConfig.ensure_unique_judge_ids,
    JudgeResult.ground_truth_label,
    JudgeResult.prompt_variant,
)
