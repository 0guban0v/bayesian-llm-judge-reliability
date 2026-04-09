"""Typed configuration and result schemas."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROMPT_TEMPLATE_NAMES = {"pointwise", "pointwise_cot", "pairwise", "pairwise_cot"}


class ExperimentMetadata(BaseModel):
    """Top-level experiment metadata."""

    name: str
    seed: int = Field(gt=0)
    date: date


class DataConfig(BaseModel):
    """Dataset and artifact paths for JudgeBench processing."""

    source: Literal["judgebench"]
    hf_dataset: str = "ScalerLab/JudgeBench"
    splits: list[str] = Field(default_factory=lambda: ["gpt", "claude"])
    subset_size: int = Field(gt=0)
    categories: list[str] = Field(default_factory=list)
    output_dir: Path = Path("data/processed")
    raw_dir: Path = Path("data/raw")
    logs_dir: Path = Path("data/logs")
    item_file: str = "judgebench_items.parquet"
    matrix_file: str = "judge_matrix.parquet"

    @property
    def item_path(self) -> Path:
        """Return the processed JudgeBench item parquet path."""

        return self.output_dir / self.item_file

    @property
    def matrix_path(self) -> Path:
        """Return the judge matrix parquet path."""

        return self.output_dir / self.matrix_file


class JudgeConfig(BaseModel):
    """Configuration for a single local MLX judge."""

    id: str
    backend: Literal["mlx"] = "mlx"
    model: str
    prompt_template: str
    max_tokens: int = Field(gt=0, default=256)
    trust_remote_code: bool = False
    reverse_order: bool = False

    @field_validator("prompt_template")
    @classmethod
    def validate_prompt_template(cls, value: str) -> str:
        """Restrict prompt template names to supported variants."""

        if value not in PROMPT_TEMPLATE_NAMES:
            expected = ", ".join(sorted(PROMPT_TEMPLATE_NAMES))
            msg = f"Unsupported prompt template '{value}'. Expected one of: {expected}"
            raise ValueError(msg)
        return value


class PriorConfig(BaseModel):
    """Distribution specification for a model prior."""

    dist: Literal["normal", "lognormal"]
    loc: float
    scale: float = Field(gt=0.0)


class PriorsConfig(BaseModel):
    """Grouped priors for the IRT model."""

    theta: PriorConfig
    b: PriorConfig
    a: PriorConfig


class IRTConfig(BaseModel):
    """Bayesian IRT model specification."""

    type: Literal["1PL", "2PL"]
    priors: PriorsConfig


class InferenceConfig(BaseModel):
    """Inference hyperparameters for NumPyro or BlackJAX samplers."""

    backend: Literal["numpyro", "blackjax"] = "numpyro"
    sampler: Literal["NUTS"]
    num_warmup: int = Field(gt=0)
    num_samples: int = Field(gt=0)
    num_chains: int = Field(gt=0)
    target_accept_prob: float = Field(gt=0.0, lt=1.0)
    output_dir: Path = Path("data/processed/posteriors")
    file_name: str = "irt_posterior.npz"

    def posterior_path_for_backend(self, backend: Literal["numpyro", "blackjax"]) -> Path:
        """Return the backend-specific posterior output path."""

        if backend == "numpyro":
            return self.output_dir / self.file_name
        file_path = Path(self.file_name)
        stem = file_path.stem
        suffix = file_path.suffix or ".npz"
        if stem.endswith("_blackjax"):
            return self.output_dir / f"{stem}{suffix}"
        return self.output_dir / f"{stem}_blackjax{suffix}"

    @property
    def posterior_path(self) -> Path:
        """Return the configured backend's posterior output path."""

        return self.posterior_path_for_backend(self.backend)


class ExperimentConfig(BaseModel):
    """Single source of truth for the experiment pipeline."""

    experiment: ExperimentMetadata
    data: DataConfig
    judges: list[JudgeConfig]
    inference: InferenceConfig
    model: IRTConfig

    @model_validator(mode="after")
    def ensure_unique_judge_ids(self) -> ExperimentConfig:
        """Validate that judge identifiers are unique."""

        judge_ids = [judge.id for judge in self.judges]
        if len(judge_ids) != len(set(judge_ids)):
            raise ValueError("Judge IDs must be unique.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an experiment configuration from YAML."""

        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)

    def ensure_directories(self) -> None:
        """Create the directories used by the pipeline."""

        for path in (
            self.data.output_dir,
            self.data.raw_dir,
            self.data.logs_dir,
            self.inference.output_dir,
            Path("figures"),
            Path("report"),
        ):
            path.mkdir(parents=True, exist_ok=True)


class JudgeResult(BaseModel):
    """Structured JSONL record for a single judge decision."""

    item_id: str
    judge_id: str
    timestamp: datetime
    source: str
    question: str
    ground_truth_label: Literal["A>B", "B>A"]
    prompt_variant: str
    prompt_order: Literal["original", "reversed"]
    raw_response: str
    parsed_verdict: Literal["A", "B", "TIE", "UNKNOWN"] | None
    correct: bool | None
    latency_ms: int = Field(ge=0)

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""

        payload = self.model_dump()
        payload["timestamp"] = self.timestamp.isoformat()
        return payload
