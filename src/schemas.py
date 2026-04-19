"""Typed configuration and result schemas."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr, model_validator


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
    max_tokens: int = Field(gt=0, default=256)
    trust_remote_code: bool = False
    reverse_order: bool = False


class PriorConfig(BaseModel):
    """Location and scale parameters for a model prior."""

    dist: Literal["normal", "lognormal"]
    loc: float
    scale: float = Field(gt=0.0)


class PriorsConfig(BaseModel):
    """Grouped priors for the IRT model."""

    theta: PriorConfig
    b: PriorConfig
    a: PriorConfig
    tau_theta: PriorConfig | None = None


class IRTConfig(BaseModel):
    """Bayesian IRT model specification."""

    type: Literal["1PL", "2PL"]
    variant: Literal["global", "source_hier"] = "global"
    priors: PriorsConfig


class InferenceConfig(BaseModel):
    """Inference hyperparameters for NUTS sampling."""

    sampler: Literal["NUTS"]
    num_warmup: int = Field(gt=0)
    num_samples: int = Field(gt=0)
    num_chains: int = Field(gt=0)
    target_accept_prob: float = Field(gt=0.0, lt=1.0)
    output_dir: Path = Path("data/processed/posteriors")
    file_name: str = "irt_posterior.npz"

    @property
    def posterior_path(self) -> Path:
        """Return the posterior output path."""

        return self.output_dir / self.file_name

    @property
    def inferencedata_path(self) -> Path:
        """Return the ArviZ InferenceData NetCDF path saved alongside the posterior archive."""

        return self.output_dir / Path(self.file_name).with_suffix(".nc").name


class AnalysisPlotsConfig(BaseModel):
    """Configurable plotting defaults for analysis outputs."""

    max_sources: int = Field(gt=0, default=8)


class AnalysisReportConfig(BaseModel):
    """Configurable reporting defaults for generated exports."""

    standout_judge_id: str = "deepseek-r1-distill-qwen-14b"
    standout_case_limit: int = Field(gt=0, default=3)
    response_synopsis_chars: int = Field(gt=0, default=120)


class AnalysisConfig(BaseModel):
    """Analysis and report policy defaults."""

    plots: AnalysisPlotsConfig = Field(default_factory=AnalysisPlotsConfig)
    report: AnalysisReportConfig = Field(default_factory=AnalysisReportConfig)


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    backend: Literal["mlflow"] = "mlflow"
    experiment_name: str = "bayesian-llm-judge-reliability"
    tracking_dir: Path = Path("mlruns")


class ExperimentConfig(BaseModel):
    """Single source of truth for the experiment pipeline."""

    experiment: ExperimentMetadata
    data: DataConfig
    judges: list[JudgeConfig]
    inference: InferenceConfig
    model: IRTConfig
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    _project_root: Path = PrivateAttr(default=Path.cwd())

    @model_validator(mode="after")
    def ensure_unique_judge_ids(self) -> ExperimentConfig:
        """Validate that judge identifiers are present and unique."""

        if not self.judges:
            raise ValueError("At least one judge must be configured.")
        judge_ids = [judge.id for judge in self.judges]
        if len(judge_ids) != len(set(judge_ids)):
            raise ValueError("Judge IDs must be unique.")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load an experiment configuration from YAML."""

        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        config = cls.model_validate(payload)
        project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
        config._project_root = project_root
        config.data.output_dir = _resolve_project_path(project_root, config.data.output_dir)
        config.data.raw_dir = _resolve_project_path(project_root, config.data.raw_dir)
        config.data.logs_dir = _resolve_project_path(project_root, config.data.logs_dir)
        config.inference.output_dir = _resolve_project_path(project_root, config.inference.output_dir)
        config.tracking.tracking_dir = _resolve_project_path(project_root, config.tracking.tracking_dir)
        return config

    @property
    def figures_dir(self) -> Path:
        """Return the repository-relative figures directory."""

        return self._project_root / "figures"

    @property
    def report_dir(self) -> Path:
        """Return the repository-relative report directory."""

        return self._project_root / "report"

    @property
    def tracking_uri(self) -> str:
        """Return the MLflow tracking URI for this experiment."""

        return self.tracking.tracking_dir.resolve().as_uri()

    @property
    def tracked_output_dir(self) -> Path:
        """Return the config-scoped local staging directory for tracked run artifacts."""

        return self._project_root / ".tracked_runs" / self.experiment.name

    @property
    def tracked_figures_dir(self) -> Path:
        """Return the config-scoped figure staging directory for tracked runs."""

        return self.tracked_output_dir / "figures"

    @property
    def tracked_report_generated_dir(self) -> Path:
        """Return the config-scoped generated-report staging directory for tracked runs."""

        return self.tracked_output_dir / "report_generated"

    def ensure_directories(self) -> None:
        """Create the directories used by the pipeline."""

        for path in (
            self.data.output_dir,
            self.data.raw_dir,
            self.data.logs_dir,
            self.inference.output_dir,
            self.figures_dir,
            self.report_dir,
            self.tracking.tracking_dir,
            self.tracked_figures_dir,
            self.tracked_report_generated_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class JudgeResult(BaseModel):
    """Structured JSONL record for a single judge decision."""

    item_id: str
    item_key: str
    judge_id: str
    timestamp: datetime
    source: str
    question: str
    ground_truth_label: Literal["A>B", "B>A"]
    prompt_variant: str
    prompt_protocol_version: str
    prompt_order: Literal["original", "reversed"]
    model: str
    max_tokens: int = Field(gt=0)
    trust_remote_code: bool = False
    reverse_order: bool = False
    raw_response: str
    parsed_verdict: Literal["A", "B"] | None
    correct: bool | None
    latency_ms: int = Field(ge=0)

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary."""

        payload = self.model_dump()
        payload["timestamp"] = self.timestamp.isoformat()
        return payload


def unique_model_requests(judges: list[JudgeConfig]) -> list[tuple[str, bool]]:
    """Return distinct `(model, trust_remote_code)` pairs preserving config order."""

    seen: set[tuple[str, bool]] = set()
    ordered: list[tuple[str, bool]] = []
    for judge in judges:
        key = (judge.model, judge.trust_remote_code)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _resolve_project_path(project_root: Path, path: Path) -> Path:
    """Resolve a config path relative to the repository root."""

    if path.is_absolute():
        return path
    return project_root / path
