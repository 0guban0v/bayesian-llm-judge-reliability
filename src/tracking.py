"""Local MLflow tracking helpers for experiment runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from src.analysis.posterior_queries import probability_judge_a_exceeds_b, rank_judges
from src.analysis.posterior_utils import flatten_draws
from src.schemas import ExperimentConfig

PAIRWISE_SUPPORT_THRESHOLD = 0.8


def normalize_process_ru_maxrss_bytes(raw_ru_maxrss: int, current_platform: str) -> int:
    """Normalize process ru_maxrss into bytes across platforms."""

    normalized = max(0, raw_ru_maxrss)
    if current_platform == "darwin":
        return normalized
    return normalized * 1024


def total_ram_bytes() -> int | None:
    """Return total physical RAM in bytes when available."""

    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        return None
    if page_size <= 0 or phys_pages <= 0:
        return None
    return page_size * phys_pages


def inferred_accelerator() -> str:
    """Return a compact inferred accelerator label for the current host."""

    if sys.platform == "darwin" and platform.machine().lower() == "arm64":
        return "apple_metal"
    if sys.platform == "darwin":
        return "metal"
    return "cpu"


def _read_command_output(command: list[str]) -> str | None:
    """Return stripped command output or None when the probe is unavailable."""

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def cpu_name() -> str:
    """Return the best available CPU / SoC name for the current host."""

    candidates = [
        platform.processor(),
        platform.uname().processor,
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
    ]
    for candidate in candidates:
        value = candidate.strip()
        if value and value.lower() != "unknown":
            return value
    if sys.platform == "darwin":
        return _read_command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"
    return "unknown"


def system_telemetry_params() -> dict[str, str | int]:
    """Return static system/runtime fields worth logging once per run."""

    params: dict[str, str | int] = {
        "host_platform": sys.platform,
        "host_machine": platform.machine(),
        "cpu_name": cpu_name(),
        "cpu_count_logical": os.cpu_count() or 0,
        "python_version": platform.python_version(),
        "accelerator": inferred_accelerator(),
    }
    if sys.platform == "darwin" and params["cpu_name"].startswith("Apple "):
        params["apple_soc"] = params["cpu_name"]
    ram_bytes = total_ram_bytes()
    if ram_bytes is not None:
        params["total_ram_bytes"] = ram_bytes
    return params


def process_telemetry_metrics() -> dict[str, float]:
    """Return dynamic process-level telemetry metrics."""

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_bytes = normalize_process_ru_maxrss_bytes(usage.ru_maxrss, sys.platform)
    return {"peak_rss_bytes": float(peak_rss_bytes), "peak_rss_gb": peak_rss_bytes / (1024**3)}


def _mlflow():
    import mlflow

    return mlflow


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(config: ExperimentConfig) -> str:
    """Return a compact MLflow run name for the current config."""

    split_token = "-".join(config.data.splits)
    return f"{config.experiment.name}-{split_token}-{config.model.type}-{config.model.variant}"


@contextmanager
def tracked_run(config: ExperimentConfig) -> Iterator[object]:
    """Open an MLflow run for the configured experiment."""

    mlflow = _mlflow()
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)
    with mlflow.start_run(run_name=run_name(config)) as run:
        mlflow.set_tags(
            {
                "experiment_name": config.experiment.name,
                "model_type": config.model.type,
                "model_variant": config.model.variant,
                "data_splits": ",".join(config.data.splits),
                "data_subset_size": str(config.data.subset_size),
                "experiment_seed": str(config.experiment.seed),
            }
        )
        yield run


def log_config(config: ExperimentConfig) -> None:
    """Log config parameters and the resolved YAML payload."""

    mlflow = _mlflow()
    mlflow.log_params(
        {
            "experiment_name": config.experiment.name,
            "seed": config.experiment.seed,
            "subset_size": config.data.subset_size,
            "splits": ",".join(config.data.splits),
            "model_type": config.model.type,
            "model_variant": config.model.variant,
            "num_warmup": config.inference.num_warmup,
            "num_samples": config.inference.num_samples,
            "num_chains": config.inference.num_chains,
            "target_accept_prob": config.inference.target_accept_prob,
            "theta_prior_dist": config.model.priors.theta.dist,
            "theta_prior_loc": config.model.priors.theta.loc,
            "theta_prior_scale": config.model.priors.theta.scale,
            "b_prior_dist": config.model.priors.b.dist,
            "b_prior_loc": config.model.priors.b.loc,
            "b_prior_scale": config.model.priors.b.scale,
        }
    )
    mlflow.log_params(
        {
            "a_prior_dist": config.model.priors.a.dist,
            "a_prior_loc": config.model.priors.a.loc,
            "a_prior_scale": config.model.priors.a.scale,
        }
    )
    if config.model.priors.tau_theta is not None:
        mlflow.log_params(
            {
                "tau_theta_prior_dist": config.model.priors.tau_theta.dist,
                "tau_theta_prior_loc": config.model.priors.tau_theta.loc,
                "tau_theta_prior_scale": config.model.priors.tau_theta.scale,
            }
        )
    payload = yaml.safe_dump(json.loads(config.model_dump_json()), sort_keys=False)
    mlflow.log_text(payload, "resolved_config.yaml")
    mlflow.log_params(system_telemetry_params())


def log_data_artifacts(config: ExperimentConfig) -> None:
    """Log item and matrix artifact hashes."""

    mlflow = _mlflow()
    artifact_meta: dict[str, str] = {}
    if config.data.item_path.exists():
        artifact_meta["item_path"] = str(config.data.item_path)
        artifact_meta["item_sha256"] = file_sha256(config.data.item_path)
    if config.data.matrix_path.exists():
        artifact_meta["matrix_path"] = str(config.data.matrix_path)
        artifact_meta["matrix_sha256"] = file_sha256(config.data.matrix_path)
    if artifact_meta:
        mlflow.log_params(artifact_meta)


def rank_order_string(posterior: dict[str, np.ndarray]) -> str:
    """Return a stable rank order string for judge means."""

    ranking = rank_judges(posterior)
    return ">".join(ranking.get_column("judge_id").cast(str).to_list())


def resolved_pairwise_count(posterior: dict[str, np.ndarray], threshold: float = PAIRWISE_SUPPORT_THRESHOLD) -> int:
    """Return the number of pairwise comparisons with at least threshold support in either direction."""

    judge_ids = [str(judge_id) for judge_id in posterior["judge_ids"]]
    count = 0
    for judge_a, judge_b in combinations(judge_ids, 2):
        probability = probability_judge_a_exceeds_b(posterior, judge_a, judge_b)
        if probability >= threshold or probability <= 1.0 - threshold:
            count += 1
    return count


def log_posterior_metrics(config: ExperimentConfig, posterior: dict[str, np.ndarray]) -> None:
    """Log judge summaries, pairwise comparisons, and derived ranking metrics."""

    mlflow = _mlflow()
    ranking = rank_judges(posterior)
    theta_samples = flatten_draws(posterior["theta"])
    judge_ids = [str(judge_id) for judge_id in posterior["judge_ids"]]
    top_judge = str(ranking.item(0, "judge_id"))
    second_judge = str(ranking.item(1, "judge_id")) if ranking.height > 1 else top_judge
    top_vs_second = probability_judge_a_exceeds_b(posterior, top_judge, second_judge)
    mlflow.log_metrics(
        {
            "top_vs_second_probability": top_vs_second,
            "resolved_pairwise_count": float(resolved_pairwise_count(posterior)),
            "theta_range": float(theta_samples.mean(axis=0).max() - theta_samples.mean(axis=0).min()),
        }
    )
    mlflow.log_params({"top_judge": top_judge, "rank_order": rank_order_string(posterior)})
    for row in ranking.to_dicts():
        judge_id = str(row["judge_id"])
        mlflow.log_metrics(
            {
                f"theta_mean__{judge_id}": float(row["theta_mean"]),
                f"theta_p05__{judge_id}": float(row["theta_p05"]),
                f"theta_p95__{judge_id}": float(row["theta_p95"]),
            }
        )
    for judge_a, judge_b in combinations(judge_ids, 2):
        probability = probability_judge_a_exceeds_b(posterior, judge_a, judge_b)
        mlflow.log_metrics({f"pairwise__{judge_a}__gt__{judge_b}": probability})
    if config.inference.posterior_path.exists():
        mlflow.log_param("posterior_path", str(config.inference.posterior_path))
        mlflow.log_artifact(str(config.inference.posterior_path), artifact_path="posterior")
    if config.inference.inferencedata_path.exists():
        mlflow.log_param("inferencedata_path", str(config.inference.inferencedata_path))
        mlflow.log_artifact(str(config.inference.inferencedata_path), artifact_path="posterior")


def log_metrics(metrics: dict[str, float]) -> None:
    """Log a flat metric mapping."""

    _mlflow().log_metrics(metrics)


def log_artifact(path: Path, artifact_path: str) -> None:
    """Log a file artifact if it exists."""

    if path.exists():
        _mlflow().log_artifact(str(path), artifact_path=artifact_path)
