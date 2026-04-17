"""Shared helpers for Bayesian IRT inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.analysis.posterior_archive import POSTERIOR_SCHEMA_VERSION
from src.data.matrix_semantics import ITEM_METADATA_COLUMNS
from src.schemas import IRTConfig


@dataclass(frozen=True)
class PriorSpec:
    """Concrete prior parameters for a single latent variable."""

    dist: str
    loc: float
    scale: float


@dataclass(frozen=True)
class ModelPriors:
    """Fully bound prior parameters for the configured IRT model."""

    theta: PriorSpec
    b: PriorSpec
    a: PriorSpec
    tau_theta: PriorSpec | None = None


def build_model_priors(model_config: IRTConfig) -> ModelPriors:
    """Return immutable prior parameters derived from the experiment config."""

    return ModelPriors(
        theta=PriorSpec(
            dist=model_config.priors.theta.dist,
            loc=model_config.priors.theta.loc,
            scale=model_config.priors.theta.scale,
        ),
        b=PriorSpec(
            dist=model_config.priors.b.dist,
            loc=model_config.priors.b.loc,
            scale=model_config.priors.b.scale,
        ),
        a=PriorSpec(
            dist=model_config.priors.a.dist,
            loc=model_config.priors.a.loc,
            scale=model_config.priors.a.scale,
        ),
        tau_theta=(
            PriorSpec(
                dist=model_config.priors.tau_theta.dist,
                loc=model_config.priors.tau_theta.loc,
                scale=model_config.priors.tau_theta.scale,
            )
            if model_config.priors.tau_theta is not None
            else None
        ),
    )


def sample_prior_values(
    prior: PriorSpec,
    *,
    rng: np.random.Generator,
    size: tuple[int, ...],
) -> np.ndarray:
    """Draw prior samples using the configured distribution family."""

    if prior.dist == "normal":
        return rng.normal(prior.loc, prior.scale, size=size)
    if prior.dist == "lognormal":
        return rng.lognormal(prior.loc, prior.scale, size=size)
    raise ValueError(f"Unsupported prior distribution '{prior.dist}' for prior predictive draw")


def aggregate_judge_accuracy_ppc(
    posterior_predictive_correct: np.ndarray,
    observations: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Aggregate posterior predictive Bernoulli draws into per-judge accuracy summaries."""

    flattened_draws = np.asarray(posterior_predictive_correct, dtype=float).reshape(
        -1,
        posterior_predictive_correct.shape[-1],
    )
    judge_idx = np.asarray(observations["judge_idx"], dtype=int)
    n_judges = int(observations["n_judges"])
    judge_accuracy_draws = np.empty((flattened_draws.shape[0], n_judges), dtype=float)
    for judge_index in range(n_judges):
        judge_mask = judge_idx == judge_index
        if not np.any(judge_mask):
            raise ValueError(f"No posterior predictive observations found for judge index {judge_index}")
        judge_accuracy_draws[:, judge_index] = flattened_draws[:, judge_mask].mean(axis=1)
    return {
        "judge_accuracy_ppc_mean": judge_accuracy_draws.mean(axis=0),
        "judge_accuracy_ppc_p05": np.quantile(judge_accuracy_draws, 0.05, axis=0),
        "judge_accuracy_ppc_p95": np.quantile(judge_accuracy_draws, 0.95, axis=0),
    }


def load_matrix_observations(matrix: pl.DataFrame | Path) -> dict[str, Any]:
    """Convert a wide item-by-judge matrix into long-form IRT observations."""

    prepared_matrix = pl.read_parquet(matrix) if isinstance(matrix, Path) else matrix
    judge_ids = [column for column in prepared_matrix.columns if column not in ITEM_METADATA_COLUMNS]
    item_ids = prepared_matrix.get_column("item_id").to_list()
    source_ids = prepared_matrix.get_column("source").unique(maintain_order=True).to_list()
    item_lookup = pl.DataFrame(
        {
            "item_id": item_ids,
            "item_idx": np.arange(len(item_ids), dtype=np.int32),
            "source": prepared_matrix.get_column("source"),
        }
    )
    judge_lookup = pl.DataFrame(
        {
            "judge_id": judge_ids,
            "judge_idx": np.arange(len(judge_ids), dtype=np.int32),
        }
    )
    source_lookup = pl.DataFrame(
        {
            "source": source_ids,
            "source_idx": np.arange(len(source_ids), dtype=np.int32),
        }
    )
    observations = (
        prepared_matrix.select(["item_id", *judge_ids])
        .unpivot(
            on=judge_ids,
            index="item_id",
            variable_name="judge_id",
            value_name="correct",
        )
        .drop_nulls("correct")
        .join(item_lookup, on="item_id", how="left")
        .join(judge_lookup, on="judge_id", how="left")
        .join(source_lookup, on="source", how="left")
        .sort(["item_idx", "judge_idx"])
    )
    return {
        "correct": observations.get_column("correct").cast(pl.Int32).to_numpy(),
        "judge_idx": observations.get_column("judge_idx").to_numpy(),
        "item_idx": observations.get_column("item_idx").to_numpy(),
        "source_idx": observations.get_column("source_idx").to_numpy(),
        "n_judges": len(judge_ids),
        "n_items": len(item_ids),
        "n_sources": len(source_ids),
        "judge_ids": np.asarray(judge_ids, dtype=str),
        "item_ids": np.asarray(item_ids, dtype=str),
        "source_ids": np.asarray(source_ids, dtype=str),
    }


def save_posterior(
    path: Path,
    samples: dict[str, np.ndarray],
    observations: dict[str, Any],
    model_type: str,
    metadata: dict[str, np.ndarray] | None = None,
) -> None:
    """Persist posterior samples and metadata to `.npz`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **samples,
        "judge_ids": observations["judge_ids"],
        "item_ids": observations["item_ids"],
        "source_ids": observations["source_ids"],
        "model_type": np.asarray(model_type),
        "n_obs": np.asarray(observations["correct"].shape[0]),
        "posterior_schema_version": np.asarray(POSTERIOR_SCHEMA_VERSION),
    }
    if metadata is not None:
        payload.update(metadata)
    np.savez(path, **payload)


def summarize_judges(samples: dict[str, np.ndarray], judge_ids: np.ndarray) -> pl.DataFrame:
    """Summarize posterior judge reliabilities."""

    theta_samples = samples["theta"].reshape(-1, samples["theta"].shape[-1])
    return pl.DataFrame(
        {
            "judge_id": judge_ids.tolist(),
            "theta_mean": theta_samples.mean(axis=0),
            "theta_sd": theta_samples.std(axis=0),
        }
    ).sort("theta_mean", descending=True)


def summarize_item_parameters(samples: dict[str, np.ndarray]) -> pl.DataFrame:
    """Summarize posterior item parameter distributions compactly for CLI logs."""

    rows: list[dict[str, float | str]] = []
    b_samples = samples["b"].reshape(-1, samples["b"].shape[-1])
    rows.append(
        {
            "parameter": "b",
            "mean": float(b_samples.mean()),
            "sd": float(b_samples.std()),
            "min": float(b_samples.mean(axis=0).min()),
            "max": float(b_samples.mean(axis=0).max()),
        }
    )
    if "a" in samples:
        a_samples = samples["a"].reshape(-1, samples["a"].shape[-1])
        rows.append(
            {
                "parameter": "a",
                "mean": float(a_samples.mean()),
                "sd": float(a_samples.std()),
                "min": float(a_samples.mean(axis=0).min()),
                "max": float(a_samples.mean(axis=0).max()),
            }
        )
    return pl.DataFrame(rows)
