"""Shared posterior-data helpers used by diagnostics, plots, and reports."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.matrix_semantics import judge_columns, observed_accuracy_frame


def _flatten_parameter(samples: np.ndarray) -> np.ndarray:
    """Flatten all parameter dimensions while preserving chain and draw axes."""

    if samples.ndim < 2:
        raise ValueError("Posterior arrays must have chain and draw axes.")
    return samples.reshape(samples.shape[0], samples.shape[1], -1)


def flatten_draws(samples: np.ndarray) -> np.ndarray:
    """Collapse chain and draw axes into a single posterior sample axis."""

    flattened = _flatten_parameter(samples)
    return flattened.reshape(-1, flattened.shape[-1])


def has_source_reliability(posterior: dict[str, np.ndarray] | object | None) -> bool:
    """Return whether a posterior includes source-specific reliability samples."""

    if posterior is None:
        return False
    if isinstance(posterior, dict):
        payload = posterior
    else:
        payload = getattr(posterior, "payload", None)
        if not isinstance(payload, dict):
            return False
    return payload.get("theta_source") is not None and payload.get("source_ids") is not None


def observed_accuracy(matrix: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return judge IDs and observed accuracies from the processed matrix."""

    observed = observed_accuracy_frame(matrix, judge_columns(matrix))
    return (
        observed.get_column("judge_id").to_numpy().astype(str),
        observed.get_column("accuracy").to_numpy().astype(float),
    )


def validate_posterior_judge_order(
    matrix_judge_ids: np.ndarray,
    posterior: dict[str, np.ndarray],
) -> None:
    """Ensure posterior judge order matches the processed matrix column order."""

    posterior_judge_ids = np.asarray(posterior["judge_ids"], dtype=str)
    if np.array_equal(matrix_judge_ids, posterior_judge_ids):
        return
    msg = (
        "Posterior judge order does not match matrix judge column order. "
        f"matrix={matrix_judge_ids.tolist()} posterior={posterior_judge_ids.tolist()}"
    )
    raise ValueError(msg)


def validate_judge_accuracy_ppc_summaries(posterior: dict[str, np.ndarray]) -> None:
    """Ensure posterior archive contains judge-level PPC summaries aligned to judge_ids."""

    judge_ids = np.asarray(posterior["judge_ids"], dtype=str)
    required_fields = ("judge_accuracy_ppc_mean", "judge_accuracy_ppc_p05", "judge_accuracy_ppc_p95")
    missing_fields = [field for field in required_fields if field not in posterior]
    if missing_fields:
        raise ValueError(
            "Posterior archive is unsupported for PPC outputs. Re-run inference to create an archive "
            f"with judge accuracy PPC summaries. missing={missing_fields}"
        )
    for field in required_fields:
        values = np.asarray(posterior[field], dtype=float)
        if values.ndim != 1:
            raise ValueError(f"Posterior {field} must be rank 1, found rank {values.ndim}")
        if len(values) != len(judge_ids):
            raise ValueError(
                f"Posterior {field} length does not match judge_ids length. "
                f"{field}={len(values)} judge_ids={len(judge_ids)}"
            )


def judge_accuracy_ppc_summaries(posterior: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return saved judge-level posterior predictive accuracy summaries."""

    validate_judge_accuracy_ppc_summaries(posterior)
    return (
        np.asarray(posterior["judge_accuracy_ppc_mean"], dtype=float),
        np.asarray(posterior["judge_accuracy_ppc_p05"], dtype=float),
        np.asarray(posterior["judge_accuracy_ppc_p95"], dtype=float),
    )


def ordered_source_ids(matrix: pl.DataFrame, posterior: dict[str, np.ndarray]) -> list[str]:
    """Return source IDs ordered by observed item count, then posterior order."""

    posterior_source_ids = [str(source_id) for source_id in posterior.get("source_ids", [])]
    if not posterior_source_ids:
        return []
    posterior_index = {source_id: index for index, source_id in enumerate(posterior_source_ids)}
    source_counts = (
        matrix.group_by("source")
        .len()
        .rename({"len": "item_count"})
        .sort(["item_count", "source"], descending=[True, False])
    )
    counts_map = {str(row["source"]): int(row["item_count"]) for row in source_counts.to_dicts()}
    return sorted(
        posterior_source_ids,
        key=lambda source_id: (
            -counts_map.get(source_id, 0),
            posterior_index[source_id],
        ),
    )


def top_source_ids(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
    max_sources: int = 8,
) -> list[str]:
    """Return the most data-rich source IDs for small-multiple plotting."""

    return ordered_source_ids(matrix, posterior)[:max_sources]


def source_reliability_summary(
    posterior: dict[str, np.ndarray],
    ordered_sources: list[str],
) -> pl.DataFrame:
    """Return posterior means and intervals for judge reliability by source."""

    if "theta_source" not in posterior or "source_ids" not in posterior:
        raise ValueError("Posterior does not contain source-aware reliability samples.")
    theta_source = posterior["theta_source"]
    flattened = theta_source.reshape(-1, theta_source.shape[-2], theta_source.shape[-1])
    judge_ids = [str(judge_id) for judge_id in posterior["judge_ids"].tolist()]
    source_ids = [str(source_id) for source_id in posterior["source_ids"].tolist()]
    source_index_map = {source_id: index for index, source_id in enumerate(source_ids)}
    rows: list[dict[str, float | str]] = []
    for source_id in ordered_sources:
        source_index = source_index_map[source_id]
        source_samples = flattened[:, :, source_index]
        means = source_samples.mean(axis=0)
        lowers = np.quantile(source_samples, 0.05, axis=0)
        uppers = np.quantile(source_samples, 0.95, axis=0)
        for judge_index, judge_id in enumerate(judge_ids):
            rows.append(
                {
                    "source": source_id,
                    "judge_id": judge_id,
                    "theta_mean": float(means[judge_index]),
                    "theta_p05": float(lowers[judge_index]),
                    "theta_p95": float(uppers[judge_index]),
                }
            )
    return pl.DataFrame(rows)


def validate_posterior_plot_inputs(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> None:
    """Validate matrix and posterior alignment before posterior-backed plotting."""

    matrix_judge_ids = np.asarray(judge_columns(matrix), dtype=str)
    validate_posterior_judge_order(matrix_judge_ids, posterior)
    matrix_item_ids = matrix.get_column("item_id").cast(pl.String).to_numpy().astype(str)
    posterior_item_ids = np.asarray(posterior["item_ids"], dtype=str)
    if not np.array_equal(matrix_item_ids, posterior_item_ids):
        raise ValueError(
            "Posterior item_ids do not match processed matrix item order. "
            f"matrix={matrix_item_ids.tolist()} posterior={posterior_item_ids.tolist()}"
        )
    matrix_source_ids = matrix.get_column("source").cast(pl.String).unique(maintain_order=True).to_numpy().astype(str)
    posterior_source_ids = np.asarray(posterior["source_ids"], dtype=str)
    if not np.array_equal(matrix_source_ids, posterior_source_ids):
        raise ValueError(
            "Posterior source_ids do not match processed matrix source order. "
            f"matrix={matrix_source_ids.tolist()} posterior={posterior_source_ids.tolist()}"
        )
    validate_judge_accuracy_ppc_summaries(posterior)
