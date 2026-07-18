"""Shared semantics for judge-matrix construction and summaries."""

from __future__ import annotations

import logging

import polars as pl

from src.schemas import RepeatPolicy

ITEM_METADATA_COLUMNS = {"item_key", "item_id", "original_id", "split", "source", "question", "label"}
DUPLICATE_SAMPLE_SIZE = 5


def judge_columns(matrix: pl.DataFrame) -> list[str]:
    """Return the columns corresponding to judge outputs."""

    return [column for column in matrix.columns if column not in ITEM_METADATA_COLUMNS]


def summarize_matrix(matrix: pl.DataFrame) -> pl.DataFrame:
    """Compute response-rate and accuracy summaries for each judge."""

    total = matrix.height
    summaries: list[dict[str, float | int | str]] = []
    for judge_id in judge_columns(matrix):
        series = matrix.get_column(judge_id)
        responded = int(series.is_not_null().sum())
        accuracy = float(series.drop_nulls().mean() or 0.0)
        summaries.append(
            {
                "judge_id": judge_id,
                "responded_items": responded,
                "response_rate": responded / total if total else 0.0,
                "accuracy": accuracy,
            }
        )
    return pl.DataFrame(summaries).sort("judge_id")


def observed_accuracy_frame(matrix: pl.DataFrame, judge_ids: list[str] | None = None) -> pl.DataFrame:
    """Return ordered observed accuracies for downstream joins and plots."""

    summary = summarize_matrix(matrix)
    if judge_ids is None:
        return summary.select(["judge_id", "accuracy"])
    order_lookup = {judge_id: index for index, judge_id in enumerate(judge_ids)}
    filtered = summary.filter(pl.col("judge_id").is_in(judge_ids)).with_columns(
        pl.col("judge_id").replace_strict(order_lookup, return_dtype=pl.UInt16).alias("judge_order")
    )
    return filtered.sort("judge_order").select(["judge_id", "accuracy"])


def resolve_original_judgments(
    logs: pl.DataFrame,
    *,
    repeat_policy: RepeatPolicy = "reject",
    duplicate_logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Resolve repeated tasks, then return scored original-order judgments."""

    if repeat_policy not in {"reject", "first", "latest"}:
        raise ValueError(f"Unsupported repeat policy: {repeat_policy}")

    if logs.height == 0:
        return pl.DataFrame(
            schema={
                "item_key": pl.String,
                "item_id": pl.String,
                "judge_id": pl.String,
                "correct_int": pl.Int8,
            }
        )
    ordered_logs = logs.with_row_index("log_order")
    duplicate_keys = ["item_key", "judge_id", "prompt_order"]
    duplicate_judgments = ordered_logs.group_by(duplicate_keys, maintain_order=True).len().filter(pl.col("len") > 1)
    if duplicate_judgments.height > 0:
        if repeat_policy == "reject":
            duplicate = duplicate_judgments.sort(duplicate_keys).row(0, named=True)
            raise ValueError(
                f"Duplicate judgment: item_key={duplicate['item_key']} judge_id={duplicate['judge_id']} "
                f"prompt_order={duplicate['prompt_order']} count={duplicate['len']}"
            )
        if duplicate_logger is not None:
            duplicate_logger.warning(
                "duplicate judgments resolved repeat_policy=%s group_count=%s group_sample=%s",
                repeat_policy,
                duplicate_judgments.height,
                duplicate_judgments.head(DUPLICATE_SAMPLE_SIZE).to_dicts(),
            )

    keep = "first" if repeat_policy == "first" else "last"
    return (
        ordered_logs.unique(subset=duplicate_keys, keep=keep, maintain_order=True)
        .filter(pl.col("prompt_order").eq("original") & pl.col("correct").is_not_null())
        .with_columns(pl.col("correct").cast(pl.Int8).alias("correct_int"))
        .select(["log_order", "item_key", "judge_id", "correct_int"])
    )


def pivot_original_judgments(resolved_judgments: pl.DataFrame) -> pl.DataFrame:
    """Pivot resolved scored original-order judgments into a wide judge matrix."""

    if resolved_judgments.height == 0:
        return pl.DataFrame(schema={"item_key": pl.String})
    return resolved_judgments.pivot(
        index="item_key",
        on="judge_id",
        values="correct_int",
        aggregate_function="first",
    )
