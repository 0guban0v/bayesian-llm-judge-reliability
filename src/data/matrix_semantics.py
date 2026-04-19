"""Shared semantics for judge-matrix construction and summaries."""

from __future__ import annotations

import logging

import polars as pl

ITEM_METADATA_COLUMNS = {"item_key", "item_id", "original_id", "split", "source", "question", "label"}


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


def first_original_judgments(
    logs: pl.DataFrame,
    *,
    duplicate_logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Return first scored original-order judgments per item/judge pair."""

    if logs.height == 0:
        return pl.DataFrame(
            schema={
                "item_key": pl.String,
                "item_id": pl.String,
                "judge_id": pl.String,
                "correct_int": pl.Int8,
            }
        )
    original_logs = (
        logs.filter(pl.col("prompt_order").eq("original") & pl.col("correct").is_not_null())
        .with_row_index("log_order")
        .sort("log_order")
        .with_columns(pl.col("correct").cast(pl.Int8).alias("correct_int"))
    )
    if original_logs.height == 0:
        return pl.DataFrame(
            schema={
                "item_key": pl.String,
                "item_id": pl.String,
                "judge_id": pl.String,
                "correct_int": pl.Int8,
            }
        )
    duplicate_judgments = original_logs.group_by(["item_key", "judge_id"]).len().filter(pl.col("len") > 1)
    if duplicate_judgments.height > 0 and duplicate_logger is not None:
        duplicate_logger.warning(
            "duplicate original-order judgments detected; keeping first result per item/judge pair duplicates=%s",
            duplicate_judgments.select(["item_key", "judge_id", "len"]).to_dicts(),
        )
    return (
        original_logs.select(["log_order", "item_key", "judge_id", "correct_int"])
        .group_by(["item_key", "judge_id"], maintain_order=True)
        .agg(pl.col("correct_int").first().alias("correct_int"))
    )


def pivot_original_judgments(first_judgments: pl.DataFrame) -> pl.DataFrame:
    """Pivot first scored original-order judgments into a wide judge matrix."""

    if first_judgments.height == 0:
        return pl.DataFrame(schema={"item_key": pl.String})
    return first_judgments.pivot(
        index="item_key",
        on="judge_id",
        values="correct_int",
        aggregate_function="first",
    )
