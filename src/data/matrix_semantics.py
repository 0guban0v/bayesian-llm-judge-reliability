"""Shared semantics for judge-matrix construction and summaries."""

from __future__ import annotations

import logging

import polars as pl

from src.data.item_identity import ITEM_CONTENT_HASH_PATTERN
from src.schemas import RepeatPolicy

ITEM_METADATA_COLUMNS = {"item_key", "item_id", "original_id", "split", "source", "question", "label"}
DUPLICATE_SAMPLE_SIZE = 5
ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_COLUMNS = [
    "analysis_schema_version",
    "item_key",
    "item_content_hash",
    "item_id",
    "source",
    "split",
    "judge_id",
    "prompt_order",
    "repeat_index",
    "displayed_choice",
    "normalized_choice",
    "gold_choice",
    "correct",
    "valid",
]
ANALYSIS_SCHEMA = {
    "analysis_schema_version": pl.UInt16,
    "item_key": pl.String,
    "item_content_hash": pl.String,
    "item_id": pl.String,
    "source": pl.String,
    "split": pl.String,
    "judge_id": pl.String,
    "prompt_order": pl.String,
    "repeat_index": pl.Int64,
    "displayed_choice": pl.String,
    "normalized_choice": pl.String,
    "gold_choice": pl.String,
    "correct": pl.Boolean,
    "valid": pl.Boolean,
}


def empty_analysis_table() -> pl.DataFrame:
    """Return an empty frame matching the persisted analysis-table schema."""

    return pl.DataFrame(schema=ANALYSIS_SCHEMA)


def empty_resolved_judgments() -> pl.DataFrame:
    """Return empty frame matching resolved-judgment output schema."""

    return pl.DataFrame(
        schema={
            "log_order": pl.UInt32,
            "item_key": pl.String,
            "judge_id": pl.String,
            "correct_int": pl.Int8,
        }
    )


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

    if logs.height == 0:
        if repeat_policy not in {"reject", "first", "latest"}:
            raise ValueError(f"Unsupported repeat policy: {repeat_policy}")
        return empty_resolved_judgments()
    return (
        resolve_judgments(logs, repeat_policy=repeat_policy, duplicate_logger=duplicate_logger)
        .filter(pl.col("prompt_order").eq("original") & pl.col("correct").is_not_null())
        .with_columns(pl.col("correct").cast(pl.Int8).alias("correct_int"))
        .select(["log_order", "item_key", "judge_id", "correct_int"])
    )


def resolve_judgments(
    logs: pl.DataFrame,
    *,
    repeat_policy: RepeatPolicy = "reject",
    duplicate_logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Resolve repeated tasks while retaining judgments from every prompt order."""

    if repeat_policy not in {"reject", "first", "latest"}:
        raise ValueError(f"Unsupported repeat policy: {repeat_policy}")
    if logs.height == 0:
        return logs.with_row_index("log_order")

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
    return ordered_logs.unique(subset=duplicate_keys, keep=keep, maintain_order=True)


def validate_analysis_table(analysis: pl.DataFrame) -> None:
    """Require the persisted long-form artifact to satisfy its versioned choice contract."""

    missing_columns = [column for column in ANALYSIS_COLUMNS if column not in analysis.columns]
    if missing_columns:
        raise ValueError(f"Analysis table is missing required columns: {', '.join(missing_columns)}")
    if analysis.height == 0:
        raise ValueError("Analysis table contains no judgments.")
    invalid_types = [
        f"{column}={analysis.schema[column]} (expected {expected_type})"
        for column, expected_type in ANALYSIS_SCHEMA.items()
        if analysis.schema[column] != expected_type
    ]
    if invalid_types:
        raise ValueError(f"Analysis table has incompatible column types: {', '.join(invalid_types)}")

    non_null_columns = [
        "analysis_schema_version",
        "item_key",
        "item_content_hash",
        "item_id",
        "source",
        "split",
        "judge_id",
        "prompt_order",
        "repeat_index",
        "gold_choice",
        "valid",
    ]
    if analysis.select(pl.any_horizontal(pl.col(non_null_columns).is_null()).any()).item():
        raise ValueError("Analysis table contains null values in required identity or task columns.")

    versions = analysis.get_column("analysis_schema_version").unique().to_list()
    if versions != [ANALYSIS_SCHEMA_VERSION]:
        raise ValueError(
            f"Unsupported analysis schema version(s): {versions}; expected {ANALYSIS_SCHEMA_VERSION}. "
            "Rebuild the analysis artifacts from current judge logs."
        )
    duplicate_keys = ["item_key", "judge_id", "prompt_order"]
    if analysis.select(duplicate_keys).is_duplicated().any():
        raise ValueError("Analysis table contains duplicate item, judge, and prompt-order rows.")
    if analysis.filter(~pl.col("prompt_order").is_in(["original", "reversed"])).height:
        raise ValueError("Analysis table contains unsupported prompt orders.")
    if analysis.filter(~pl.col("gold_choice").is_in(["A", "B"])).height:
        raise ValueError("Analysis table contains unsupported gold choices.")
    if analysis.filter(~pl.col("item_content_hash").str.contains(ITEM_CONTENT_HASH_PATTERN)).height:
        raise ValueError("Analysis table contains malformed item content hashes.")
    if analysis.filter(pl.col("repeat_index") < 0).height:
        raise ValueError("Analysis table contains negative repeat indices.")
    item_identity_columns = ["item_key", "item_content_hash", "item_id", "source", "split", "gold_choice"]
    if analysis.select(item_identity_columns).unique().get_column("item_key").is_duplicated().any():
        raise ValueError("Analysis table contains inconsistent item identity or gold choice metadata.")
    for column in ("displayed_choice", "normalized_choice"):
        if analysis.filter(pl.col(column).is_not_null() & ~pl.col(column).is_in(["A", "B"])).height:
            raise ValueError(f"Analysis table contains unsupported {column} values.")

    expected_displayed = (
        pl.when(pl.col("prompt_order").eq("reversed"))
        .then(pl.col("normalized_choice").replace_strict({"A": "B", "B": "A"}, default=None))
        .otherwise(pl.col("normalized_choice"))
    )
    expected_correct = pl.col("normalized_choice").eq(pl.col("gold_choice"))
    inconsistent = analysis.filter(
        (pl.col("valid") != pl.col("normalized_choice").is_not_null())
        | (pl.col("displayed_choice") != expected_displayed).fill_null(False)
        | (pl.col("displayed_choice").is_null() != pl.col("normalized_choice").is_null())
        | (pl.col("correct") != expected_correct).fill_null(False)
        | (pl.col("correct").is_null() != pl.col("normalized_choice").is_null())
    )
    if inconsistent.height:
        row = inconsistent.select(duplicate_keys).row(0, named=True)
        raise ValueError(
            "Analysis table contains inconsistent choice semantics for "
            f"item_key={row['item_key']} judge_id={row['judge_id']} prompt_order={row['prompt_order']}."
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
