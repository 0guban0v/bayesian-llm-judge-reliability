"""Validate processed JudgeBench artifacts and summarize judge coverage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from src.data.loader import (
    build_and_write_matrix,
    load_judge_logs,
    load_or_prepare_items,
    select_current_item_logs,
    validate_item_content_hashes,
)
from src.data.matrix_semantics import judge_columns as shared_judge_columns
from src.data.matrix_semantics import summarize_matrix as shared_summarize_matrix
from src.logging_utils import configure_logging, format_table_for_log
from src.schemas import ExperimentConfig, JudgeConfig

logger = logging.getLogger(__name__)
PROMPT_ORDER_COVERAGE_SAMPLE_SIZE = 5


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def judge_columns(matrix: pl.DataFrame) -> list[str]:
    """Return the columns corresponding to judge outputs."""

    return shared_judge_columns(matrix)


def summarize_matrix(matrix: pl.DataFrame) -> pl.DataFrame:
    """Compute response-rate and accuracy summaries for each judge."""

    return shared_summarize_matrix(matrix)


def validate_items(items: pl.DataFrame) -> None:
    """Run core validation checks over the sampled JudgeBench items."""

    if items.height == 0:
        raise ValueError("No JudgeBench items were loaded.")
    validate_item_content_hashes(items)
    if items.get_column("item_key").is_duplicated().any():
        raise ValueError("Sampled JudgeBench items contain duplicate split-qualified item keys.")
    invalid_labels = items.filter(~pl.col("label").is_in(["A>B", "B>A"]))
    if invalid_labels.height > 0:
        raise ValueError("Sampled JudgeBench items contain unsupported labels.")


def validate_matrix(matrix: pl.DataFrame, expected_judges: list[str]) -> None:
    """Validate that the matrix has all configured judge columns."""

    missing_columns = [judge_id for judge_id in expected_judges if judge_id not in matrix.columns]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(f"Missing judge columns in processed matrix: {joined}")


def assert_complete_judge_coverage(matrix: pl.DataFrame, expected_judges: list[str]) -> None:
    """Require every configured judge to have non-null outputs for every sampled item."""

    validate_matrix(matrix, expected_judges)
    summary = summarize_matrix(matrix)
    incomplete = summary.filter(pl.col("responded_items") != matrix.height)
    if incomplete.height == 0:
        return
    details = ", ".join(
        f"{row['judge_id']} ({row['responded_items']}/{matrix.height})"
        for row in incomplete.select(["judge_id", "responded_items"]).to_dicts()
    )
    raise ValueError(
        f"Inference requires complete judge coverage for all configured judges. Incomplete judges: {details}"
    )


def assert_complete_prompt_order_coverage(
    items: pl.DataFrame,
    logs: pl.DataFrame,
    judges: list[JudgeConfig],
) -> None:
    """Require every configured item, judge, order, and repeat task to be logged."""

    expected_rows = [
        {
            "item_key": item["item_key"],
            "item_content_hash": item["item_content_hash"],
            "judge_id": judge.id,
            "prompt_order": prompt_order,
            "repeat_index": repeat_index,
        }
        for item in items.select(["item_key", "item_content_hash"]).iter_rows(named=True)
        for judge in judges
        for prompt_order in judge.prompt_orders
        for repeat_index in range(judge.num_repeats)
    ]
    expected = pl.DataFrame(
        expected_rows,
        schema={
            "item_key": pl.String,
            "item_content_hash": pl.String,
            "judge_id": pl.String,
            "prompt_order": pl.String,
            "repeat_index": pl.Int64,
        },
    )
    if expected.height == 0:
        return

    task_columns = expected.columns
    current_logs = select_current_item_logs(items, logs)
    completed = current_logs.select(task_columns).unique(maintain_order=True)
    missing = expected.join(completed, on=task_columns, how="anti")
    if missing.height == 0:
        return

    expected_counts = expected.group_by(["judge_id", "prompt_order"], maintain_order=True).len(name="expected")
    missing_counts = missing.group_by(["judge_id", "prompt_order"], maintain_order=True).len(name="missing")
    incomplete = expected_counts.join(missing_counts, on=["judge_id", "prompt_order"], how="inner").with_columns(
        (pl.col("expected") - pl.col("missing")).alias("completed")
    )
    details = ", ".join(
        f"{row['judge_id']}/{row['prompt_order']} ({row['completed']}/{row['expected']})"
        for row in incomplete.iter_rows(named=True)
    )
    sample = ", ".join(
        f"item_key={row['item_key']} judge_id={row['judge_id']} prompt_order={row['prompt_order']} "
        f"repeat_index={row['repeat_index']}"
        for row in missing.head(PROMPT_ORDER_COVERAGE_SAMPLE_SIZE).iter_rows(named=True)
    )
    raise ValueError(
        "Inference requires complete prompt-order coverage for all configured judge tasks. "
        f"Incomplete judge orders: {details}. Missing task sample: {sample}"
    )


def main() -> None:
    """CLI entrypoint for validation."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    items = load_or_prepare_items(config)
    validate_items(items)
    logs = load_judge_logs(config.data.logs_dir)
    matrix = build_and_write_matrix(config, items, logs)
    validate_matrix(matrix, [judge.id for judge in config.judges])
    summary = summarize_matrix(matrix)
    logger.info("items_ok")
    try:
        assert_complete_judge_coverage(matrix, [judge.id for judge in config.judges])
        assert_complete_prompt_order_coverage(items, logs, config.judges)
    except ValueError as exc:
        logger.warning("inference_ready=false reason=%s", exc)
    else:
        logger.info("inference_ready=true")
    if logger.isEnabledFor(logging.INFO):
        logger.info("validation summary\n%s", format_table_for_log(summary))


if __name__ == "__main__":
    main()
