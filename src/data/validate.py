"""Validate processed JudgeBench artifacts and summarize judge coverage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from src.data.loader import build_and_write_matrix, load_or_prepare_items
from src.data.matrix_semantics import judge_columns as shared_judge_columns
from src.data.matrix_semantics import summarize_matrix as shared_summarize_matrix
from src.logging_utils import configure_logging, format_table_for_log
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


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
    if items.get_column("item_id").is_duplicated().any():
        raise ValueError("Sampled JudgeBench items contain duplicate item IDs.")
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


def main() -> None:
    """CLI entrypoint for validation."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    items = load_or_prepare_items(config)
    validate_items(items)
    matrix = build_and_write_matrix(config, items)
    validate_matrix(matrix, [judge.id for judge in config.judges])
    summary = summarize_matrix(matrix)
    logger.info("items_ok")
    try:
        assert_complete_judge_coverage(matrix, [judge.id for judge in config.judges])
    except ValueError as exc:
        logger.warning("inference_ready=false reason=%s", exc)
    else:
        logger.info("inference_ready=true")
    if logger.isEnabledFor(logging.INFO):
        logger.info("validation summary\n%s", format_table_for_log(summary))


if __name__ == "__main__":
    main()
