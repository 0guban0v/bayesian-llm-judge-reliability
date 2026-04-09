"""Validate processed JudgeBench artifacts and summarize judge coverage."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from src.data.loader import ITEM_METADATA_COLUMNS, build_and_write_matrix, load_or_prepare_items
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

    return [column for column in matrix.columns if column not in ITEM_METADATA_COLUMNS]


def summarize_matrix(matrix: pl.DataFrame) -> pl.DataFrame:
    """Compute response-rate and accuracy summaries for each judge."""

    summaries: list[dict[str, float | int | str]] = []
    for judge_id in judge_columns(matrix):
        series = matrix.get_column(judge_id)
        responded = int(series.is_not_null().sum())
        total = matrix.height
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
    logger.info("validation summary\n%s", format_table_for_log(summary))


if __name__ == "__main__":
    main()
