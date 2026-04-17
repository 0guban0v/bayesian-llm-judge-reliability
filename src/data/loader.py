"""Load JudgeBench, prepare item subsets, and build judge matrices."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import polars as pl
from datasets import load_dataset

from src.data.matrix_semantics import (
    ITEM_METADATA_COLUMNS,
    first_original_judgments,
    pivot_original_judgments,
)
from src.logging_utils import configure_logging
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)

ITEM_COLUMNS = [
    "item_id",
    "original_id",
    "split",
    "source",
    "question",
    "response_model",
    "response_a",
    "response_b",
    "label",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    parser.add_argument(
        "--refresh-items",
        action="store_true",
        help="Force a fresh sample from JudgeBench instead of reusing cached parquet.",
    )
    parser.add_argument(
        "--rebuild-matrix",
        action="store_true",
        help="Recompute the processed judge matrix from JSONL logs.",
    )
    return parser.parse_args()


def _compile_category_token_sequences(categories: list[str]) -> list[list[str]]:
    """Normalize configured categories into comparable token sequences once."""

    return [_category_tokens(category) for category in categories]


def _matches_category_token_sequences(
    source_tokens: list[str],
    category_token_sequences: list[list[str]],
) -> bool:
    """Return whether source tokens contain any precompiled category token sequence."""

    if not category_token_sequences:
        return True
    return any(_contains_token_sequence(source_tokens, category_tokens) for category_tokens in category_token_sequences)


def _matches_categories(source: str, categories: list[str]) -> bool:
    """Return whether a JudgeBench source matches any configured category token sequence."""

    return _matches_category_token_sequences(
        _category_tokens(source),
        _compile_category_token_sequences(categories),
    )


def _category_tokens(value: str) -> list[str]:
    """Normalize a source/category string into comparable lowercase word tokens."""

    return [token for token in re.split(r"[^a-z0-9]+", value.casefold()) if token]


def _contains_token_sequence(source_tokens: list[str], category_tokens: list[str]) -> bool:
    """Return whether category tokens appear contiguously inside source tokens."""

    if not category_tokens:
        return False
    window = len(category_tokens)
    for start in range(len(source_tokens) - window + 1):
        if source_tokens[start : start + window] == category_tokens:
            return True
    return False


def _dataset_to_frame(dataset_name: str, split_name: str) -> pl.DataFrame:
    """Load one JudgeBench split into a normalized Polars DataFrame."""

    dataset = load_dataset(dataset_name, split=split_name)
    frame = pl.from_arrow(dataset.data.table)
    return (
        frame.rename(
            {
                "pair_id": "item_id",
                "response_A": "response_a",
                "response_B": "response_b",
            }
        )
        .with_columns(pl.lit(split_name).alias("split"))
        .select(ITEM_COLUMNS)
    )


def load_judgebench_frame(config: ExperimentConfig) -> pl.DataFrame:
    """Load and filter JudgeBench according to config."""

    frames = [_dataset_to_frame(config.data.hf_dataset, split_name) for split_name in config.data.splits]
    combined = pl.concat(frames, how="vertical")
    category_token_sequences = _compile_category_token_sequences(config.data.categories)
    if not config.data.categories:
        filtered = combined
    else:
        filtered = combined.filter(
            pl.col("source").map_elements(
                lambda source: _matches_category_token_sequences(
                    _category_tokens(str(source)),
                    category_token_sequences,
                ),
                return_dtype=pl.Boolean,
            )
        )
    subset_size = min(config.data.subset_size, filtered.height)
    return filtered.sample(n=subset_size, seed=config.experiment.seed, shuffle=True)


def write_frame(frame: pl.DataFrame, path: Path) -> None:
    """Persist a Polars DataFrame to parquet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def load_or_prepare_items(config: ExperimentConfig, refresh: bool = False) -> pl.DataFrame:
    """Load cached item parquet or build a fresh JudgeBench subset."""

    config.ensure_directories()
    if config.data.item_path.exists() and not refresh:
        return pl.read_parquet(config.data.item_path)
    items = load_judgebench_frame(config)
    write_frame(items, config.data.item_path)
    raw_snapshot_path = config.data.raw_dir / "judgebench_subset.jsonl"
    with raw_snapshot_path.open("w", encoding="utf-8") as handle:
        for row in items.to_dicts():
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return items


def load_judge_logs(logs_dir: Path) -> pl.DataFrame:
    """Load append-only JSONL judge logs into Polars."""

    rows: list[dict[str, object]] = []
    for log_path in sorted(logs_dir.glob("*.jsonl")):
        with log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        return pl.DataFrame(
            schema={
                "item_id": pl.String,
                "judge_id": pl.String,
                "prompt_order": pl.String,
                "correct": pl.Boolean,
            }
        )
    return pl.DataFrame(rows)


def build_binary_matrix(
    items: pl.DataFrame,
    logs: pl.DataFrame,
    judge_ids: list[str],
) -> pl.DataFrame:
    """Build an item-by-judge correctness matrix from original-order logs."""

    first_judgments = first_original_judgments(logs, duplicate_logger=logger)
    if first_judgments.height == 0:
        matrix = items.select(sorted(ITEM_METADATA_COLUMNS))
    else:
        pivoted = pivot_original_judgments(first_judgments)
        matrix = items.select(sorted(ITEM_METADATA_COLUMNS)).join(pivoted, on="item_id", how="left")

    for judge_id in judge_ids:
        if judge_id not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(None, dtype=pl.Int8).alias(judge_id))
    ordered_columns = sorted(ITEM_METADATA_COLUMNS) + judge_ids
    return matrix.select(ordered_columns)


def build_and_write_matrix(
    config: ExperimentConfig,
    items: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build and persist the judge matrix parquet."""

    prepared_items = items if items is not None else load_or_prepare_items(config)
    logs = load_judge_logs(config.data.logs_dir)
    matrix = build_binary_matrix(prepared_items, logs, [judge.id for judge in config.judges])
    write_frame(matrix, config.data.matrix_path)
    return matrix


def main() -> None:
    """CLI entrypoint for JudgeBench loading and matrix preparation."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    items = load_or_prepare_items(config, refresh=args.refresh_items)
    logger.info("wrote item subset to %s (%s rows)", config.data.item_path, items.height)
    if args.rebuild_matrix or any(config.data.logs_dir.glob("*.jsonl")):
        matrix = build_and_write_matrix(config, items)
        logger.info("wrote judge matrix to %s (%s rows)", config.data.matrix_path, matrix.height)


if __name__ == "__main__":
    main()
