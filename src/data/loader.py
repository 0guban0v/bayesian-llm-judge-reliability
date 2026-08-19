"""Load JudgeBench, prepare item subsets, and build judge matrices."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import polars as pl
from datasets import load_dataset
from pydantic import ValidationError

from src.data.item_identity import ITEM_CONTENT_FIELDS, item_content_hash, validate_item_content_hash
from src.data.matrix_semantics import (
    ANALYSIS_COLUMNS,
    ANALYSIS_SCHEMA_VERSION,
    ITEM_METADATA_COLUMNS,
    empty_analysis_table,
    pivot_original_judgments,
    resolve_judgments,
    resolve_original_judgments,
    validate_analysis_table,
)
from src.logging_utils import configure_logging
from src.schemas import ExperimentConfig, JudgeResult, RepeatPolicy

logger = logging.getLogger(__name__)

PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 19
STALE_ITEM_KEY_SAMPLE_SIZE = 5

ITEM_COLUMNS = [
    "item_key",
    "item_content_hash",
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
        .with_columns(
            [
                pl.lit(split_name).alias("split"),
                pl.format("{}:{}", pl.lit(split_name), pl.col("item_id").cast(pl.String)).alias("item_key"),
            ]
        )
        .with_columns(
            pl.struct(ITEM_CONTENT_FIELDS)
            .map_elements(item_content_hash, return_dtype=pl.String)
            .alias("item_content_hash")
        )
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
    frame.write_parquet(
        path,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


def validate_item_content_hashes(items: pl.DataFrame) -> None:
    """Require every prepared item to match its stored content hash."""

    required_columns = ["item_key", "item_content_hash", *ITEM_CONTENT_FIELDS]
    missing_columns = [column for column in required_columns if column not in items.columns]
    if missing_columns:
        raise ValueError(f"JudgeBench items are missing content-hash columns: {', '.join(missing_columns)}")
    for item in items.select(required_columns).iter_rows(named=True):
        validate_item_content_hash(item)


def validate_cached_items(items: pl.DataFrame, item_path: Path) -> None:
    """Require cached item parquets to contain the current content-addressed schema."""

    if "item_key" not in items.columns:
        raise ValueError(
            f"Cached JudgeBench items at {item_path} are unsupported because they predate split-qualified item keys. "
            "Re-run with --refresh-items to rebuild the cached item subset."
        )
    if "item_content_hash" not in items.columns:
        raise ValueError(
            f"Cached JudgeBench items at {item_path} are unsupported because they predate item content hashes. "
            "Re-run with --refresh-items to rebuild the cached item subset."
        )
    validate_item_content_hashes(items)


def load_or_prepare_items(config: ExperimentConfig, refresh: bool = False) -> pl.DataFrame:
    """Load cached item parquet or build a fresh JudgeBench subset."""

    config.ensure_directories()
    if config.data.item_path.exists() and not refresh:
        items = pl.read_parquet(config.data.item_path)
        validate_cached_items(items, config.data.item_path)
        return items
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
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Judge log {log_path} is malformed at line {line_number}: invalid JSON: {exc.msg}."
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Judge log {log_path} is malformed at line {line_number}: expected JSON object.")
                if "item_key" not in record:
                    raise ValueError(
                        f"Judge log {log_path} is unsupported at line {line_number} because it predates "
                        "split-qualified item keys. Delete it and re-run judges with the current pipeline."
                    )
                if "item_content_hash" not in record:
                    raise ValueError(
                        f"Judge log {log_path} is unsupported at line {line_number} because it predates item "
                        "content hashes. Delete it and re-run judges with the current item content."
                    )
                if "repeat_index" not in record:
                    raise ValueError(
                        f"Judge log {log_path} is unsupported at line {line_number} because it predates explicit "
                        "repeat indices. Delete it and re-run judges with the current pipeline."
                    )
                try:
                    parsed_record = JudgeResult.from_persisted_record(record)
                except (ValidationError, ValueError) as exc:
                    if not isinstance(exc, ValidationError):
                        raise ValueError(f"Judge log {log_path} is malformed at line {line_number}: {exc}.") from exc
                    first_error = exc.errors()[0]
                    field = ".".join(str(part) for part in first_error["loc"])
                    raise ValueError(
                        f"Judge log {log_path} is malformed at line {line_number}: {field}: {first_error['msg']}."
                    ) from exc
                rows.append(parsed_record.to_json_dict())
    if not rows:
        return pl.DataFrame(
            schema={
                "item_key": pl.String,
                "item_content_hash": pl.String,
                "item_id": pl.String,
                "judge_id": pl.String,
                "prompt_order": pl.String,
                "repeat_index": pl.Int64,
                "correct": pl.Boolean,
            }
        )
    return pl.DataFrame(rows)


def select_current_item_logs(items: pl.DataFrame, logs: pl.DataFrame) -> pl.DataFrame:
    """Keep log rows whose content hash matches current item content."""

    required_column = "item_content_hash"
    if required_column not in items.columns:
        raise ValueError(f"JudgeBench items are missing content-hash columns: {required_column}")
    if required_column not in logs.columns:
        raise ValueError("Judge logs are missing content-hash column: item_content_hash")

    current_items = items.select(["item_key", "item_content_hash"])
    logs_for_current_keys = logs.join(current_items.select("item_key"), on="item_key", how="semi")
    stale_logs = logs_for_current_keys.join(
        current_items,
        on=["item_key", "item_content_hash"],
        how="anti",
    )
    if stale_logs.height > 0:
        stale_item_keys = stale_logs.get_column("item_key").unique(maintain_order=True)
        logger.warning(
            "stale judgments excluded because item content changed rows=%s item_key_count=%s item_key_sample=%s",
            stale_logs.height,
            len(stale_item_keys),
            stale_item_keys.head(STALE_ITEM_KEY_SAMPLE_SIZE).to_list(),
        )
    return logs.join(
        current_items,
        on=["item_key", "item_content_hash"],
        how="semi",
    )


def build_binary_matrix(
    items: pl.DataFrame,
    logs: pl.DataFrame,
    judge_ids: list[str],
    repeat_policy: RepeatPolicy = "reject",
) -> pl.DataFrame:
    """Build an item-by-judge correctness matrix from original-order logs."""

    current_logs = select_current_item_logs(items, logs)
    resolved_judgments = resolve_original_judgments(
        current_logs,
        repeat_policy=repeat_policy,
        duplicate_logger=logger,
    )
    if resolved_judgments.height == 0:
        matrix = items.select(sorted(ITEM_METADATA_COLUMNS))
    else:
        pivoted = pivot_original_judgments(resolved_judgments)
        matrix = items.select(sorted(ITEM_METADATA_COLUMNS)).join(pivoted, on="item_key", how="left")

    for judge_id in judge_ids:
        if judge_id not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(None, dtype=pl.Int8).alias(judge_id))
    ordered_columns = sorted(ITEM_METADATA_COLUMNS) + judge_ids
    return matrix.select(ordered_columns)


def build_analysis_table(
    items: pl.DataFrame,
    logs: pl.DataFrame,
    repeat_policy: RepeatPolicy = "reject",
) -> pl.DataFrame:
    """Build the canonical order-level judgment table from current logs."""

    current_logs = select_current_item_logs(items, logs)
    if current_logs.height == 0:
        return empty_analysis_table()

    required_log_columns = {
        "item_key",
        "item_content_hash",
        "judge_id",
        "prompt_order",
        "repeat_index",
        "parsed_verdict",
        "correct",
    }
    missing_log_columns = sorted(required_log_columns - set(current_logs.columns))
    if missing_log_columns:
        raise ValueError(f"Judge logs are missing analysis columns: {', '.join(missing_log_columns)}")

    required_item_columns = {"item_key", "item_content_hash", "item_id", "source", "split", "label"}
    missing_item_columns = sorted(required_item_columns - set(items.columns))
    if missing_item_columns:
        raise ValueError(f"JudgeBench items are missing analysis columns: {', '.join(missing_item_columns)}")

    resolved = resolve_judgments(
        current_logs,
        repeat_policy=repeat_policy,
        duplicate_logger=logger,
    )
    joined = resolved.select(
        [
            "log_order",
            "item_key",
            "item_content_hash",
            "judge_id",
            "prompt_order",
            "repeat_index",
            "parsed_verdict",
            pl.col("correct").alias("logged_correct"),
        ]
    ).join(
        items.select(["item_key", "item_content_hash", "item_id", "source", "split", "label"]),
        on=["item_key", "item_content_hash"],
        how="inner",
        validate="m:1",
    )
    analysis = (
        joined.with_columns(
            [
                pl.lit(ANALYSIS_SCHEMA_VERSION, dtype=pl.UInt16).alias("analysis_schema_version"),
                pl.col("parsed_verdict").cast(pl.String).alias("normalized_choice"),
                pl.when(pl.col("label").eq("A>B")).then(pl.lit("A")).otherwise(pl.lit("B")).alias("gold_choice"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("prompt_order").eq("reversed"))
                .then(pl.col("normalized_choice").replace_strict({"A": "B", "B": "A"}, default=None))
                .otherwise(pl.col("normalized_choice"))
                .alias("displayed_choice"),
                pl.col("normalized_choice").eq(pl.col("gold_choice")).alias("correct"),
                pl.col("normalized_choice").is_not_null().alias("valid"),
            ]
        )
        .sort("log_order")
    )
    inconsistent_logged_correct = analysis.filter(~pl.col("logged_correct").eq_missing(pl.col("correct")))
    if inconsistent_logged_correct.height:
        row = inconsistent_logged_correct.row(0, named=True)
        raise ValueError(
            "Judge log correctness is inconsistent with its normalized choice for "
            f"item_key={row['item_key']} judge_id={row['judge_id']} prompt_order={row['prompt_order']}."
        )

    analysis = analysis.select(ANALYSIS_COLUMNS)
    validate_analysis_table(analysis)
    return analysis


def build_binary_matrix_from_analysis(
    items: pl.DataFrame,
    analysis: pl.DataFrame,
    judge_ids: list[str],
) -> pl.DataFrame:
    """Build the original-order wide compatibility export from the analysis table."""

    if analysis.height == 0:
        matrix = items.select(sorted(ITEM_METADATA_COLUMNS))
    else:
        resolved_original = (
            analysis.filter(pl.col("prompt_order").eq("original") & pl.col("correct").is_not_null())
            .with_columns(pl.col("correct").cast(pl.Int8).alias("correct_int"))
            .select(["item_key", "judge_id", "correct_int"])
        )
        pivoted = pivot_original_judgments(resolved_original)
        matrix = items.select(sorted(ITEM_METADATA_COLUMNS)).join(pivoted, on="item_key", how="left")

    for judge_id in judge_ids:
        if judge_id not in matrix.columns:
            matrix = matrix.with_columns(pl.lit(None, dtype=pl.Int8).alias(judge_id))
    return matrix.select(sorted(ITEM_METADATA_COLUMNS) + judge_ids)


def build_and_write_analysis_artifacts(
    config: ExperimentConfig,
    items: pl.DataFrame | None = None,
    logs: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build and persist the canonical analysis table and wide compatibility export."""

    prepared_items = items if items is not None else load_or_prepare_items(config)
    prepared_logs = logs if logs is not None else load_judge_logs(config.data.logs_dir)
    analysis = build_analysis_table(
        prepared_items,
        prepared_logs,
        repeat_policy=config.data.repeat_policy,
    )
    matrix = build_binary_matrix_from_analysis(
        prepared_items,
        analysis,
        [judge.id for judge in config.judges],
    )
    write_frame(analysis, config.data.analysis_path)
    write_frame(matrix, config.data.matrix_path)
    return analysis, matrix


def build_and_write_matrix(
    config: ExperimentConfig,
    items: pl.DataFrame | None = None,
    logs: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build both analysis artifacts and return the wide compatibility export."""

    _, matrix = build_and_write_analysis_artifacts(config, items, logs)
    return matrix


def main() -> None:
    """CLI entrypoint for JudgeBench loading and matrix preparation."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    items = load_or_prepare_items(config, refresh=args.refresh_items)
    logger.info("wrote item subset to %s (%s rows)", config.data.item_path, items.height)
    if args.rebuild_matrix or any(config.data.logs_dir.glob("*.jsonl")):
        analysis, matrix = build_and_write_analysis_artifacts(config, items)
        logger.info("wrote analysis table to %s (%s rows)", config.data.analysis_path, analysis.height)
        logger.info("wrote judge matrix to %s (%s rows)", config.data.matrix_path, matrix.height)


if __name__ == "__main__":
    main()
