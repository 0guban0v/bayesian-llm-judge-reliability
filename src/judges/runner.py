"""Run local MLX judges over JudgeBench items and log JSONL records."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, TextIO, cast

from src.data.loader import load_or_prepare_items
from src.judges.mlx_backend import clear_model_cache, generate_text
from src.judges.parsers import parse_correctness, parse_verdict, swap_verdict
from src.judges.prompts import format_prompt
from src.logging_utils import configure_logging
from src.schemas import ExperimentConfig, JudgeConfig, JudgeResult

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the judge runner."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    parser.add_argument("--judge", type=str, default=None, help="Optional judge ID to run.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional item limit for smoke runs.",
    )
    return parser.parse_args()


def build_log_path(logs_dir: Path, judge_id: str) -> Path:
    """Return the append-only JSONL path for a judge."""

    return logs_dir / f"{judge_id}.jsonl"


def load_processed_keys(log_path: Path) -> set[tuple[str, str]]:
    """Return previously logged `(item_id, prompt_order)` pairs for resumable runs."""

    if not log_path.exists():
        return set()
    processed: set[tuple[str, str]] = set()
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            processed.add((record["item_id"], record["prompt_order"]))
    return processed


def select_judges(config: ExperimentConfig, judge_id: str | None) -> list[JudgeConfig]:
    """Select all configured judges or a single requested judge."""

    if judge_id is None:
        return config.judges
    filtered = [judge for judge in config.judges if judge.id == judge_id]
    if not filtered:
        raise ValueError(f"Judge '{judge_id}' not found in config.")
    return filtered


def prompt_payload(item: dict[str, Any], prompt_order: str) -> tuple[str, str, str]:
    """Return question and ordered responses for a prompt invocation."""

    if prompt_order == "reversed":
        return item["question"], item["response_b"], item["response_a"]
    return item["question"], item["response_a"], item["response_b"]


def judge_item(
    judge: JudgeConfig,
    item: dict[str, Any],
    prompt_order: str,
) -> JudgeResult:
    """Run one MLX judge on one item and normalize the verdict back to original order."""

    question, response_a, response_b = prompt_payload(item, prompt_order)
    prompt = format_prompt(
        judge.prompt_template,
        question=question,
        response_a=response_a,
        response_b=response_b,
    )
    start = time.perf_counter()
    raw_response = generate_text(
        model_name=judge.model,
        prompt=prompt,
        max_tokens=judge.max_tokens,
        trust_remote_code=judge.trust_remote_code,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    verdict = parse_verdict(raw_response)
    if prompt_order == "reversed":
        verdict = swap_verdict(verdict)
    correct = parse_correctness(verdict, item["label"])
    parsed_verdict = verdict if verdict != "UNKNOWN" else None
    assert prompt_order in {"original", "reversed"}
    parsed_verdict_literal = (
        cast(Literal["A", "B"], parsed_verdict) if parsed_verdict is not None else None
    )
    return JudgeResult(
        item_id=item["item_id"],
        judge_id=judge.id,
        timestamp=datetime.now(UTC),
        source=item["source"],
        question=item["question"],
        ground_truth_label=item["label"],
        prompt_variant=judge.prompt_template,
        prompt_order=cast(Literal["original", "reversed"], prompt_order),
        raw_response=raw_response,
        parsed_verdict=parsed_verdict_literal,
        correct=correct,
        latency_ms=latency_ms,
    )


def write_result(handle: TextIO, result: JudgeResult) -> None:
    """Write one judge result to an already-open JSONL handle."""

    handle.write(json.dumps(result.to_json_dict(), ensure_ascii=True) + "\n")
    handle.flush()


def run_judge(
    config: ExperimentConfig,
    judge: JudgeConfig,
    items: list[dict[str, Any]],
) -> int:
    """Evaluate a configured judge over all pending items."""

    log_path = build_log_path(config.data.logs_dir, judge.id)
    processed = load_processed_keys(log_path)
    prompt_orders = ("original", "reversed") if judge.reverse_order else ("original",)
    tasks: list[tuple[dict[str, Any], str]] = []
    for item in items:
        for prompt_order in prompt_orders:
            key = (item["item_id"], prompt_order)
            if key not in processed:
                tasks.append((item, prompt_order))
    logger.info(
        "judge=%s model=%s prompt_variant=%s pending_tasks=%s log_path=%s",
        judge.id,
        judge.model,
        judge.prompt_template,
        len(tasks),
        log_path,
    )
    if not tasks:
        logger.info("judge=%s nothing to do", judge.id)
        return 0

    completed = 0
    with log_path.open("a", encoding="utf-8") as handle:
        for item_index, (item, prompt_order) in enumerate(tasks, start=1):
            logger.info(
                "judge=%s start item=%s order=%s index=%s/%s",
                judge.id,
                item["item_id"],
                prompt_order,
                item_index,
                len(tasks),
            )
            result = judge_item(judge, item, prompt_order)
            write_result(handle, result)
            completed += 1
            logger.info(
                "judge=%s done item=%s order=%s verdict=%s correct=%s latency_ms=%s",
                judge.id,
                result.item_id,
                result.prompt_order,
                result.parsed_verdict,
                result.correct,
                result.latency_ms,
            )
    logger.info("judge=%s wrote_results=%s log_path=%s", judge.id, completed, log_path)
    return completed


def run_all(config: ExperimentConfig, judge_id: str | None, limit: int | None) -> None:
    """Run the judge evaluation pipeline."""

    config.ensure_directories()
    items = load_or_prepare_items(config)
    if limit is not None:
        items = items.head(limit)
    materialized_items = items.to_dicts()
    logger.info(
        "loaded_items=%s judge_filter=%s limit=%s",
        items.height,
        judge_id or "all",
        limit if limit is not None else "none",
    )
    for judge in select_judges(config, judge_id):
        try:
            completed = run_judge(config, judge, materialized_items)
            logger.info("judge=%s summary wrote_new_judgments=%s", judge.id, completed)
        finally:
            clear_model_cache()


def main() -> None:
    """CLI entrypoint for judge evaluation."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    run_all(config, args.judge, args.limit)


if __name__ == "__main__":
    main()
