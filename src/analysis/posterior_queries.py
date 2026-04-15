"""Posterior comparison helpers for judge reliabilities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.analysis.diagnostics import flatten_draws
from src.analysis.posterior_archive import load_posterior
from src.logging_utils import configure_logging, format_table_for_log
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for posterior comparison queries."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    parser.add_argument(
        "--judge-a",
        type=str,
        default=None,
        help="Judge ID for the first comparison.",
    )
    parser.add_argument(
        "--judge-b",
        type=str,
        default=None,
        help="Judge ID for the second comparison.",
    )
    return parser.parse_args()


def judge_index_map(judge_ids: np.ndarray) -> dict[str, int]:
    """Map judge IDs to column indices in posterior arrays."""

    return {str(judge_id): index for index, judge_id in enumerate(judge_ids.tolist())}


def resolve_judge_indices(
    posterior: dict[str, np.ndarray],
    judge_a: str,
    judge_b: str,
) -> tuple[int, int]:
    """Resolve two judge IDs to posterior indices with a user-facing error."""

    mapping = judge_index_map(posterior["judge_ids"])
    try:
        return mapping[judge_a], mapping[judge_b]
    except KeyError as exc:
        available = ", ".join(sorted(mapping))
        missing = exc.args[0]
        msg = f"Unknown judge ID '{missing}'. Available judge IDs: {available}"
        raise ValueError(msg) from exc


def credible_interval(samples: np.ndarray, level: float = 0.9) -> tuple[float, float]:
    """Return an equal-tailed credible interval."""

    alpha = (1.0 - level) / 2.0
    return float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha))


def probability_judge_a_exceeds_b(
    posterior: dict[str, np.ndarray],
    judge_a: str,
    judge_b: str,
) -> float:
    """Compute `P(theta_a > theta_b | data)`."""

    theta_samples = flatten_draws(posterior["theta"])
    judge_a_index, judge_b_index = resolve_judge_indices(posterior, judge_a, judge_b)
    difference = theta_samples[:, judge_a_index] - theta_samples[:, judge_b_index]
    return float((difference > 0.0).mean())


def effect_size(
    posterior: dict[str, np.ndarray],
    judge_a: str,
    judge_b: str,
) -> float:
    """Compute the standardized posterior mean difference between two judges."""

    theta_samples = flatten_draws(posterior["theta"])
    judge_a_index, judge_b_index = resolve_judge_indices(posterior, judge_a, judge_b)
    difference = theta_samples[:, judge_a_index] - theta_samples[:, judge_b_index]
    return float(difference.mean() / difference.std())


def rank_judges(posterior: dict[str, np.ndarray]) -> pl.DataFrame:
    """Rank judges by posterior mean reliability."""

    theta_samples = flatten_draws(posterior["theta"])
    means = theta_samples.mean(axis=0)
    lowers = np.quantile(theta_samples, 0.05, axis=0)
    uppers = np.quantile(theta_samples, 0.95, axis=0)
    return pl.DataFrame(
        {
            "judge_id": posterior["judge_ids"].tolist(),
            "theta_mean": means,
            "theta_p05": lowers,
            "theta_p95": uppers,
        }
    ).sort("theta_mean", descending=True)


def main() -> None:
    """CLI entrypoint for posterior summary queries."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    posterior_path = config.inference.posterior_path
    posterior = load_posterior(posterior_path)
    backend = str(posterior.get("backend", "unknown"))
    logger.info(
        "loaded posterior from %s using backend=%s",
        posterior_path,
        backend,
    )
    if args.judge_a and args.judge_b:
        probability = probability_judge_a_exceeds_b(posterior, args.judge_a, args.judge_b)
        theta_samples = flatten_draws(posterior["theta"])
        judge_a_index, judge_b_index = resolve_judge_indices(
            posterior,
            args.judge_a,
            args.judge_b,
        )
        difference = theta_samples[:, judge_a_index] - theta_samples[:, judge_b_index]
        interval = credible_interval(difference)
        standardized = effect_size(posterior, args.judge_a, args.judge_b)
        logger.info("p(theta_%s > theta_%s) = %.3f", args.judge_a, args.judge_b, probability)
        logger.info("90%% credible interval = (%.3f, %.3f)", interval[0], interval[1])
        logger.info("standardized difference = %.3f", standardized)
    else:
        if logger.isEnabledFor(logging.INFO):
            logger.info("judge ranking\n%s", format_table_for_log(rank_judges(posterior)))


if __name__ == "__main__":
    main()
