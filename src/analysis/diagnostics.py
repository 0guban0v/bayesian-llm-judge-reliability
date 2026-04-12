"""Diagnostics and diagnostic plots for saved posterior samples."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.logging_utils import configure_logging, format_table_for_log
from src.schemas import ExperimentConfig

POSTERIOR_METADATA_KEYS = {
    "judge_ids",
    "item_ids",
    "source_ids",
    "diverging",
    "model_type",
    "n_obs",
    "backend",
    "experiment_seed",
    "num_chains",
}
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for posterior diagnostics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def load_posterior(path: Path) -> dict[str, np.ndarray]:
    """Load a saved posterior archive."""

    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _flatten_parameter(samples: np.ndarray) -> np.ndarray:
    """Flatten all parameter dimensions while preserving chain and draw axes."""

    if samples.ndim < 2:
        raise ValueError("Posterior arrays must have chain and draw axes.")
    return samples.reshape(samples.shape[0], samples.shape[1], -1)


def flatten_draws(samples: np.ndarray) -> np.ndarray:
    """Collapse chain and draw axes into a single posterior sample axis."""

    flattened = _flatten_parameter(samples)
    return flattened.reshape(-1, flattened.shape[-1])


def _split_chains(samples: np.ndarray) -> np.ndarray:
    """Split each chain in half for split-R-hat computation."""

    flattened = _flatten_parameter(samples)
    half_draws = flattened.shape[1] // 2
    if half_draws < 2:
        return np.empty((0, 0, flattened.shape[-1]), dtype=flattened.dtype)
    first_half = flattened[:, :half_draws, :]
    second_half = flattened[:, -half_draws:, :]
    return np.concatenate([first_half, second_half], axis=0)


def _compute_standard_rhat(flattened: np.ndarray) -> np.ndarray:
    """Compute standard R-hat for already-flattened chain draws."""

    chains, draws, features = flattened.shape
    if chains < 2 or draws < 2:
        return np.full(features, np.nan)
    chain_means = flattened.mean(axis=1)
    overall_mean = chain_means.mean(axis=0)
    between = draws * ((chain_means - overall_mean) ** 2).sum(axis=0) / (chains - 1)
    within = flattened.var(axis=1, ddof=1).mean(axis=0)
    variance_estimate = ((draws - 1) / draws) * within + between / draws
    ratio = np.divide(
        variance_estimate,
        within,
        out=np.full(features, np.nan, dtype=float),
        where=within != 0,
    )
    return np.sqrt(ratio)


def compute_rhat(samples: np.ndarray) -> np.ndarray:
    """Compute split-R-hat for each flattened parameter dimension."""

    split_samples = _split_chains(samples)
    if split_samples.size == 0:
        feature_count = _flatten_parameter(samples).shape[-1]
        return np.full(feature_count, np.nan)
    return _compute_standard_rhat(split_samples)


def _autocorrelation_1d(series: np.ndarray) -> np.ndarray:
    """Compute the autocorrelation sequence for a 1D series."""

    centered = series - series.mean()
    variance = np.dot(centered, centered)
    if variance == 0.0:
        return np.ones(series.shape[0])
    correlation = np.correlate(centered, centered, mode="full")[series.shape[0] - 1 :]
    return correlation / variance


def compute_ess(samples: np.ndarray) -> np.ndarray:
    """Approximate effective sample size for each flattened parameter dimension."""

    flattened = _flatten_parameter(samples)
    chains, draws, features = flattened.shape
    ess = np.empty(features, dtype=float)
    for feature_index in range(features):
        acov = np.mean(
            [
                _autocorrelation_1d(flattened[chain_index, :, feature_index])
                for chain_index in range(chains)
            ],
            axis=0,
        )
        positive_sum = 0.0
        for lag in range(1, draws - 1, 2):
            pair_sum = acov[lag] + acov[lag + 1]
            if pair_sum <= 0.0:
                break
            positive_sum += pair_sum
        ess[feature_index] = chains * draws / (1.0 + 2.0 * positive_sum)
    return ess


def summarize_diagnostics(posterior: dict[str, np.ndarray]) -> pl.DataFrame:
    """Build a compact diagnostic summary table."""

    rows: list[dict[str, float | int | str]] = []
    divergences = int(np.asarray(posterior.get("diverging", np.array([]))).sum())
    for name, values in posterior.items():
        if name in POSTERIOR_METADATA_KEYS or name == "diverging":
            continue
        rhat = compute_rhat(values)
        ess = compute_ess(values)
        rows.append(
            {
                "parameter": name,
                "rhat_max": float(np.nanmax(rhat)),
                "ess_min": float(np.nanmin(ess)),
                "divergences": divergences,
            }
        )
    return pl.DataFrame(rows).sort("parameter")


def infer_chain_count(posterior: dict[str, np.ndarray]) -> int:
    """Infer the number of chains from the first posterior parameter array."""

    for name, values in posterior.items():
        if name in POSTERIOR_METADATA_KEYS or name == "diverging":
            continue
        return int(values.shape[0])
    raise ValueError("Posterior archive does not contain any parameter arrays.")


def plot_trace(samples: np.ndarray, labels: np.ndarray, title: str) -> plt.Figure:
    """Create a trace plot for the first few parameter coordinates."""

    flattened = _flatten_parameter(samples)
    feature_count = min(flattened.shape[-1], len(labels), 5)
    fig, axes = plt.subplots(feature_count, 1, figsize=(10, 2.4 * feature_count), sharex=True)
    if feature_count == 1:
        axes = np.asarray([axes])
    for axis, feature_index in zip(axes, range(feature_count), strict=False):
        for chain_index in range(flattened.shape[0]):
            axis.plot(flattened[chain_index, :, feature_index], alpha=0.8, linewidth=0.9)
        axis.set_title(str(labels[feature_index]))
    axes[0].figure.suptitle(title)
    axes[-1].set_xlabel("draw")
    fig.tight_layout()
    return fig


def plot_rank_histogram(samples: np.ndarray, labels: np.ndarray, title: str) -> plt.Figure:
    """Create rank histograms for the first few parameter coordinates."""

    flattened = _flatten_parameter(samples)
    feature_count = min(flattened.shape[-1], len(labels), 5)
    fig, axes = plt.subplots(feature_count, 1, figsize=(10, 2.2 * feature_count))
    if feature_count == 1:
        axes = np.asarray([axes])
    for axis, feature_index in zip(axes, range(feature_count), strict=False):
        pooled = flattened[:, :, feature_index].reshape(-1)
        for chain_index in range(flattened.shape[0]):
            chain_values = flattened[chain_index, :, feature_index]
            ranks = np.searchsorted(np.sort(pooled), chain_values, side="left")
            axis.hist(ranks, bins=20, alpha=0.5, density=True)
        axis.set_title(str(labels[feature_index]))
    axes[0].figure.suptitle(title)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    """Save a matplotlib figure as PNG."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entrypoint for diagnostic summaries and plots."""

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
    chain_count = infer_chain_count(posterior)
    if chain_count < 2:
        logger.warning(
            (
                "posterior contains %s chain; R-hat is computed from split single-chain "
                "draws, but multi-chain inference is recommended for a more robust "
                "convergence assessment"
            ),
            chain_count,
        )
    summary = summarize_diagnostics(posterior)
    if logger.isEnabledFor(logging.INFO):
        logger.info("diagnostic summary\n%s", format_table_for_log(summary))
    judge_ids = posterior["judge_ids"]
    item_ids = posterior["item_ids"]
    trace_dir = config.figures_dir / "diagnostics"
    save_figure(
        plot_trace(posterior["theta"], judge_ids, "Judge Reliability Trace"),
        trace_dir / "theta_trace",
    )
    save_figure(
        plot_trace(posterior["b"], item_ids, "Item Difficulty Trace"),
        trace_dir / "b_trace",
    )
    save_figure(
        plot_rank_histogram(posterior["theta"], judge_ids, "Judge Reliability Rank Plot"),
        trace_dir / "theta_rank",
    )
    logger.info("saved diagnostic figures to %s", trace_dir)
    if "a" in posterior:
        save_figure(
            plot_rank_histogram(posterior["a"], item_ids, "Item Discrimination Rank Plot"),
            trace_dir / "a_rank",
        )


if __name__ == "__main__":
    main()
