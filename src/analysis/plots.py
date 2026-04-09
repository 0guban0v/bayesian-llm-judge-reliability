"""Posterior plots and simple posterior predictive checks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.analysis.diagnostics import flatten_draws, load_posterior
from src.data.loader import ITEM_METADATA_COLUMNS
from src.logging_utils import configure_logging
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    """Save a matplotlib figure as PNG."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_judge_reliability_ridge(posterior: dict[str, np.ndarray]) -> plt.Figure:
    """Plot stacked posterior densities for judge reliability."""

    theta_samples = flatten_draws(posterior["theta"])
    judge_ids = posterior["judge_ids"]
    ordering = np.argsort(theta_samples.mean(axis=0))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(judge_ids))))
    x_grid = np.linspace(theta_samples.min() - 0.5, theta_samples.max() + 0.5, 200)
    for row_index, judge_index in enumerate(ordering):
        values = theta_samples[:, judge_index]
        hist, edges = np.histogram(values, bins=40, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = np.interp(x_grid, centers, hist, left=0.0, right=0.0)
        baseline = row_index * 1.1
        ax.fill_between(x_grid, baseline, baseline + density, alpha=0.7)
        ax.text(x_grid[0], baseline + 0.15, str(judge_ids[judge_index]), fontsize=9, va="bottom")
    ax.set_yticks([])
    ax.set_xlabel("Posterior reliability (theta)")
    ax.set_title("Judge Reliability Posterior Ridge Plot")
    fig.tight_layout()
    return fig


def plot_item_parameter_scatter(posterior: dict[str, np.ndarray]) -> plt.Figure:
    """Scatter plot of item difficulty versus discrimination."""

    b_mean = flatten_draws(posterior["b"]).mean(axis=0)
    a_mean = (
        flatten_draws(posterior["a"]).mean(axis=0) if "a" in posterior else np.ones_like(b_mean)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(b_mean, a_mean, alpha=0.75)
    ax.set_xlabel("Item difficulty mean (b)")
    ax.set_ylabel("Item discrimination mean (a)")
    ax.set_title("Item Parameter Summary")
    fig.tight_layout()
    return fig


def observed_accuracy(matrix: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return judge IDs and observed accuracies from the processed matrix."""

    judge_ids = np.asarray(
        [column for column in matrix.columns if column not in ITEM_METADATA_COLUMNS]
    )
    accuracies = np.asarray(
        [float(matrix.get_column(judge_id).drop_nulls().mean() or 0.0) for judge_id in judge_ids]
    )
    return judge_ids, accuracies


def validate_posterior_judge_order(
    matrix_judge_ids: np.ndarray,
    posterior: dict[str, np.ndarray],
) -> None:
    """Ensure posterior judge order matches the processed matrix column order."""

    posterior_judge_ids = np.asarray(posterior["judge_ids"], dtype=str)
    if np.array_equal(matrix_judge_ids, posterior_judge_ids):
        return
    msg = (
        "Posterior judge order does not match matrix judge column order. "
        f"matrix={matrix_judge_ids.tolist()} posterior={posterior_judge_ids.tolist()}"
    )
    raise ValueError(msg)


def posterior_predictive_judge_accuracy(
    posterior: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate posterior predictive judge accuracy intervals."""

    theta = flatten_draws(posterior["theta"])
    b = flatten_draws(posterior["b"])
    a = flatten_draws(posterior["a"]) if "a" in posterior else np.ones_like(b)
    max_draws = min(theta.shape[0], 250)
    theta = theta[:max_draws]
    b = b[:max_draws]
    a = a[:max_draws]
    predictive = []
    for draw_index in range(max_draws):
        logits = a[draw_index][None, :] * (theta[draw_index][:, None] - b[draw_index][None, :])
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        predictive.append(probabilities.mean(axis=1))
    predictive_array = np.asarray(predictive)
    return (
        predictive_array.mean(axis=0),
        np.quantile(predictive_array, 0.05, axis=0),
        np.quantile(predictive_array, 0.95, axis=0),
    )


def plot_posterior_predictive_check(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Compare observed judge accuracy to posterior predictive intervals."""

    judge_ids, observed = observed_accuracy(matrix)
    validate_posterior_judge_order(judge_ids, posterior)
    predicted_mean, lower, upper = posterior_predictive_judge_accuracy(posterior)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(judge_ids))))
    y_positions = np.arange(len(judge_ids))
    ax.errorbar(
        predicted_mean,
        y_positions,
        xerr=[predicted_mean - lower, upper - predicted_mean],
        fmt="o",
        label="posterior predictive",
    )
    ax.scatter(observed, y_positions, marker="s", label="observed")
    ax.set_yticks(y_positions, judge_ids)
    ax.set_xlabel("Accuracy")
    ax.set_title("Posterior Predictive Check by Judge")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    """CLI entrypoint for posterior plots."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    posterior_path = config.inference.posterior_path
    posterior = load_posterior(posterior_path)
    matrix = pl.read_parquet(config.data.matrix_path)
    figures_dir = Path("figures")
    logger.info(
        "loaded posterior from %s using backend=%s",
        posterior_path,
        config.inference.backend,
    )
    save_figure(plot_judge_reliability_ridge(posterior), figures_dir / "judge_reliability_ridge")
    save_figure(plot_item_parameter_scatter(posterior), figures_dir / "item_parameter_scatter")
    save_figure(
        plot_posterior_predictive_check(matrix, posterior),
        figures_dir / "posterior_predictive",
    )
    hero_figure = plot_judge_reliability_ridge(posterior)
    hero_figure.savefig(figures_dir / "hero.png", dpi=300, bbox_inches="tight")
    plt.close(hero_figure)
    logger.info("saved posterior figures to %s", figures_dir)


if __name__ == "__main__":
    main()
