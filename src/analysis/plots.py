"""Posterior plots and simple posterior predictive checks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.analysis.figure_paths import (
    JUDGE_ACCURACY_PPC_STEM,
    JUDGE_RELIABILITY_BY_SOURCE_STEM,
    JUDGE_RELIABILITY_RIDGE_STEM,
    PRIOR_PREDICTIVE_STEM,
    figure_base_path,
    remove_figure_output,
)
from src.analysis.plot_config import (
    EXPORT_DPI,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_TICK,
    JUDGE_LABEL_PINS,
    judge_color_map,
    source_display_label,
    style_axis,
)
from src.analysis.posterior_archive import load_posterior
from src.analysis.posterior_utils import (
    flatten_draws,
    has_source_reliability,
    judge_accuracy_ppc_summaries,
    observed_accuracy,
    source_reliability_summary,
    top_source_ids,
    validate_posterior_judge_order,
    validate_posterior_plot_inputs,
)
from src.logging_utils import configure_logging
from src.models.irt_common import build_model_priors, sample_prior_values
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Return a numerically stable logistic transform for arbitrary real inputs."""

    positive_mask = values >= 0.0
    negative_values = values[~positive_mask]
    result = np.empty_like(values, dtype=float)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-values[positive_mask]))
    exp_values = np.exp(negative_values)
    result[~positive_mask] = exp_values / (1.0 + exp_values)
    return result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    """Save a matplotlib figure as PNG."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def cleanup_posterior_figure_outputs(
    figures_dir: Path,
    *,
    keep_ridge: bool,
    keep_source: bool,
    keep_ppc: bool,
) -> None:
    """Remove posterior-backed figures that are not valid for current run."""

    if not keep_ridge:
        remove_figure_output(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM)
    if not keep_source:
        remove_figure_output(figures_dir, JUDGE_RELIABILITY_BY_SOURCE_STEM)
    if not keep_ppc:
        remove_figure_output(figures_dir, JUDGE_ACCURACY_PPC_STEM)


def sample_prior_predictive_probabilities(
    matrix: pl.DataFrame,
    config: ExperimentConfig,
    *,
    num_draws: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample prior predictive correctness probabilities and judge-average accuracies."""

    priors = build_model_priors(config.model)
    rng = np.random.default_rng(config.experiment.seed)
    n_items = matrix.height
    n_judges = len(config.judges)
    theta = sample_prior_values(priors.theta, rng=rng, size=(num_draws, n_judges))
    b = sample_prior_values(priors.b, rng=rng, size=(num_draws, n_items))
    if config.model.type == "2PL":
        a = sample_prior_values(priors.a, rng=rng, size=(num_draws, n_items))
    else:
        a = np.ones((num_draws, n_items))
    if config.model.variant == "source_hier":
        if priors.tau_theta is None:
            raise ValueError("source_hier prior predictive simulation requires tau_theta")
        source_ids = matrix.get_column("source").cast(pl.String).unique(maintain_order=True).to_list()
        source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
        item_source_idx = np.asarray(
            [source_lookup[source_id] for source_id in matrix.get_column("source").cast(pl.String)],
            dtype=int,
        )
        tau_theta = sample_prior_values(priors.tau_theta, rng=rng, size=(num_draws, n_judges))
        theta_source = rng.normal(
            loc=theta[:, :, None],
            scale=tau_theta[:, :, None],
            size=(num_draws, n_judges, len(source_ids)),
        )
        theta_by_item = np.take(theta_source, item_source_idx, axis=2)
        logits = a[:, None, :] * (theta_by_item - b[:, None, :])
    else:
        logits = a[:, None, :] * (theta[:, :, None] - b[:, None, :])
    probabilities = stable_sigmoid(logits)
    return probabilities.reshape(-1), probabilities.mean(axis=2).reshape(-1)


def plot_prior_predictive_probabilities(
    matrix: pl.DataFrame,
    config: ExperimentConfig,
    *,
    num_draws: int = 1000,
) -> plt.Figure:
    """Plot prior predictive judge-mean calibration on the probability scale."""

    _, judge_means = sample_prior_predictive_probabilities(
        matrix,
        config,
        num_draws=num_draws,
    )
    prior_color = "#7a8793"
    fig, mean_axis = plt.subplots(figsize=(6.2, 3.8))
    mean_axis.axvspan(0.45, 0.55, color="#d9dde3", alpha=0.55, zorder=0)
    mean_axis.hist(
        judge_means,
        bins=30,
        density=True,
        histtype="step",
        color=prior_color,
        linewidth=1.4,
    )
    mean_axis.axvline(0.5, color="#9aa0a6", linestyle="--", linewidth=1.0)
    mean_axis.set_xlabel("Prior predictive judge mean accuracy")
    mean_axis.set_ylabel("Density")
    style_axis(mean_axis)
    fig.tight_layout()
    return fig


def plot_judge_accuracy_ppc(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot observed judge accuracy against posterior predictive intervals."""

    observed_judge_ids, observed = observed_accuracy(matrix)
    validate_posterior_judge_order(observed_judge_ids, posterior)
    predicted_mean, predictive_lower, predictive_upper = judge_accuracy_ppc_summaries(posterior)
    judge_ids = np.asarray(posterior["judge_ids"], dtype=str)
    color_map = judge_color_map(judge_ids)
    ordering = np.argsort(flatten_draws(posterior["theta"]).mean(axis=0))[::-1]
    ordered_judge_ids = [str(judge_ids[index]) for index in ordering]
    fig_height = max(2.8, 0.8 * len(ordered_judge_ids) + 1.1)
    fig, axis = plt.subplots(figsize=(6.4, fig_height))
    accuracy_values = np.concatenate([observed, predicted_mean, predictive_lower, predictive_upper])
    accuracy_min = float(np.min(accuracy_values))
    accuracy_max = float(np.max(accuracy_values))
    accuracy_pad = max(0.01, 0.08 * (accuracy_max - accuracy_min))
    axis.set_xlim(max(0.0, accuracy_min - accuracy_pad), min(1.0, accuracy_max + accuracy_pad))

    for row_index, judge_id in enumerate(ordered_judge_ids):
        judge_index = int(np.where(judge_ids == judge_id)[0][0])
        baseline = len(ordered_judge_ids) - 1 - row_index
        axis.hlines(
            baseline,
            float(predictive_lower[judge_index]),
            float(predictive_upper[judge_index]),
            color=color_map[judge_id],
            linewidth=1.6,
            alpha=0.95,
        )
        axis.scatter(
            float(predicted_mean[judge_index]),
            baseline,
            marker="o",
            s=34,
            color=color_map[judge_id],
            zorder=3,
        )
        axis.scatter(
            float(observed[judge_index]),
            baseline,
            marker="s",
            s=34,
            color=color_map[judge_id],
            zorder=4,
        )

    axis.set_yticks(np.arange(len(ordered_judge_ids)))
    axis.set_yticklabels(
        [JUDGE_LABEL_PINS.get(judge_id, judge_id) for judge_id in reversed(ordered_judge_ids)],
        fontsize=FONT_SIZE_TICK,
    )
    axis.set_xlabel("Accuracy")
    axis.text(
        0.98,
        0.98,
        "square = observed\ncircle = predicted\nline = 90% interval",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE_ANNOTATION,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.5},
    )
    style_axis(axis)
    fig.tight_layout()
    return fig


def plot_judge_reliability_ridge(
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot stacked posterior densities for judge reliability."""

    theta_samples = flatten_draws(posterior["theta"])
    judge_ids = posterior["judge_ids"]
    color_map = judge_color_map(judge_ids)
    ordering = np.argsort(theta_samples.mean(axis=0))
    fig, ridge_axis = plt.subplots(figsize=(6.2, 8.0))
    x_min = float(np.quantile(theta_samples, 0.01))
    x_max = float(np.quantile(theta_samples, 0.99))
    x_span = max(1e-6, x_max - x_min)
    label_x = x_min - 0.28 * x_span
    for row_index, judge_index in enumerate(ordering):
        values = theta_samples[:, judge_index]
        hist, edges = np.histogram(values, bins=40, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        positive_mask = hist > 0.0
        centers = centers[positive_mask]
        density = hist[positive_mask]
        if density.size == 0:
            continue
        baseline = row_index * 1.1
        judge_id = str(judge_ids[judge_index])
        ridge_axis.fill_between(
            centers,
            baseline,
            baseline + density,
            alpha=0.6,
            color=color_map[judge_id],
        )
        peak_height = float(density.max())
        ridge_axis.text(
            label_x,
            baseline + peak_height * 0.48,
            JUDGE_LABEL_PINS.get(judge_id, judge_id),
            ha="left",
            va="center",
            fontsize=FONT_SIZE_TICK,
            color=color_map[judge_id],
            fontweight="semibold",
        )
    ridge_axis.set_xlim(label_x - 0.02 * x_span, x_max + 0.25 * x_span)
    ridge_axis.set_yticks([])
    ridge_axis.spines["left"].set_visible(False)
    ridge_axis.set_xlabel("Posterior reliability (theta)")
    style_axis(ridge_axis)
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.12, top=0.98)
    return fig


def staggered_heatmap_judge_labels(judge_ids: list[str]) -> list[str]:
    """Return centered heatmap labels with alternating vertical staggering."""

    labels = [JUDGE_LABEL_PINS.get(judge_id, judge_id) for judge_id in judge_ids]
    return [label if index % 2 == 0 else f"\n{label}" for index, label in enumerate(labels)]


def plot_judge_reliability_by_source(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
    *,
    max_sources: int = 8,
) -> plt.Figure:
    """Plot source-specific judge reliability means as an annotated heatmap."""

    if "theta_source" not in posterior or "source_ids" not in posterior:
        raise ValueError("Posterior does not contain source-aware reliability samples.")
    ordered_sources = top_source_ids(matrix, posterior, max_sources=max_sources)
    if not ordered_sources:
        raise ValueError("Source facet plotting requires at least one source.")
    summary = source_reliability_summary(posterior, ordered_sources)
    summary_sources = summary.get_column("source").cast(pl.String).unique(maintain_order=True).to_list()
    if summary_sources != ordered_sources:
        raise ValueError(
            "Source reliability summary does not match requested source order. "
            f"summary={summary_sources} requested={ordered_sources}"
        )
    judge_ids = np.asarray(posterior["judge_ids"], dtype=str)
    global_theta = flatten_draws(posterior["theta"]).mean(axis=0)
    global_ordering = np.argsort(global_theta)[::-1]
    ordered_judge_ids = [str(judge_ids[index]) for index in global_ordering]
    pivoted = summary.pivot(on="judge_id", index="source", values="theta_mean")
    heatmap = np.column_stack(
        [pivoted.get_column(judge_id).to_numpy().astype(float) for judge_id in ordered_judge_ids]
    ).T
    transposed_heatmap = heatmap.T
    max_abs = float(np.max(np.abs(heatmap))) if heatmap.size else 1.0
    color_limit = max(0.25, max_abs)
    side = max(6.5, 0.95 * max(len(ordered_sources), len(ordered_judge_ids)))
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(
        transposed_heatmap,
        cmap="RdBu",
        vmin=-color_limit,
        vmax=color_limit,
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(ordered_judge_ids)))
    ax.set_xticklabels(
        staggered_heatmap_judge_labels(ordered_judge_ids),
        fontsize=FONT_SIZE_TICK,
    )
    ax.set_yticks(np.arange(len(ordered_sources)))
    ax.set_yticklabels(
        [source_display_label(source_id) for source_id in ordered_sources],
        fontsize=FONT_SIZE_TICK,
    )
    for row_index, _source_id in enumerate(ordered_sources):
        for column_index, _judge_id in enumerate(ordered_judge_ids):
            value = transposed_heatmap[row_index, column_index]
            text_color = "white" if abs(value) > 0.45 * color_limit else "#202124"
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=FONT_SIZE_ANNOTATION,
                color=text_color,
                fontweight="bold" if abs(value) > 0.5 else "normal",
            )
    colorbar_axis = inset_axes(
        ax,
        width="3.5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    fig.colorbar(image, cax=colorbar_axis)
    style_axis(ax)
    fig.subplots_adjust(left=0.2, right=0.88, bottom=0.18, top=0.94)
    return fig


def main() -> None:
    """CLI entrypoint for posterior plots."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    posterior_path = config.inference.posterior_path
    matrix = pl.read_parquet(config.data.matrix_path)
    figures_dir = config.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_figure(
        plot_prior_predictive_probabilities(matrix, config),
        figure_base_path(figures_dir, PRIOR_PREDICTIVE_STEM),
    )
    if not posterior_path.exists():
        cleanup_posterior_figure_outputs(figures_dir, keep_ridge=False, keep_source=False, keep_ppc=False)
        logger.warning(
            "posterior not found at %s; saved prior predictive figure only to %s",
            posterior_path,
            figures_dir,
        )
        return

    posterior = load_posterior(posterior_path)
    backend = str(posterior.get("backend", "unknown"))
    logger.info("loaded posterior from %s using backend=%s", posterior_path, backend)
    validate_posterior_plot_inputs(matrix, posterior)
    save_figure(
        plot_judge_accuracy_ppc(matrix, posterior),
        figure_base_path(figures_dir, JUDGE_ACCURACY_PPC_STEM),
    )
    save_figure(
        plot_judge_reliability_ridge(posterior),
        figure_base_path(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM),
    )
    if has_source_reliability(posterior):
        save_figure(
            plot_judge_reliability_by_source(matrix, posterior, max_sources=config.analysis.plots.max_sources),
            figure_base_path(figures_dir, JUDGE_RELIABILITY_BY_SOURCE_STEM),
        )
    else:
        cleanup_posterior_figure_outputs(figures_dir, keep_ridge=True, keep_source=False, keep_ppc=True)
    logger.info("saved posterior figures to %s", figures_dir)


if __name__ == "__main__":
    main()
