"""Posterior plots and simple posterior predictive checks."""

from __future__ import annotations

import argparse
import colorsys
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.analysis.figure_paths import (
    JUDGE_ACCURACY_PPC_STEM,
    JUDGE_RELIABILITY_BY_SOURCE_STEM,
    JUDGE_RELIABILITY_RIDGE_STEM,
    SEPARATION_STEM,
    TRACE_THETA_TAU_STEM,
    figure_base_path,
    remove_figure_output,
)
from src.analysis.plot_config import (
    COLOR_SURFACE,
    COLOR_TEXT_DARK,
    COLOR_TEXT_LIGHT,
    EXPORT_DPI,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_TICK,
    FONT_SIZE_TITLE,
    judge_color_map,
    judge_display_label,
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
    keep_trace: bool,
    keep_separation: bool,
) -> None:
    """Remove posterior-backed figures that are not valid for current run."""

    if not keep_ridge:
        remove_figure_output(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM)
    if not keep_source:
        remove_figure_output(figures_dir, JUDGE_RELIABILITY_BY_SOURCE_STEM)
    if not keep_ppc:
        remove_figure_output(figures_dir, JUDGE_ACCURACY_PPC_STEM)
    if not keep_trace:
        remove_figure_output(figures_dir, TRACE_THETA_TAU_STEM)
    if not keep_separation:
        remove_figure_output(figures_dir, SEPARATION_STEM)


def _lighten_color(color: str, amount: float) -> str:
    """Blend a color toward white by a fixed amount."""

    red, green, blue = to_rgb(color)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
    lightness = min(1.0, lightness + (1.0 - lightness) * amount)
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


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
        [judge_display_label(judge_id) for judge_id in reversed(ordered_judge_ids)],
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
        bbox={"facecolor": COLOR_SURFACE, "edgecolor": "none", "alpha": 0.9, "pad": 2.5},
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
            judge_display_label(judge_id),
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


def plot_trace_theta_tau_theta(posterior: dict[str, np.ndarray]) -> plt.Figure:
    """Plot chain traces and per-chain densities for judge-level theta and tau_theta."""

    judge_ids = np.asarray(posterior["judge_ids"], dtype=str)
    judge_labels = [judge_display_label(judge_id) for judge_id in judge_ids]
    color_map = judge_color_map(judge_ids)
    parameter_blocks: list[tuple[str, str, np.ndarray]] = [
        ("theta", "theta", np.asarray(posterior["theta"], dtype=float))
    ]
    if "tau_theta" in posterior:
        parameter_blocks.append(("tau_theta", "tau_theta", np.asarray(posterior["tau_theta"], dtype=float)))
    row_specs: list[tuple[str, str, np.ndarray]] = []
    for _parameter_name, symbol, values in parameter_blocks:
        for judge_index, judge_label in enumerate(judge_labels):
            row_specs.append((f"{symbol}\n{judge_label}", str(judge_ids[judge_index]), values[:, :, judge_index]))
    n_rows = len(row_specs)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(9.6, max(4.8, 1.05 * n_rows + 0.6)),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.2, "hspace": 0.55},
        squeeze=False,
    )
    for row_index, (row_label, judge_id, chain_draws) in enumerate(row_specs):
        trace_axis, density_axis = axes[row_index]
        judge_color = color_map[judge_id]
        n_chains, n_draws = chain_draws.shape
        draw_index = np.arange(n_draws)
        for chain_index in range(n_chains):
            trace_axis.plot(
                draw_index,
                chain_draws[chain_index],
                color=judge_color,
                linewidth=1.0,
                alpha=0.45 + 0.2 * chain_index,
            )
            hist, edges = np.histogram(chain_draws[chain_index], bins=36, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            positive_mask = hist > 0.0
            density_axis.plot(
                centers[positive_mask],
                hist[positive_mask],
                color=judge_color,
                linewidth=1.1,
                alpha=0.45 + 0.2 * chain_index,
            )
        trace_axis.text(
            -0.18,
            0.5,
            row_label,
            transform=trace_axis.transAxes,
            ha="right",
            va="center",
            fontsize=FONT_SIZE_TICK,
            color=judge_color,
            fontweight="semibold",
            linespacing=1.1,
        )
        if row_index == 0:
            trace_axis.set_title("Trace", fontsize=FONT_SIZE_TITLE)
            density_axis.set_title("Density", fontsize=FONT_SIZE_TITLE)
        if row_index != n_rows - 1:
            trace_axis.tick_params(axis="x", labelbottom=False)
            density_axis.tick_params(axis="x", labelbottom=False)
        else:
            trace_axis.set_xlabel("Draw")
            density_axis.set_xlabel("Value")
        trace_axis.tick_params(axis="both", labelsize=FONT_SIZE_TICK - 0.5)
        density_axis.tick_params(axis="both", labelsize=FONT_SIZE_TICK - 0.5)
        style_axis(trace_axis)
        style_axis(density_axis)
    fig.subplots_adjust(left=0.3, right=0.98, bottom=0.08, top=0.96)
    return fig


def posterior_mean_item_probabilities(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> np.ndarray:
    """Return posterior mean correctness probability for each judge-item pair."""

    theta = flatten_draws(np.asarray(posterior["theta"], dtype=float))
    b = flatten_draws(np.asarray(posterior["b"], dtype=float))
    a = flatten_draws(np.asarray(posterior["a"], dtype=float)) if "a" in posterior else np.ones_like(b)
    if has_source_reliability(posterior):
        theta_source = np.asarray(posterior["theta_source"], dtype=float).reshape(
            -1,
            posterior["theta_source"].shape[2],
            posterior["theta_source"].shape[3],
        )
        source_ids = [str(source_id) for source_id in posterior["source_ids"]]
        source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
        item_source_idx = np.asarray(
            [source_lookup[str(source_id)] for source_id in matrix.get_column("source").cast(pl.String)],
            dtype=int,
        )
        judge_term = np.take(theta_source, item_source_idx, axis=2)
    else:
        judge_term = theta[:, :, None]
    logits = a[:, None, :] * (judge_term - b[:, None, :])
    return stable_sigmoid(logits).mean(axis=0)


def plot_separation_by_judge(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot judge-wise separation strips ordered by posterior mean item probability."""

    judge_ids = np.asarray(posterior["judge_ids"], dtype=str)
    judge_colors = judge_color_map(judge_ids)
    predicted = posterior_mean_item_probabilities(matrix, posterior)
    observed = matrix.select(judge_ids.tolist()).to_numpy().astype(float).T
    ordering = np.argsort(flatten_draws(posterior["theta"]).mean(axis=0))[::-1]
    ordered_judge_ids = [str(judge_ids[index]) for index in ordering]
    n_judges = len(ordered_judge_ids)
    fig, axes = plt.subplots(
        n_judges,
        1,
        figsize=(8.6, max(5.4, 1.15 * n_judges + 0.55)),
        sharex=True,
        squeeze=False,
    )
    for row_index, judge_id in enumerate(ordered_judge_ids):
        axis = axes[row_index, 0]
        judge_index = int(np.where(judge_ids == judge_id)[0][0])
        observed_values = observed[judge_index]
        predicted_values = predicted[judge_index]
        valid_mask = ~np.isnan(observed_values)
        valid_observed = observed_values[valid_mask]
        valid_predicted = predicted_values[valid_mask]
        item_order = np.argsort(valid_predicted)
        sorted_observed = valid_observed[item_order]
        sorted_predicted = valid_predicted[item_order]
        positions = np.arange(len(sorted_predicted))
        judge_color = judge_colors[judge_id]
        incorrect_color = _lighten_color(judge_color, 0.8)
        separation_strip = np.where(sorted_observed > 0.5, 1.0, 0.0)[None, :]
        strip_top = 0.22
        curve_floor = 0.28
        axis.imshow(
            separation_strip,
            cmap=plt.matplotlib.colors.ListedColormap([incorrect_color, judge_color]),
            interpolation="nearest",
            aspect="auto",
            extent=(-0.5, len(sorted_predicted) - 0.5, 0.0, strip_top),
        )
        axis.axhline(strip_top, color=incorrect_color, linewidth=0.8, alpha=0.9)
        axis.fill_between(positions, curve_floor, sorted_predicted, color=judge_color, alpha=0.12, linewidth=0.0)
        axis.plot(positions, sorted_predicted, color=judge_color, linewidth=1.5)
        axis.text(
            0.03,
            0.96,
            judge_display_label(judge_id),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZE_TICK,
            color=judge_color,
            fontweight="semibold",
        )
        axis.set_ylim(0.0, 1.02)
        axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.tick_params(axis="y", pad=8)
        axis.tick_params(axis="y", labelsize=FONT_SIZE_TICK - 0.5)
        if row_index != n_judges - 1:
            axis.tick_params(axis="x", bottom=False, labelbottom=False)
        style_axis(axis)
    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.1, top=0.98, hspace=0.18)
    return fig


def staggered_heatmap_judge_labels(judge_ids: list[str]) -> list[str]:
    """Return centered heatmap labels with alternating vertical staggering."""

    labels = [judge_display_label(judge_id) for judge_id in judge_ids]
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
            text_color = COLOR_TEXT_LIGHT if abs(value) > 0.45 * color_limit else COLOR_TEXT_DARK
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

    if not posterior_path.exists():
        cleanup_posterior_figure_outputs(
            figures_dir,
            keep_ridge=False,
            keep_source=False,
            keep_ppc=False,
            keep_trace=False,
            keep_separation=False,
        )
        logger.warning(
            "posterior not found at %s; no posterior-backed figures saved to %s",
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
    save_figure(
        plot_trace_theta_tau_theta(posterior),
        figure_base_path(figures_dir, TRACE_THETA_TAU_STEM),
    )
    save_figure(
        plot_separation_by_judge(matrix, posterior),
        figure_base_path(figures_dir, SEPARATION_STEM),
    )
    if has_source_reliability(posterior):
        save_figure(
            plot_judge_reliability_by_source(matrix, posterior, max_sources=config.analysis.plots.max_sources),
            figure_base_path(figures_dir, JUDGE_RELIABILITY_BY_SOURCE_STEM),
        )
    else:
        cleanup_posterior_figure_outputs(
            figures_dir,
            keep_ridge=True,
            keep_source=False,
            keep_ppc=True,
            keep_trace=True,
            keep_separation=True,
        )
    logger.info("saved posterior figures to %s", figures_dir)


if __name__ == "__main__":
    main()
