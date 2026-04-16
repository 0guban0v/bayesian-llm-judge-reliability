"""Posterior plots and simple posterior predictive checks."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.analysis.diagnostics import flatten_draws
from src.analysis.figure_paths import (
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
from src.data.loader import ITEM_METADATA_COLUMNS
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


def sample_configured_prior(
    *,
    rng: np.random.Generator,
    dist_name: str,
    loc: float,
    scale: float,
    size: tuple[int, ...],
) -> np.ndarray:
    """Draw prior samples from the configured distribution family."""

    if dist_name == "normal":
        return rng.normal(loc, scale, size=size)
    if dist_name == "lognormal":
        return rng.lognormal(loc, scale, size=size)
    raise ValueError(f"Unsupported prior distribution '{dist_name}' for prior predictive draw")


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


def has_source_reliability(posterior: dict[str, np.ndarray] | None) -> bool:
    """Return whether a posterior includes source-specific reliability samples."""

    return posterior is not None and "theta_source" in posterior and "source_ids" in posterior


def cleanup_posterior_figure_outputs(
    figures_dir: Path,
    *,
    keep_ridge: bool,
    keep_source: bool,
) -> None:
    """Remove posterior-backed figures that are not valid for current run."""

    if not keep_ridge:
        remove_figure_output(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM)
    if not keep_source:
        remove_figure_output(figures_dir, JUDGE_RELIABILITY_BY_SOURCE_STEM)


def validate_posterior_plot_inputs(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> None:
    """Validate matrix and posterior alignment before posterior-backed plotting."""

    matrix_judge_ids, _ = observed_accuracy(matrix)
    validate_posterior_judge_order(matrix_judge_ids, posterior)
    validate_posterior_item_alignment(matrix, posterior)


def sample_prior_predictive_probabilities(
    matrix: pl.DataFrame,
    config: ExperimentConfig,
    *,
    num_draws: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample prior predictive correctness probabilities and judge-average accuracies."""

    priors = config.model.priors
    rng = np.random.default_rng(config.experiment.seed)
    n_items = matrix.height
    n_judges = len(config.judges)
    theta = sample_configured_prior(
        rng=rng,
        dist_name=priors.theta.dist,
        loc=priors.theta.loc,
        scale=priors.theta.scale,
        size=(num_draws, n_judges),
    )
    b = sample_configured_prior(
        rng=rng,
        dist_name=priors.b.dist,
        loc=priors.b.loc,
        scale=priors.b.scale,
        size=(num_draws, n_items),
    )
    if config.model.type == "2PL":
        a = sample_configured_prior(
            rng=rng,
            dist_name=priors.a.dist,
            loc=priors.a.loc,
            scale=priors.a.scale,
            size=(num_draws, n_items),
        )
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
        tau_theta = sample_configured_prior(
            rng=rng,
            dist_name=priors.tau_theta.dist,
            loc=priors.tau_theta.loc,
            scale=priors.tau_theta.scale,
            size=(num_draws, n_judges),
        )
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
    posterior: dict[str, np.ndarray] | None = None,
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
    posterior_predictive_overlay = None
    posterior_overlay_judge_ids: np.ndarray | None = None
    if posterior is not None and "theta" in posterior and "b" in posterior:
        posterior_overlay_judge_ids = np.asarray(posterior.get("judge_ids", []), dtype=str)
        posterior_predictive_overlay, _, _ = posterior_predictive_judge_accuracy(
            matrix,
            posterior,
        )
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
    if posterior_predictive_overlay is not None and posterior_predictive_overlay.size > 0:
        if posterior_overlay_judge_ids is not None and posterior_overlay_judge_ids.size > 0:
            overlay_color_map = judge_color_map(posterior_overlay_judge_ids)
            overlay_colors = [overlay_color_map[str(judge_id)] for judge_id in posterior_overlay_judge_ids]
        else:
            overlay_colors = ["#0f4c81"] * posterior_predictive_overlay.size
        for overlay_value, overlay_color in zip(
            posterior_predictive_overlay,
            overlay_colors,
            strict=False,
        ):
            mean_axis.vlines(
                float(overlay_value),
                ymin=0.0,
                ymax=0.08,
                transform=mean_axis.get_xaxis_transform(),
                color=overlay_color,
                linewidth=1.1,
                alpha=0.8,
            )
        mean_axis.text(
            0.98,
            0.98,
            "rug = posterior means",
            transform=mean_axis.transAxes,
            ha="right",
            va="top",
            fontsize=FONT_SIZE_ANNOTATION,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.5},
        )
    mean_axis.set_xlabel("Prior predictive judge mean accuracy")
    mean_axis.set_ylabel("Density")
    style_axis(mean_axis)
    fig.tight_layout()
    return fig


def plot_judge_reliability_ridge(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot stacked posterior densities for judge reliability."""

    theta_samples = flatten_draws(posterior["theta"])
    judge_ids = posterior["judge_ids"]
    color_map = judge_color_map(judge_ids)
    observed_judge_ids, observed = observed_accuracy(matrix)
    validate_posterior_judge_order(observed_judge_ids, posterior)
    predicted_mean, predictive_lower, predictive_upper = posterior_predictive_judge_accuracy(
        matrix,
        posterior,
    )
    accuracy_values = np.concatenate([observed, predicted_mean, predictive_lower, predictive_upper])
    observed_map = {str(judge_id): float(observed[index]) for index, judge_id in enumerate(observed_judge_ids)}
    predictive_map = {
        str(judge_id): (
            float(predicted_mean[index]),
            float(predictive_lower[index]),
            float(predictive_upper[index]),
        )
        for index, judge_id in enumerate(observed_judge_ids)
    }
    ordering = np.argsort(theta_samples.mean(axis=0))
    fig, (ridge_axis, adequacy_axis) = plt.subplots(
        1,
        2,
        figsize=(8.8, 8.0),
        gridspec_kw={"width_ratios": [4.6, 1.8], "wspace": 0.06},
        sharey=True,
    )
    x_min = float(np.quantile(theta_samples, 0.01))
    x_max = float(np.quantile(theta_samples, 0.99))
    x_span = max(1e-6, x_max - x_min)
    label_x = x_min - 0.28 * x_span
    accuracy_min = float(np.min(accuracy_values))
    accuracy_max = float(np.max(accuracy_values))
    accuracy_pad = max(0.005, 0.1 * (accuracy_max - accuracy_min))
    accuracy_left = max(0.0, accuracy_min - accuracy_pad)
    accuracy_right = min(1.0, accuracy_max + accuracy_pad)
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
        observed_value = observed_map[judge_id]
        predicted_value, lower_value, upper_value = predictive_map[judge_id]
        adequacy_axis.hlines(
            baseline,
            lower_value,
            upper_value,
            color=color_map[judge_id],
            linewidth=1.4,
            alpha=0.95,
        )
        adequacy_axis.scatter(
            observed_value,
            baseline,
            marker="s",
            s=32,
            color=color_map[judge_id],
            zorder=3,
        )
        adequacy_axis.scatter(
            predicted_value,
            baseline,
            marker="o",
            s=32,
            color=color_map[judge_id],
            zorder=3,
        )
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
    adequacy_axis.set_xlim(accuracy_left, accuracy_right)
    adequacy_axis.set_xlabel("Accuracy")
    adequacy_axis.set_yticks([])
    adequacy_axis.spines["left"].set_visible(False)
    adequacy_axis.text(
        0.98,
        0.98,
        "square = observed\ncircle = predicted\nline = 90% interval",
        transform=adequacy_axis.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE_ANNOTATION,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.5},
    )
    style_axis(adequacy_axis)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.98, wspace=0.06)
    return fig


def ordered_source_ids(matrix: pl.DataFrame, posterior: dict[str, np.ndarray]) -> list[str]:
    """Return source IDs ordered by observed item count, then posterior order."""

    posterior_source_ids = [str(source_id) for source_id in posterior.get("source_ids", [])]
    if not posterior_source_ids:
        return []
    posterior_index = {source_id: index for index, source_id in enumerate(posterior_source_ids)}
    source_counts = (
        matrix.group_by("source")
        .len()
        .rename({"len": "item_count"})
        .sort(["item_count", "source"], descending=[True, False])
    )
    counts_map = {str(row["source"]): int(row["item_count"]) for row in source_counts.to_dicts()}
    return sorted(
        posterior_source_ids,
        key=lambda source_id: (
            -counts_map.get(source_id, 0),
            posterior_index[source_id],
        ),
    )


def top_source_ids(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
    max_sources: int = 8,
) -> list[str]:
    """Return the most data-rich source IDs for small-multiple plotting."""

    return ordered_source_ids(matrix, posterior)[:max_sources]


def staggered_heatmap_judge_labels(judge_ids: list[str]) -> list[str]:
    """Return centered heatmap labels with alternating vertical staggering."""

    labels = [JUDGE_LABEL_PINS.get(judge_id, judge_id) for judge_id in judge_ids]
    return [label if index % 2 == 0 else f"\n{label}" for index, label in enumerate(labels)]


def source_reliability_summary(
    posterior: dict[str, np.ndarray],
    ordered_sources: list[str],
) -> pl.DataFrame:
    """Return posterior means and intervals for judge reliability by source."""

    if "theta_source" not in posterior or "source_ids" not in posterior:
        raise ValueError("Posterior does not contain source-aware reliability samples.")
    theta_source = posterior["theta_source"]
    flattened = theta_source.reshape(-1, theta_source.shape[-2], theta_source.shape[-1])
    judge_ids = [str(judge_id) for judge_id in posterior["judge_ids"].tolist()]
    source_ids = [str(source_id) for source_id in posterior["source_ids"].tolist()]
    source_index_map = {source_id: index for index, source_id in enumerate(source_ids)}
    rows: list[dict[str, float | str]] = []
    for source_id in ordered_sources:
        source_index = source_index_map[source_id]
        source_samples = flattened[:, :, source_index]
        means = source_samples.mean(axis=0)
        lowers = np.quantile(source_samples, 0.05, axis=0)
        uppers = np.quantile(source_samples, 0.95, axis=0)
        for judge_index, judge_id in enumerate(judge_ids):
            rows.append(
                {
                    "source": source_id,
                    "judge_id": judge_id,
                    "theta_mean": float(means[judge_index]),
                    "theta_p05": float(lowers[judge_index]),
                    "theta_p95": float(uppers[judge_index]),
                }
            )
    return pl.DataFrame(rows)


def observed_accuracy(matrix: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return judge IDs and observed accuracies from the processed matrix."""

    judge_ids = np.asarray([column for column in matrix.columns if column not in ITEM_METADATA_COLUMNS])
    accuracies = np.asarray([float(matrix.get_column(judge_id).drop_nulls().mean() or 0.0) for judge_id in judge_ids])
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


def validate_posterior_item_alignment(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> None:
    """Ensure matrix items and posterior items refer to the same ordered item set."""

    if "item_ids" not in posterior:
        raise ValueError("Posterior archive does not contain item_ids required for PPC alignment.")
    posterior_item_ids = np.asarray(posterior["item_ids"], dtype=str)
    matrix_item_ids = matrix.get_column("item_id").cast(pl.String).to_numpy()
    if not np.array_equal(matrix_item_ids, posterior_item_ids):
        msg = (
            "Posterior item_ids do not match matrix item_id order. "
            f"matrix={matrix_item_ids.tolist()} posterior={posterior_item_ids.tolist()}"
        )
        raise ValueError(msg)
    if "theta_source" in posterior and "source_ids" in posterior:
        posterior_source_ids = {str(source_id) for source_id in posterior["source_ids"]}
        matrix_sources = matrix.get_column("source").cast(pl.String).to_list()
        missing_sources = sorted({source for source in matrix_sources if source not in posterior_source_ids})
        if missing_sources:
            msg = f"Matrix sources are missing from posterior source_ids. missing_sources={missing_sources}"
            raise ValueError(msg)
    elif "theta_source" in posterior:
        raise ValueError("Posterior archive contains theta_source but is missing source_ids required for PPC.")


def posterior_predictive_judge_accuracy(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate posterior predictive judge accuracy intervals."""

    validate_posterior_item_alignment(matrix, posterior)
    b = flatten_draws(posterior["b"])
    a = flatten_draws(posterior["a"]) if "a" in posterior else np.ones_like(b)
    if "theta_source" in posterior and "source_ids" in posterior:
        item_ids = [str(item_id) for item_id in posterior["item_ids"].tolist()]
        item_sources = matrix.select(["item_id", "source"]).unique(subset=["item_id"], keep="first")
        source_lookup = {str(source_id): index for index, source_id in enumerate(posterior["source_ids"])}
        source_by_item_id = {str(row["item_id"]): source_lookup[str(row["source"])] for row in item_sources.to_dicts()}
        item_source_idx = np.asarray(
            [source_by_item_id[item_id] for item_id in item_ids],
            dtype=int,
        )
        theta_source = posterior["theta_source"].reshape(
            -1,
            posterior["theta_source"].shape[-2],
            posterior["theta_source"].shape[-1],
        )
        theta_by_item = np.take(theta_source, item_source_idx, axis=2)
        logits = a[:, None, :] * (theta_by_item - b[:, None, :])
    else:
        theta = flatten_draws(posterior["theta"])
        logits = a[:, None, :] * (theta[:, :, None] - b[:, None, :])
    probabilities = stable_sigmoid(logits)
    predictive_array = probabilities.mean(axis=2)
    return (
        predictive_array.mean(axis=0),
        np.quantile(predictive_array, 0.05, axis=0),
        np.quantile(predictive_array, 0.95, axis=0),
    )


def plot_judge_reliability_by_source(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot source-specific judge reliability means as an annotated heatmap."""

    if "theta_source" not in posterior or "source_ids" not in posterior:
        raise ValueError("Posterior does not contain source-aware reliability samples.")
    ordered_sources = top_source_ids(matrix, posterior)
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
    posterior: dict[str, np.ndarray] | None = None
    if posterior_path.exists():
        try:
            posterior = load_posterior(posterior_path)
        except ValueError as exc:
            logger.warning(
                "posterior archive at %s failed validation (%s); treating as missing",
                posterior_path,
                exc,
            )
    save_figure(
        plot_prior_predictive_probabilities(matrix, config, posterior),
        figure_base_path(figures_dir, PRIOR_PREDICTIVE_STEM),
    )
    if posterior is None:
        cleanup_posterior_figure_outputs(figures_dir, keep_ridge=False, keep_source=False)
        logger.warning(
            "posterior not found at %s; saved prior predictive figure only to %s",
            posterior_path,
            figures_dir,
        )
        return
    backend = str(posterior.get("backend", "unknown"))
    logger.info(
        "loaded posterior from %s using backend=%s",
        posterior_path,
        backend,
    )
    try:
        validate_posterior_plot_inputs(matrix, posterior)
    except ValueError:
        cleanup_posterior_figure_outputs(figures_dir, keep_ridge=False, keep_source=False)
        raise
    save_figure(
        plot_judge_reliability_ridge(matrix, posterior),
        figure_base_path(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM),
    )
    if has_source_reliability(posterior):
        save_figure(
            plot_judge_reliability_by_source(matrix, posterior),
            figure_base_path(figures_dir, JUDGE_RELIABILITY_BY_SOURCE_STEM),
        )
    else:
        cleanup_posterior_figure_outputs(figures_dir, keep_ridge=True, keep_source=False)
    logger.info("saved posterior figures to %s", figures_dir)


if __name__ == "__main__":
    main()
