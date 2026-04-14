"""Posterior plots and simple posterior predictive checks."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.analysis.diagnostics import flatten_draws, load_posterior
from src.analysis.figure_paths import (
    ITEM_PARAMETER_SCATTER_STEM,
    JUDGE_RELIABILITY_BY_SOURCE_STEM,
    JUDGE_RELIABILITY_RIDGE_STEM,
    POSTERIOR_PREDICTIVE_STEM,
    PRIOR_PREDICTIVE_STEM,
    figure_base_path,
    figure_png_path,
)
from src.data.loader import ITEM_METADATA_COLUMNS
from src.logging_utils import configure_logging
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)

JUDGE_COLOR_PINS = {
    "deepseek-r1-distill-qwen-14b": "#0f4c81",
    "deepseek-r1-distill-qwen-7b": "#c44e52",
    "mistral-7b-instruct-v0-3": "#d95f02",
    "qwen2-5-7b-instruct": "#4daf4a",
    "gemma-2-9b-it": "#984ea3",
}
JUDGE_LABEL_PINS = {
    "deepseek-r1-distill-qwen-14b": "DeepSeek 14B",
    "deepseek-r1-distill-qwen-7b": "DeepSeek 7B",
    "mistral-7b-instruct-v0-3": "Mistral 7B",
    "qwen2-5-7b-instruct": "Qwen 7B",
    "gemma-2-9b-it": "Gemma 9B",
}
SOURCE_COLOR_PINS = {
    "livebench-reasoning": "#1f78b4",
    "livebench-math": "#33a02c",
    "livecodebench": "#e31a1c",
    "mmlu-pro-computer science": "#ff7f00",
    "mmlu-pro-math": "#6a3d9a",
}


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
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    """Remove unused frame lines and keep a cleaner plotting surface."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def judge_color_map(judge_ids: np.ndarray) -> dict[str, str]:
    """Return stable colors for each judge ID."""

    return {
        judge_id: JUDGE_COLOR_PINS.get(judge_id, _fallback_judge_color(judge_id))
        for judge_id in map(str, judge_ids)
    }


def source_color_map(source_ids: list[str]) -> dict[str, str]:
    """Return stable colors for each source ID."""

    return {
        source_id: SOURCE_COLOR_PINS.get(source_id, _fallback_judge_color(source_id))
        for source_id in source_ids
    }


def _fallback_judge_color(judge_id: str) -> str:
    """Generate a deterministic fallback color for judges without pinned colors."""

    digest = hashlib.sha256(judge_id.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], byteorder="big") / 65535.0
    saturation = 0.55 + (digest[2] / 255.0) * 0.15
    value = 0.65 + (digest[3] / 255.0) * 0.2
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


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
    theta = rng.normal(priors.theta.loc, priors.theta.scale, size=(num_draws, n_judges))
    b = rng.normal(priors.b.loc, priors.b.scale, size=(num_draws, n_items))
    if config.model.type == "2PL":
        a = rng.lognormal(priors.a.loc, priors.a.scale, size=(num_draws, n_items))
    else:
        a = np.ones((num_draws, n_items))
    if config.model.variant == "source_hier":
        if priors.tau_theta is None:
            raise ValueError("source_hier prior predictive simulation requires tau_theta")
        source_ids = (
            matrix.get_column("source").cast(pl.String).unique(maintain_order=True).to_list()
        )
        source_lookup = {source_id: index for index, source_id in enumerate(source_ids)}
        item_source_idx = np.asarray(
            [source_lookup[source_id] for source_id in matrix.get_column("source").cast(pl.String)],
            dtype=int,
        )
        tau_theta = rng.lognormal(
            priors.tau_theta.loc,
            priors.tau_theta.scale,
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
    *,
    num_draws: int = 1000,
) -> plt.Figure:
    """Plot compact prior predictive summaries for probability scale calibration."""

    probabilities, judge_means = sample_prior_predictive_probabilities(
        matrix,
        config,
        num_draws=num_draws,
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    probability_axis, mean_axis = axes
    probability_axis.hist(probabilities, bins=40, density=True, color="#5b6670", alpha=0.85)
    probability_axis.axvline(0.05, color="#9aa0a6", linestyle="--", linewidth=1.0)
    probability_axis.axvline(0.95, color="#9aa0a6", linestyle="--", linewidth=1.0)
    probability_axis.set_xlabel("Prior predictive P(y=1)")
    probability_axis.set_ylabel("Density")
    probability_axis.set_title("Item-level probabilities")
    style_axis(probability_axis)
    mean_axis.hist(judge_means, bins=30, density=True, color="#0f4c81", alpha=0.85)
    mean_axis.axvline(0.5, color="#9aa0a6", linestyle="--", linewidth=1.0)
    mean_axis.set_xlabel("Prior predictive judge mean accuracy")
    mean_axis.set_ylabel("Density")
    mean_axis.set_title("Judge-level averages")
    style_axis(mean_axis)
    fig.suptitle("Prior Predictive Calibration", y=1.02)
    fig.tight_layout()
    return fig


def plot_judge_reliability_ridge(posterior: dict[str, np.ndarray]) -> plt.Figure:
    """Plot stacked posterior densities for judge reliability."""

    theta_samples = flatten_draws(posterior["theta"])
    judge_ids = posterior["judge_ids"]
    color_map = judge_color_map(judge_ids)
    ordering = np.argsort(theta_samples.mean(axis=0))
    fig, ax = plt.subplots(figsize=(8, 8))
    ridge_handles: list[Patch] = []
    ordered_judge_ids: list[str] = []
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
        ax.fill_between(
            centers,
            baseline,
            baseline + density,
            alpha=0.7,
            color=color_map[judge_id],
        )
        ordered_judge_ids.append(judge_id)
        ridge_handles.append(Patch(facecolor=color_map[judge_id], edgecolor="none", label=judge_id))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Posterior reliability (theta)")
    style_axis(ax)
    ridge_handles = [
        Patch(
            facecolor=color_map[judge_id],
            edgecolor="none",
            label=JUDGE_LABEL_PINS.get(judge_id, judge_id),
        )
        for judge_id in ordered_judge_ids
    ]
    if ridge_handles:
        fig.legend(
            handles=ridge_handles,
            title="Judges",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(ridge_handles),
            frameon=False,
            fontsize=9,
            title_fontsize=9,
            handlelength=1.2,
            columnspacing=1.0,
        )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return fig


def plot_item_parameter_scatter(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Scatter plot of item difficulty versus discrimination."""

    b_mean = flatten_draws(posterior["b"]).mean(axis=0)
    a_mean = (
        flatten_draws(posterior["a"]).mean(axis=0) if "a" in posterior else np.ones_like(b_mean)
    )
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(b_mean, a_mean, alpha=0.8, color="#5b6670")
    ax.set_xlabel("Item difficulty mean (b)")
    ax.set_ylabel("Item discrimination mean (a)")
    ax.set_title("Item Parameter Summary")
    style_axis(ax)
    fig.tight_layout()
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


def _source_facet_grid_shape(source_count: int, max_columns: int = 2) -> tuple[int, int]:
    """Return subplot grid dimensions for source small multiples."""

    if source_count <= 0:
        raise ValueError("Source facet plotting requires at least one source.")
    columns = min(max_columns, source_count)
    rows = int(np.ceil(source_count / columns))
    return rows, columns


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
        missing_sources = sorted(
            {source for source in matrix_sources if source not in posterior_source_ids}
        )
        if missing_sources:
            msg = (
                "Matrix sources are missing from posterior source_ids. "
                f"missing_sources={missing_sources}"
            )
            raise ValueError(msg)
    elif "theta_source" in posterior:
        raise ValueError(
            "Posterior archive contains theta_source but is missing source_ids required for PPC."
        )


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
        source_lookup = {
            str(source_id): index for index, source_id in enumerate(posterior["source_ids"])
        }
        source_by_item_id = {
            str(row["item_id"]): source_lookup[str(row["source"])]
            for row in item_sources.to_dicts()
        }
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


def plot_posterior_predictive_check(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Compare observed judge accuracy to posterior predictive intervals."""

    judge_ids, observed = observed_accuracy(matrix)
    validate_posterior_judge_order(judge_ids, posterior)
    predicted_mean, lower, upper = posterior_predictive_judge_accuracy(matrix, posterior)
    color_map = judge_color_map(judge_ids)
    fig, ax = plt.subplots(figsize=(8, 8))
    y_positions = np.arange(len(judge_ids))
    ordered_judge_ids = [str(judge_id) for judge_id in judge_ids]
    for index, judge_id in enumerate(ordered_judge_ids):
        color = color_map[judge_id]
        ax.errorbar(
            predicted_mean[index],
            y_positions[index],
            xerr=np.asarray(
                [
                    [predicted_mean[index] - lower[index]],
                    [upper[index] - predicted_mean[index]],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
        )
        ax.scatter(observed[index], y_positions[index], marker="s", color=color)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("")
    ax.text(
        0.02,
        0.98,
        "circle = posterior predictive\nsquare = observed",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3.0},
    )
    style_axis(ax)
    model_handles = [
        Patch(
            facecolor=color_map[judge_id],
            edgecolor="none",
            label=JUDGE_LABEL_PINS.get(judge_id, judge_id),
        )
        for judge_id in ordered_judge_ids
    ]
    fig.legend(
        handles=model_handles,
        title="Judges",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(model_handles),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    return fig


def plot_judge_reliability_by_source(
    matrix: pl.DataFrame,
    posterior: dict[str, np.ndarray],
) -> plt.Figure:
    """Plot source-specific judge reliability intervals as synchronized small multiples."""

    if "theta_source" not in posterior or "source_ids" not in posterior:
        raise ValueError("Posterior does not contain source-aware reliability samples.")
    ordered_sources = top_source_ids(matrix, posterior)
    if not ordered_sources:
        raise ValueError("Source facet plotting requires at least one source.")
    summary = source_reliability_summary(posterior, ordered_sources)
    judge_ids = np.asarray(posterior["judge_ids"], dtype=str)
    global_theta = flatten_draws(posterior["theta"]).mean(axis=0)
    global_ordering = np.argsort(global_theta)[::-1]
    ordered_judge_ids = [str(judge_ids[index]) for index in global_ordering]
    color_map = judge_color_map(judge_ids)
    global_means = {
        str(judge_id): float(global_theta[index]) for index, judge_id in enumerate(judge_ids)
    }
    source_counts = matrix.group_by("source").len().rename({"len": "item_count"}).to_dicts()
    counts_map = {str(row["source"]): int(row["item_count"]) for row in source_counts}
    theta_min = float(summary.get_column("theta_p05").min())
    theta_max = float(summary.get_column("theta_p95").max())
    x_padding = max(0.1, 0.08 * (theta_max - theta_min))
    y_positions = np.arange(len(ordered_judge_ids), dtype=float)
    rows, columns = _source_facet_grid_shape(len(ordered_sources))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 3.5 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_array = np.asarray(axes).reshape(-1)
    for index, (axis, source_id) in enumerate(zip(axes_array, ordered_sources, strict=False)):
        source_rows = summary.filter(pl.col("source") == source_id)
        for y_position, judge_id in zip(y_positions, ordered_judge_ids, strict=False):
            row = source_rows.filter(pl.col("judge_id") == judge_id)
            mean = float(row.get_column("theta_mean").item())
            lower = float(row.get_column("theta_p05").item())
            upper = float(row.get_column("theta_p95").item())
            color = color_map[judge_id]
            axis.errorbar(
                mean,
                y_position,
                xerr=np.asarray([[mean - lower], [upper - mean]]),
                fmt="o",
                color=color,
                ecolor=color,
                elinewidth=1.3,
                capsize=2.5,
                markersize=4.5,
                zorder=3,
            )
            axis.scatter(
                global_means[judge_id],
                y_position,
                marker="|",
                color="#70757a",
                s=120,
                linewidths=1.1,
                zorder=2,
            )
        axis.axvline(0.0, color="#9aa0a6", linewidth=1.0, linestyle="--", alpha=0.9, zorder=1)
        axis.set_title(f"{source_id} (n={counts_map.get(source_id, 0)})", fontsize=10)
        axis.set_xlim(theta_min - x_padding, theta_max + x_padding)
        axis.set_ylim(len(ordered_judge_ids) - 0.5, -0.5)
        row_index, column_index = divmod(index, columns)
        if column_index == 0:
            axis.set_yticks(y_positions)
            axis.set_yticklabels(
                [JUDGE_LABEL_PINS.get(judge_id, judge_id) for judge_id in ordered_judge_ids],
                fontsize=9,
            )
        else:
            axis.tick_params(axis="y", labelleft=False)
        if row_index == rows - 1:
            axis.set_xlabel("Posterior reliability (theta)")
        style_axis(axis)
    for axis in axes_array[len(ordered_sources) :]:
        axis.set_visible(False)
    model_handles = [
        Patch(
            facecolor=color_map[judge_id],
            edgecolor="none",
            label=JUDGE_LABEL_PINS.get(judge_id, judge_id),
        )
        for judge_id in ordered_judge_ids
    ]
    model_handles.append(
        Line2D([], [], color="#70757a", marker="|", linestyle="None", label="global mean")
    )
    fig.legend(
        handles=model_handles,
        title="Judges",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(model_handles), 3),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
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
        logger.warning(
            "posterior not found at %s; saved prior predictive figure only to %s",
            posterior_path,
            figures_dir,
        )
        return
    posterior = load_posterior(posterior_path)
    backend = str(posterior.get("backend", "unknown"))
    logger.info(
        "loaded posterior from %s using backend=%s",
        posterior_path,
        backend,
    )
    ridge_figure = plot_judge_reliability_ridge(posterior)
    ridge_figure.savefig(
        figure_png_path(figures_dir, JUDGE_RELIABILITY_RIDGE_STEM),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(ridge_figure)
    save_figure(
        plot_item_parameter_scatter(matrix, posterior),
        figure_base_path(figures_dir, ITEM_PARAMETER_SCATTER_STEM),
    )
    save_figure(
        plot_posterior_predictive_check(matrix, posterior),
        figure_base_path(figures_dir, POSTERIOR_PREDICTIVE_STEM),
    )
    if "theta_source" in posterior and "source_ids" in posterior:
        save_figure(
            plot_judge_reliability_by_source(matrix, posterior),
            figure_base_path(figures_dir, JUDGE_RELIABILITY_BY_SOURCE_STEM),
        )
    logger.info("saved posterior figures to %s", figures_dir)


if __name__ == "__main__":
    main()
