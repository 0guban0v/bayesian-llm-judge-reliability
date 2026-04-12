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
    logits = a[:, None, :] * (theta[:, :, None] - b[:, None, :])
    probabilities = 1.0 / (1.0 + np.exp(-logits))
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
    predicted_mean, lower, upper = posterior_predictive_judge_accuracy(posterior)
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
    ax.set_title("Posterior Predictive Accuracy by Judge")
    style_axis(ax)
    model_handles = [
        Patch(
            facecolor=color_map[judge_id],
            edgecolor="none",
            label=JUDGE_LABEL_PINS.get(judge_id, judge_id),
        )
        for judge_id in ordered_judge_ids
    ]
    style_legend = [
        Line2D([], [], color="black", marker="o", linestyle="None", label="posterior predictive"),
        Line2D([], [], color="black", marker="s", linestyle="None", label="observed"),
    ]
    fig.legend(
        handles=model_handles,
        title="Judges",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(model_handles),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.legend(
        handles=style_legend,
        title="Marks",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(style_legend),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    return fig


def main() -> None:
    """CLI entrypoint for posterior plots."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    posterior_path = config.inference.posterior_path
    posterior = load_posterior(posterior_path)
    matrix = pl.read_parquet(config.data.matrix_path)
    figures_dir = config.figures_dir
    logger.info(
        "loaded posterior from %s using NumPyro",
        posterior_path,
    )
    ridge_figure = plot_judge_reliability_ridge(posterior)
    figures_dir.mkdir(parents=True, exist_ok=True)
    ridge_figure.savefig(figures_dir / "judge_reliability_ridge.png", dpi=300, bbox_inches="tight")
    plt.close(ridge_figure)
    save_figure(
        plot_item_parameter_scatter(matrix, posterior),
        figures_dir / "item_parameter_scatter",
    )
    save_figure(
        plot_posterior_predictive_check(matrix, posterior),
        figures_dir / "posterior_predictive",
    )
    logger.info("saved posterior figures to %s", figures_dir)


if __name__ == "__main__":
    main()
