"""Diagnostics summaries and compact diagnostic figures for saved posterior samples."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter

from src.analysis.plot_config import EXPORT_DPI, style_axis
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


def diagnostic_group_rows(posterior: dict[str, np.ndarray]) -> list[dict[str, object]]:
    """Return grouped diagnostic arrays for compact plotting."""

    parameter_order = ["theta", "b", "a", "tau_theta", "theta_source"]
    rows: list[dict[str, object]] = []
    for parameter_name in parameter_order:
        if parameter_name not in posterior:
            continue
        values = posterior[parameter_name]
        feature_count = int(_flatten_parameter(values).shape[-1])
        if parameter_name == "theta":
            label = f"θ\n({feature_count} judges)"
        elif parameter_name == "b":
            label = f"b\n({feature_count} items)"
        elif parameter_name == "a":
            label = f"a\n({feature_count} items)"
        elif parameter_name == "tau_theta":
            label = f"τ_θ\n({feature_count} judges)"
        else:
            label = f"θ_source\n({feature_count} source effects)"
        rows.append(
            {
                "parameter": parameter_name,
                "label": label,
                "rhat": compute_rhat(values),
                "ess": compute_ess(values),
            }
        )
    return rows


def _stack_offsets(count: int, width: float = 0.28) -> np.ndarray:
    """Return deterministic y offsets for stacked dot plots."""

    if count <= 1:
        return np.asarray([0.0])
    return np.linspace(-width, width, count)


def _trim_tick_zeros(value: float, _position: float) -> str:
    """Format tick labels without redundant trailing zeros."""

    return f"{value:g}"


def _diagnostic_ticks(limit: float, include: float | None = None) -> list[float]:
    """Build sparse axis ticks while ensuring an optional threshold is shown."""

    if limit <= 0:
        return [0.0]
    ticks: set[float] = {0.0, float(limit)}
    if include is not None and 0.0 <= include <= limit:
        ticks.add(float(include))
    if limit >= 1000:
        ticks.add(round(limit / 2.0, -2))
    return sorted(ticks)


def plot_diagnostics_summary(posterior: dict[str, np.ndarray]) -> plt.Figure:
    """Plot compact grouped R-hat and ESS diagnostics."""

    groups = diagnostic_group_rows(posterior)
    if not groups:
        raise ValueError("Posterior archive does not contain plottable diagnostic parameters.")
    divergence_count = int(np.asarray(posterior.get("diverging", np.array([]))).sum())
    fig, (rhat_axis, ess_axis) = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.8),
        gridspec_kw={"width_ratios": [1.05, 1.15], "wspace": 0.12},
        sharey=True,
    )
    group_positions = np.arange(len(groups))[::-1]
    rhat_axis.axvline(1.0, color="#6b7280", linewidth=1.0, linestyle="--")
    rhat_axis.axvline(1.01, color="#6b7280", linewidth=1.0, linestyle="--")
    ess_axis.axvline(400.0, color="#6b7280", linewidth=1.0, linestyle="--")
    for y_position, group in zip(group_positions, groups, strict=False):
        rhat_values = np.asarray(group["rhat"], dtype=float)
        ess_values = np.asarray(group["ess"], dtype=float)
        offsets = _stack_offsets(len(rhat_values))
        rhat_axis.scatter(
            rhat_values,
            y_position + offsets,
            s=14,
            color="#0f4c81",
            alpha=0.8,
            linewidths=0.0,
        )
        ess_axis.scatter(
            ess_values,
            y_position + offsets,
            s=14,
            color="#0f4c81",
            alpha=0.8,
            linewidths=0.0,
        )
    labels = [str(group["label"]) for group in groups]
    rhat_axis.set_yticks(group_positions)
    rhat_axis.set_yticklabels(labels, fontsize=9)
    rhat_axis.set_xlabel("R̂")
    ess_axis.set_xlabel("ESS")
    rhat_max = max(
        1.012,
        float(np.nanmax([np.nanmax(group["rhat"]) for group in groups])) + 0.001,
    )
    rhat_axis.set_xlim(0.998, rhat_max)
    ess_max = max(float(np.nanmax(group["ess"])) for group in groups)
    ess_limit = ess_max * 1.06
    ess_axis.set_xlim(0.0, ess_limit)
    ess_axis.set_xticks(_diagnostic_ticks(ess_limit, include=400.0))
    formatter = FuncFormatter(_trim_tick_zeros)
    rhat_axis.xaxis.set_major_formatter(formatter)
    ess_axis.xaxis.set_major_formatter(formatter)
    ess_axis.text(
        0.98,
        0.98,
        f"{divergence_count} divergences",
        transform=ess_axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.5},
    )
    style_axis(rhat_axis)
    style_axis(ess_axis)
    ess_axis.set_yticks([])
    rhat_axis.set_yticks(group_positions)
    rhat_axis.set_yticklabels(labels, fontsize=9)
    ess_axis.spines["left"].set_visible(False)
    ess_axis.tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.15, top=0.96, wspace=0.12)
    return fig


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    """Save a matplotlib figure as PNG."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entrypoint for diagnostic summaries."""

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
    save_figure(
        plot_diagnostics_summary(posterior),
        config.figures_dir / "diagnostics_summary",
    )


if __name__ == "__main__":
    main()
