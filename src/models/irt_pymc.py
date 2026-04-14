"""Fit 1PL or 2PL IRT models with PyMC NUTS."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pymc as pm

from src.logging_utils import configure_logging, format_table_for_log
from src.models.irt_common import (
    build_model_priors,
    load_matrix_observations,
    save_posterior,
    summarize_item_parameters,
    summarize_judges,
)
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PyMC inference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def _build_model(config: ExperimentConfig, observations: dict[str, Any]) -> pm.Model:
    """Build a PyMC IRT model matching the configured type and variant."""

    priors = build_model_priors(config.model)
    if config.model.variant == "source_hier" and priors.tau_theta is None:
        raise ValueError("source_hier variant requires model.priors.tau_theta in config")
    coords = {
        "judge": observations["judge_ids"].tolist(),
        "item": observations["item_ids"].tolist(),
        "source": observations["source_ids"].tolist(),
        "obs": np.arange(observations["correct"].shape[0]),
    }
    with pm.Model(coords=coords) as model:
        judge_idx = pm.Data("judge_idx", observations["judge_idx"], dims="obs")
        item_idx = pm.Data("item_idx", observations["item_idx"], dims="obs")
        source_idx = pm.Data("source_idx", observations["source_idx"], dims="obs")
        theta = pm.Normal("theta", mu=priors.theta.loc, sigma=priors.theta.scale, dims="judge")
        b = pm.Normal("b", mu=priors.b.loc, sigma=priors.b.scale, dims="item")
        if config.model.variant == "source_hier":
            tau_theta = pm.LogNormal(
                "tau_theta",
                mu=priors.tau_theta.loc,
                sigma=priors.tau_theta.scale,
                dims="judge",
            )
            theta_source = pm.Normal(
                "theta_source",
                mu=theta[:, None],
                sigma=tau_theta[:, None],
                dims=("judge", "source"),
            )
            judge_term = theta_source[judge_idx, source_idx]
        else:
            judge_term = theta[judge_idx]
        if config.model.type == "2PL":
            a = pm.LogNormal("a", mu=priors.a.loc, sigma=priors.a.scale, dims="item")
            logits = a[item_idx] * (judge_term - b[item_idx])
        else:
            logits = judge_term - b[item_idx]
        pm.Bernoulli(
            "correct",
            logit_p=logits,
            observed=observations["correct"],
            dims="obs",
        )
    return model


def _extract_samples(idata: Any, config: ExperimentConfig) -> dict[str, np.ndarray]:
    """Convert an InferenceData posterior into the archive schema used downstream."""

    posterior = idata.posterior
    samples = {
        "theta": posterior["theta"].transpose("chain", "draw", "judge").values,
        "b": posterior["b"].transpose("chain", "draw", "item").values,
        "diverging": idata.sample_stats["diverging"].transpose("chain", "draw").values,
    }
    if config.model.type == "2PL":
        samples["a"] = posterior["a"].transpose("chain", "draw", "item").values
    if config.model.variant == "source_hier":
        samples["tau_theta"] = posterior["tau_theta"].transpose("chain", "draw", "judge").values
        samples["theta_source"] = (
            posterior["theta_source"].transpose("chain", "draw", "judge", "source").values
        )
    return {name: np.asarray(values) for name, values in samples.items()}


def run_mcmc(
    config: ExperimentConfig,
    observations: dict[str, Any],
) -> tuple[Any, dict[str, np.ndarray]]:
    """Run PyMC NUTS for the configured model."""

    model = _build_model(config, observations)
    with model:
        idata = pm.sample(
            tune=config.inference.num_warmup,
            draws=config.inference.num_samples,
            chains=config.inference.num_chains,
            cores=1,
            target_accept=config.inference.target_accept_prob,
            random_seed=config.experiment.seed,
            discard_tuned_samples=True,
            return_inferencedata=True,
            progressbar=logger.isEnabledFor(logging.INFO),
        )
    return idata, _extract_samples(idata, config)


def run_and_save_posterior(config: ExperimentConfig, matrix: pl.DataFrame | None = None) -> None:
    """Run PyMC inference and persist posterior samples."""

    config.ensure_directories()
    observations = load_matrix_observations(
        matrix if matrix is not None else config.data.matrix_path
    )
    _, samples = run_mcmc(config, observations)
    output_path = config.inference.posterior_path
    save_posterior(
        output_path,
        samples,
        observations,
        config.model.type,
        metadata={
            "backend": np.asarray("pymc"),
            "experiment_seed": np.asarray(config.experiment.seed),
            "num_chains": np.asarray(samples["theta"].shape[0]),
        },
    )
    summary = summarize_judges(samples, observations["judge_ids"])
    item_summary = summarize_item_parameters(samples)
    logger.info("saved_posterior=%s", output_path)
    if logger.isEnabledFor(logging.INFO):
        logger.info("judge summary\n%s", format_table_for_log(summary))
        logger.info("item parameter summary\n%s", format_table_for_log(item_summary))


def main() -> None:
    """CLI entrypoint for PyMC IRT inference."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    run_and_save_posterior(config)


if __name__ == "__main__":
    main()
