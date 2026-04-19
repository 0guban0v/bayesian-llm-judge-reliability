"""Fit 1PL or 2PL IRT models with PyMC NUTS."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pymc as pm

from src.models.irt_common import aggregate_judge_accuracy_ppc, build_model_priors
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


def _sample_configured_prior(
    name: str,
    prior,
    *,
    dims: str | tuple[str, ...],
) -> pm.TensorVariable:
    """Sample a configured scalar or vector prior using the declared distribution family."""

    if prior.dist == "normal":
        return pm.Normal(name, mu=prior.loc, sigma=prior.scale, dims=dims)
    if prior.dist == "lognormal":
        return pm.LogNormal(name, mu=prior.loc, sigma=prior.scale, dims=dims)
    raise ValueError(f"Unsupported prior distribution '{prior.dist}' for {name}")


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
        theta = _sample_configured_prior("theta", priors.theta, dims="judge")
        b = _sample_configured_prior("b", priors.b, dims="item")
        if config.model.variant == "source_hier":
            tau_theta = _sample_configured_prior("tau_theta", priors.tau_theta, dims="judge")
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
            a = _sample_configured_prior("a", priors.a, dims="item")
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
        samples["theta_source"] = posterior["theta_source"].transpose("chain", "draw", "judge", "source").values
    return {name: np.asarray(values) for name, values in samples.items()}


def run_mcmc(
    config: ExperimentConfig,
    observations: dict[str, Any],
) -> tuple[Any, dict[str, np.ndarray], dict[str, np.ndarray]]:
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
        if config.inference.save_log_likelihood:
            pm.compute_log_likelihood(
                idata,
                var_names=["correct"],
                extend_inferencedata=True,
                progressbar=False,
            )
        ppc_idata = pm.sample_posterior_predictive(
            idata,
            var_names=["correct"],
            return_inferencedata=True,
            progressbar=False,
        )
        idata.extend(ppc_idata)
    samples = _extract_samples(idata, config)
    posterior_predictive_correct = np.asarray(
        idata.posterior_predictive["correct"].transpose("chain", "draw", "obs").values
    )
    ppc_summary = aggregate_judge_accuracy_ppc(posterior_predictive_correct, observations)
    return idata, samples, ppc_summary
