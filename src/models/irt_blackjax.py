"""Fit the 2PL IRT model with a manual BlackJAX NUTS loop."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from src.logging_utils import configure_logging
from src.models.irt_numpyro import (
    ModelPriors,
    build_model_priors,
    load_matrix_observations,
    save_posterior,
)
from src.schemas import ExperimentConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BlackJAX inference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def make_logdensity_fn(
    correct: jnp.ndarray,
    judge_idx: jnp.ndarray,
    item_idx: jnp.ndarray,
    n_judges: int,
    n_items: int,
    priors: ModelPriors,
) -> Any:
    """Create the log posterior for the unconstrained 2PL parameterization."""

    def logdensity(position: dict[str, jnp.ndarray]) -> jnp.ndarray:
        theta = position["theta"]
        b = position["b"]
        log_a = position["log_a"]
        a = jnp.exp(log_a)
        logits = a[item_idx] * (theta[judge_idx] - b[item_idx])
        log_prior_theta = jnp.sum(
            jax.scipy.stats.norm.logpdf(theta, priors.theta.loc, priors.theta.scale)
        )
        log_prior_b = jnp.sum(jax.scipy.stats.norm.logpdf(b, priors.b.loc, priors.b.scale))
        log_prior_log_a = jnp.sum(jax.scipy.stats.norm.logpdf(log_a, priors.a.loc, priors.a.scale))
        log_likelihood = jnp.sum(
            correct * jax.nn.log_sigmoid(logits) + (1.0 - correct) * jax.nn.log_sigmoid(-logits)
        )
        return log_prior_theta + log_prior_b + log_prior_log_a + log_likelihood

    return logdensity


def initial_position(key: jax.Array, n_judges: int, n_items: int) -> dict[str, jnp.ndarray]:
    """Generate a small random initialization for BlackJAX."""

    theta_key, b_key, a_key = jax.random.split(key, 3)
    return {
        "theta": 0.1 * jax.random.normal(theta_key, (n_judges,)),
        "b": 0.1 * jax.random.normal(b_key, (n_items,)),
        "log_a": 0.1 * jax.random.normal(a_key, (n_items,)),
    }


def inference_loop(
    rng_key: jax.Array,
    kernel: Any,
    state: Any,
    num_samples: int,
) -> tuple[Any, Any]:
    """Run a BlackJAX transition kernel with `jax.lax.scan`."""

    keys = jax.random.split(rng_key, num_samples)

    def one_step(carry: Any, step_key: jax.Array) -> tuple[Any, tuple[Any, Any]]:
        new_state, info = kernel.step(step_key, carry)
        return new_state, (new_state, info)

    return jax.lax.scan(one_step, state, keys)


def run_chain(
    chain_key: jax.Array,
    config: ExperimentConfig,
    observations: dict[str, Any],
    priors: ModelPriors,
) -> dict[str, np.ndarray]:
    """Run warmup and sampling for one independent BlackJAX chain."""

    init_key, warmup_key, sample_key = jax.random.split(chain_key, 3)
    logdensity_fn = make_logdensity_fn(
        correct=jnp.asarray(observations["correct"], dtype=jnp.float32),
        judge_idx=jnp.asarray(observations["judge_idx"], dtype=jnp.int32),
        item_idx=jnp.asarray(observations["item_idx"], dtype=jnp.int32),
        n_judges=observations["n_judges"],
        n_items=observations["n_items"],
        priors=priors,
    )
    start = initial_position(init_key, observations["n_judges"], observations["n_items"])
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        target_acceptance_rate=config.inference.target_accept_prob,
    )
    (state, parameters), _ = warmup.run(
        warmup_key,
        start,
        num_steps=config.inference.num_warmup,
    )
    kernel = blackjax.nuts(logdensity_fn, **parameters)
    _, (states, infos) = inference_loop(sample_key, kernel, state, config.inference.num_samples)
    position = jax.tree.map(np.asarray, states.position)
    return {
        "theta": position["theta"],
        "b": position["b"],
        "a": np.exp(position["log_a"]),
        "diverging": np.asarray(infos.is_divergent),
    }


def run_blackjax(config: ExperimentConfig, observations: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run BlackJAX NUTS and return posterior samples with a chain axis."""

    rng_key = jax.random.PRNGKey(config.experiment.seed)
    priors = build_model_priors(config.model)
    chain_keys = jax.random.split(rng_key, config.inference.num_chains)
    chain_samples = []
    for chain_index, chain_key in enumerate(chain_keys, start=1):
        logger.info("running blackjax chain %s/%s", chain_index, config.inference.num_chains)
        chain_samples.append(run_chain(chain_key, config, observations, priors))
    return {
        "theta": np.stack([chain["theta"] for chain in chain_samples], axis=0),
        "b": np.stack([chain["b"] for chain in chain_samples], axis=0),
        "a": np.stack([chain["a"] for chain in chain_samples], axis=0),
        "diverging": np.stack([chain["diverging"] for chain in chain_samples], axis=0),
    }


def main() -> None:
    """CLI entrypoint for BlackJAX inference."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    config.ensure_directories()
    observations = load_matrix_observations(config.data.matrix_path)
    samples = run_blackjax(config, observations)
    output_path = config.inference.posterior_path_for_backend("blackjax")
    save_posterior(
        output_path,
        samples,
        observations,
        "2PL",
        metadata={
            "backend": np.asarray("blackjax"),
            "experiment_seed": np.asarray(config.experiment.seed),
            "num_chains": np.asarray(config.inference.num_chains),
        },
    )
    logger.info("saved_posterior=%s", output_path)


if __name__ == "__main__":
    main()
