"""Fit 1PL or 2PL IRT models with NumPyro NUTS."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
import polars as pl
from numpyro.infer import MCMC, NUTS

from src.data.loader import ITEM_METADATA_COLUMNS
from src.logging_utils import configure_logging, format_table_for_log
from src.schemas import ExperimentConfig, IRTConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriorSpec:
    """Concrete prior parameters for a single latent variable."""

    loc: float
    scale: float


@dataclass(frozen=True)
class ModelPriors:
    """Fully bound prior parameters for the configured IRT model."""

    theta: PriorSpec
    b: PriorSpec
    a: PriorSpec


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for NumPyro inference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def build_model_priors(model_config: IRTConfig) -> ModelPriors:
    """Return immutable prior parameters derived from the experiment config."""

    return ModelPriors(
        theta=PriorSpec(
            loc=model_config.priors.theta.loc,
            scale=model_config.priors.theta.scale,
        ),
        b=PriorSpec(
            loc=model_config.priors.b.loc,
            scale=model_config.priors.b.scale,
        ),
        a=PriorSpec(
            loc=model_config.priors.a.loc,
            scale=model_config.priors.a.scale,
        ),
    )


def irt_1pl(
    correct=None,
    judge_idx=None,
    item_idx=None,
    n_judges=None,
    n_items=None,
    *,
    priors: ModelPriors,
):
    """One-parameter logistic IRT model."""

    theta = numpyro.sample(
        "theta",
        dist.Normal(
            priors.theta.loc,
            priors.theta.scale,
        ).expand([n_judges]),
    )
    b = numpyro.sample(
        "b",
        dist.Normal(priors.b.loc, priors.b.scale).expand([n_items]),
    )
    logits = theta[judge_idx] - b[item_idx]
    with numpyro.plate("obs", len(correct)):
        numpyro.sample("correct", dist.Bernoulli(logits=logits), obs=correct)


def irt_2pl(
    correct=None,
    judge_idx=None,
    item_idx=None,
    n_judges=None,
    n_items=None,
    *,
    priors: ModelPriors,
):
    """Two-parameter logistic IRT model."""

    theta = numpyro.sample(
        "theta",
        dist.Normal(
            priors.theta.loc,
            priors.theta.scale,
        ).expand([n_judges]),
    )
    b = numpyro.sample(
        "b",
        dist.Normal(priors.b.loc, priors.b.scale).expand([n_items]),
    )
    a = numpyro.sample(
        "a",
        dist.LogNormal(priors.a.loc, priors.a.scale).expand([n_items]),
    )
    logits = a[item_idx] * (theta[judge_idx] - b[item_idx])
    with numpyro.plate("obs", len(correct)):
        numpyro.sample("correct", dist.Bernoulli(logits=logits), obs=correct)


def load_matrix_observations(matrix_path: Path) -> dict[str, Any]:
    """Convert a wide item-by-judge matrix into long-form IRT observations."""

    matrix = pl.read_parquet(matrix_path)
    judge_ids = [column for column in matrix.columns if column not in ITEM_METADATA_COLUMNS]
    item_ids = matrix.get_column("item_id").to_list()
    item_lookup = pl.DataFrame(
        {
            "item_id": item_ids,
            "item_idx": np.arange(len(item_ids), dtype=np.int32),
        }
    )
    judge_lookup = pl.DataFrame(
        {
            "judge_id": judge_ids,
            "judge_idx": np.arange(len(judge_ids), dtype=np.int32),
        }
    )
    observations = (
        matrix.select(["item_id", *judge_ids])
        .unpivot(
            on=judge_ids,
            index="item_id",
            variable_name="judge_id",
            value_name="correct",
        )
        .drop_nulls("correct")
        .join(item_lookup, on="item_id", how="left")
        .join(judge_lookup, on="judge_id", how="left")
        .sort(["item_idx", "judge_idx"])
    )
    return {
        "correct": observations.get_column("correct").cast(pl.Int32).to_numpy(),
        "judge_idx": observations.get_column("judge_idx").to_numpy(),
        "item_idx": observations.get_column("item_idx").to_numpy(),
        "n_judges": len(judge_ids),
        "n_items": len(item_ids),
        "judge_ids": np.asarray(judge_ids, dtype=str),
        "item_ids": np.asarray(item_ids, dtype=str),
    }


def run_mcmc(
    config: ExperimentConfig,
    observations: dict[str, Any],
) -> tuple[MCMC, dict[str, np.ndarray]]:
    """Run NumPyro NUTS for the configured model."""

    priors = build_model_priors(config.model)
    model_fn = irt_2pl if config.model.type == "2PL" else irt_1pl
    model = partial(model_fn, priors=priors)
    kernel = NUTS(model, target_accept_prob=config.inference.target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=config.inference.num_warmup,
        num_samples=config.inference.num_samples,
        num_chains=config.inference.num_chains,
        progress_bar=True,
    )
    rng_key = random.PRNGKey(config.experiment.seed)
    mcmc.run(
        rng_key,
        correct=observations["correct"],
        judge_idx=observations["judge_idx"],
        item_idx=observations["item_idx"],
        n_judges=observations["n_judges"],
        n_items=observations["n_items"],
    )
    samples = {
        name: np.asarray(values) for name, values in mcmc.get_samples(group_by_chain=True).items()
    }
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    samples["diverging"] = np.asarray(extra_fields.get("diverging", []))
    return mcmc, samples


def save_posterior(
    path: Path,
    samples: dict[str, np.ndarray],
    observations: dict[str, Any],
    model_type: str,
    metadata: dict[str, np.ndarray] | None = None,
) -> None:
    """Persist posterior samples and metadata to `.npz`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **samples,
        "judge_ids": observations["judge_ids"],
        "item_ids": observations["item_ids"],
        "model_type": np.asarray(model_type),
        "n_obs": np.asarray(observations["correct"].shape[0]),
    }
    if metadata is not None:
        payload.update(metadata)
    np.savez(
        path,
        **payload,
    )


def summarize_judges(samples: dict[str, np.ndarray], judge_ids: np.ndarray) -> pl.DataFrame:
    """Summarize posterior judge reliabilities."""

    theta_samples = samples["theta"].reshape(-1, samples["theta"].shape[-1])
    return pl.DataFrame(
        {
            "judge_id": judge_ids.tolist(),
            "theta_mean": theta_samples.mean(axis=0),
            "theta_sd": theta_samples.std(axis=0),
        }
    ).sort("theta_mean", descending=True)


def main() -> None:
    """CLI entrypoint for NumPyro IRT inference."""

    configure_logging()
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    config.ensure_directories()
    observations = load_matrix_observations(config.data.matrix_path)
    _, samples = run_mcmc(config, observations)
    output_path = config.inference.posterior_path_for_backend("numpyro")
    save_posterior(
        output_path,
        samples,
        observations,
        config.model.type,
        metadata={
            "backend": np.asarray("numpyro"),
            "experiment_seed": np.asarray(config.experiment.seed),
            "num_chains": np.asarray(config.inference.num_chains),
        },
    )
    summary = summarize_judges(samples, observations["judge_ids"])
    logger.info("saved_posterior=%s", output_path)
    if logger.isEnabledFor(logging.INFO):
        logger.info("judge summary\n%s", format_table_for_log(summary))


if __name__ == "__main__":
    main()
