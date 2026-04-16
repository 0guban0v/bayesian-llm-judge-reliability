"""Shared posterior archive schema and validated loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSTERIOR_SCHEMA_VERSION = 1
_REQUIRED_KEYS = {"theta", "b", "judge_ids", "item_ids", "source_ids", "model_type", "n_obs"}
_OPTIONAL_KEYS = {
    "a",
    "tau_theta",
    "theta_source",
    "diverging",
    "backend",
    "experiment_seed",
    "num_chains",
    "posterior_schema_version",
}
_SUPPORTED_SCHEMA_VERSIONS = {0, POSTERIOR_SCHEMA_VERSION}


@dataclass(frozen=True)
class PosteriorArchive:
    """Validated posterior archive payload loaded from `.npz`."""

    payload: dict[str, np.ndarray]
    schema_version: int

    def as_dict(self) -> dict[str, np.ndarray]:
        """Return a shallow copy of validated archive arrays."""

        return dict(self.payload)


def _require_keys(payload: dict[str, np.ndarray]) -> None:
    missing = sorted(_REQUIRED_KEYS - payload.keys())
    if missing:
        raise ValueError(f"Posterior archive is missing required keys: {', '.join(missing)}")


def _require_ndim(payload: dict[str, np.ndarray], key: str, ndim: int) -> np.ndarray:
    values = np.asarray(payload[key])
    if values.ndim != ndim:
        raise ValueError(f"Posterior field '{key}' must have rank {ndim}, found rank {values.ndim}")
    return values


def _require_scalar(payload: dict[str, np.ndarray], key: str) -> np.ndarray:
    values = np.asarray(payload[key])
    if values.ndim != 0:
        raise ValueError(f"Posterior field '{key}' must be a scalar value")
    return values


def validate_posterior_payload(payload: dict[str, np.ndarray]) -> PosteriorArchive:
    """Validate archive keys, shapes, and metadata relationships."""

    schema_version = int(np.asarray(payload.get("posterior_schema_version", np.asarray(0))))
    if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported posterior schema version: {schema_version}")
    _require_keys(payload)
    theta = _require_ndim(payload, "theta", 3)
    b = _require_ndim(payload, "b", 3)
    judge_ids = _require_ndim(payload, "judge_ids", 1)
    item_ids = _require_ndim(payload, "item_ids", 1)
    source_ids = _require_ndim(payload, "source_ids", 1)
    model_type = str(_require_scalar(payload, "model_type"))
    _require_scalar(payload, "n_obs")
    if theta.shape[2] != len(judge_ids):
        raise ValueError(
            f"Posterior theta shape does not match judge_ids length. theta={theta.shape} judge_ids={len(judge_ids)}"
        )
    if b.shape[2] != len(item_ids):
        raise ValueError(f"Posterior b shape does not match item_ids length. b={b.shape} item_ids={len(item_ids)}")
    if "a" in payload:
        a = _require_ndim(payload, "a", 3)
        if a.shape[2] != len(item_ids):
            raise ValueError(f"Posterior a shape does not match item_ids length. a={a.shape} item_ids={len(item_ids)}")
    if model_type == "2PL" and "a" not in payload:
        raise ValueError("Posterior archive for model_type '2PL' must contain 'a'")
    if "tau_theta" in payload:
        tau_theta = _require_ndim(payload, "tau_theta", 3)
        if tau_theta.shape[2] != len(judge_ids):
            raise ValueError(
                "Posterior tau_theta shape does not match judge_ids length. "
                f"tau_theta={tau_theta.shape} judge_ids={len(judge_ids)}"
            )
    if "theta_source" in payload:
        theta_source = _require_ndim(payload, "theta_source", 4)
        if theta_source.shape[2] != len(judge_ids) or theta_source.shape[3] != len(source_ids):
            raise ValueError(
                "Posterior theta_source shape does not match judge_ids/source_ids lengths. "
                f"theta_source={theta_source.shape} judge_ids={len(judge_ids)} source_ids={len(source_ids)}"
            )
    if "diverging" in payload:
        diverging = _require_ndim(payload, "diverging", 2)
        if diverging.shape != theta.shape[:2]:
            raise ValueError(
                "Posterior diverging shape does not match chain/draw axes. "
                f"diverging={diverging.shape} theta={theta.shape[:2]}"
            )
    if "num_chains" in payload:
        num_chains = int(_require_scalar(payload, "num_chains"))
        if num_chains != theta.shape[0]:
            raise ValueError(
                "Posterior num_chains does not match sample arrays. "
                f"num_chains={num_chains} theta_chains={theta.shape[0]}"
            )
    validated = {
        key: np.asarray(value) for key, value in payload.items() if key in _REQUIRED_KEYS or key in _OPTIONAL_KEYS
    }
    validated["judge_ids"] = validated["judge_ids"].astype(str)
    validated["item_ids"] = validated["item_ids"].astype(str)
    validated["source_ids"] = validated["source_ids"].astype(str)
    validated["model_type"] = np.asarray(model_type)
    validated["posterior_schema_version"] = np.asarray(schema_version)
    return PosteriorArchive(payload=validated, schema_version=schema_version)


def load_posterior_archive(path: Path) -> PosteriorArchive:
    """Load and validate a saved posterior archive."""

    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    return validate_posterior_payload(payload)


def load_posterior(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a saved posterior archive as a plain mapping."""

    return load_posterior_archive(path).as_dict()
