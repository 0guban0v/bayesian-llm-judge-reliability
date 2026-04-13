"""Prefetch configured MLX judge models with local platform checks."""

from __future__ import annotations

import argparse
import logging
import os
import platform
import subprocess
from pathlib import Path

from src.logging_utils import configure_logging
from src.schemas import ExperimentConfig, unique_model_requests

LOGGER = logging.getLogger("setup_models")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def ensure_mlx_platform() -> None:
    """Fail fast when local MLX execution prerequisites are not present."""

    if platform.system() != "Darwin":
        raise RuntimeError("mlx setup requires macOS on Apple Silicon.")

    machine = platform.machine().lower()
    processor = platform.processor().lower()
    if machine not in {"arm64", "aarch64"} and processor not in {"arm", "arm64", "apple"}:
        raise RuntimeError(
            "mlx setup requires Apple Silicon (arm64); Intel macOS is not supported."
        )
    result = subprocess.run(
        ["system_profiler", "SPDisplaysDataType"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("system_profiler is unavailable; cannot verify Metal support.")
    if "Metal" not in result.stdout:
        raise RuntimeError(
            "No Metal-capable GPU was detected by system_profiler. "
            "MLX requires a local macOS session with Metal available."
        )


def load_model(model_name: str, trust_remote_code: bool) -> None:
    """Load one MLX model into the local cache."""

    from mlx_lm import load

    tokenizer_config = {"trust_remote_code": True} if trust_remote_code else None
    if tokenizer_config is None:
        load(model_name)
    else:
        load(model_name, tokenizer_config=tokenizer_config)


def login_if_configured() -> None:
    """Authenticate to Hugging Face when a token is available."""

    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)


def prefetch_models(config: ExperimentConfig) -> None:
    """Load each configured model once and fail if any load does not succeed."""

    failures: list[str] = []
    for model_name, trust_remote_code in unique_model_requests(config.judges):
        LOGGER.info("loading %s", model_name)
        try:
            load_model(model_name, trust_remote_code)
        except Exception as exc:
            failures.append(f"{model_name}: {exc}")
            LOGGER.exception("failed %s", model_name)
            continue
        LOGGER.info("loaded %s", model_name)

    if failures:
        LOGGER.error("setup summary: failures detected")
        for failure in failures:
            LOGGER.error("- %s", failure)
        raise SystemExit(1)

    LOGGER.info("setup summary: all pinned models loaded")


def main() -> None:
    """CLI entrypoint for model setup."""

    configure_logging()
    args = parse_args()
    ensure_mlx_platform()
    login_if_configured()
    config = ExperimentConfig.from_yaml(args.config)
    try:
        prefetch_models(config)
    except SystemExit:
        raise
    except Exception:
        LOGGER.exception("MLX model setup failed")
        LOGGER.error(
            "This usually means Metal could not be initialized from the current process. "
            "If you are on macOS with Apple Silicon, retry from a normal local Terminal session."
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
