"""Verify that candidate MLX judge models support constrained verdict decoding."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from src.judges.token_constraints import (
    collect_verdict_token_forms,
    resolve_verdict_label_token_ids,
    resolve_verdict_token_ids,
)
from src.judges.token_constraints import (
    resolve_eos_token_ids as shared_resolve_eos_token_ids,
)
from src.schemas import ExperimentConfig, unique_model_requests

LOGGER = logging.getLogger("verify_models")


class ModelVerificationResult(BaseModel):
    """Compatibility report for one model ID under the current MLX backend."""

    model: str
    trust_remote_code: bool = False
    loadable: bool = False
    has_chat_template: bool = False
    verdict_a_token_ids: list[int] = Field(default_factory=list)
    verdict_b_token_ids: list[int] = Field(default_factory=list)
    verdict_token_ids: list[int] = Field(default_factory=list)
    eos_token_ids: list[int] = Field(default_factory=list)
    token_forms: dict[str, list[int]] = Field(default_factory=dict)
    error: str | None = None

    @property
    def supported(self) -> bool:
        """Return whether the model satisfies runtime constraints for this repo."""

        return (
            self.loadable
            and bool(self.verdict_a_token_ids)
            and bool(self.verdict_b_token_ids)
            and bool(self.verdict_token_ids)
            and bool(self.eos_token_ids)
            and self.error is None
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Verify MLX model IDs for this repo's constrained verdict decoder.")
    parser.add_argument(
        "models",
        nargs="*",
        help="Optional Hugging Face model IDs. Defaults to judges[*].model from --config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Experiment config used to resolve default model IDs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code through to tokenizer loading.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of human-oriented logs.",
    )
    return parser.parse_args()


def resolve_eos_token_ids(tokenizer: Any) -> list[int]:
    """Return EOS token IDs available on the tokenizer."""

    try:
        return shared_resolve_eos_token_ids(tokenizer)
    except ValueError:
        return []


def verify_model(model_name: str, trust_remote_code: bool) -> ModelVerificationResult:
    """Load one model and verify constrained-decoding prerequisites."""

    from mlx_lm import load

    tokenizer_config = {"trust_remote_code": True} if trust_remote_code else None
    try:
        if tokenizer_config is None:
            _, tokenizer = load(model_name)
        else:
            _, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
    except Exception as exc:
        return ModelVerificationResult(
            model=model_name,
            trust_remote_code=trust_remote_code,
            error=str(exc),
        )

    try:
        token_forms = collect_verdict_token_forms(tokenizer)
        verdict_a_token_ids, verdict_b_token_ids = resolve_verdict_label_token_ids(token_forms)
        verdict_token_ids = resolve_verdict_token_ids(tokenizer)
        eos_token_ids = shared_resolve_eos_token_ids(tokenizer)
    except Exception as exc:
        return ModelVerificationResult(
            model=model_name,
            trust_remote_code=trust_remote_code,
            loadable=True,
            has_chat_template=hasattr(tokenizer, "apply_chat_template"),
            error=str(exc),
        )

    return ModelVerificationResult(
        model=model_name,
        trust_remote_code=trust_remote_code,
        loadable=True,
        has_chat_template=hasattr(tokenizer, "apply_chat_template"),
        verdict_a_token_ids=verdict_a_token_ids,
        verdict_b_token_ids=verdict_b_token_ids,
        verdict_token_ids=verdict_token_ids,
        eos_token_ids=eos_token_ids,
        token_forms=token_forms,
    )


def configure_logging(json_output: bool) -> None:
    """Configure CLI logging."""

    log_format = "%(message)s" if json_output else "%(asctime)s %(levelname)s [verify_models] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        stream=sys.stdout if json_output else None,
    )


def log_result(result: ModelVerificationResult) -> None:
    """Emit one human-readable verification summary."""

    status = "PASS" if result.supported else "FAIL"
    LOGGER.info(
        ("%s model=%s trust_remote_code=%s loadable=%s chat_template=%s a_ids=%s b_ids=%s verdict_ids=%s eos_ids=%s"),
        status,
        result.model,
        result.trust_remote_code,
        result.loadable,
        result.has_chat_template,
        result.verdict_a_token_ids,
        result.verdict_b_token_ids,
        result.verdict_token_ids,
        result.eos_token_ids,
    )
    LOGGER.info(
        "token forms model=%s trust_remote_code=%s %s",
        result.model,
        result.trust_remote_code,
        json.dumps(result.token_forms),
    )
    if result.error is not None:
        LOGGER.error(
            "error model=%s trust_remote_code=%s %s",
            result.model,
            result.trust_remote_code,
            result.error,
        )


def main() -> None:
    """Run model verification for all requested model IDs."""

    args = parse_args()
    configure_logging(args.json)
    if args.models:
        requests = [(model_name, args.trust_remote_code) for model_name in args.models]
    else:
        config = ExperimentConfig.from_yaml(args.config)
        requests = unique_model_requests(config.judges)
    results = [verify_model(model_name, trust_remote_code) for model_name, trust_remote_code in requests]

    if args.json:
        LOGGER.info("%s", json.dumps([result.model_dump() for result in results], indent=2))
    else:
        for result in results:
            log_result(result)

    if not all(result.supported for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
