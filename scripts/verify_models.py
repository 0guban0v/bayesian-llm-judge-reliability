"""Verify that candidate MLX judge models support constrained verdict decoding."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from pydantic import BaseModel, Field

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

    parser = argparse.ArgumentParser(
        description="Verify MLX model IDs for this repo's constrained verdict decoder."
    )
    parser.add_argument("models", nargs="+", help="One or more Hugging Face model IDs.")
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


def encode_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Encode text without adding special tokens when tokenizer supports it."""

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def resolve_verdict_label_token_ids(
    token_forms: dict[str, list[int]],
) -> tuple[list[int], list[int]]:
    """Return single-token encodings for A and B separately."""

    a_ids: set[int] = set()
    b_ids: set[int] = set()
    for form in ("A", " A"):
        token_ids = token_forms[form]
        if len(token_ids) == 1:
            a_ids.add(token_ids[0])
    for form in ("B", " B"):
        token_ids = token_forms[form]
        if len(token_ids) == 1:
            b_ids.add(token_ids[0])
    return sorted(a_ids), sorted(b_ids)


def resolve_eos_token_ids(tokenizer: Any) -> list[int]:
    """Return EOS token IDs available on the tokenizer."""

    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids:
        return [int(token_id) for token_id in eos_token_ids]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return [int(eos_token_id)]
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
        token_forms = {
            "A": encode_token_ids(tokenizer, "A"),
            "B": encode_token_ids(tokenizer, "B"),
            " A": encode_token_ids(tokenizer, " A"),
            " B": encode_token_ids(tokenizer, " B"),
        }
        verdict_a_token_ids, verdict_b_token_ids = resolve_verdict_label_token_ids(token_forms)
        eos_token_ids = resolve_eos_token_ids(tokenizer)
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
        verdict_token_ids=sorted(set(verdict_a_token_ids + verdict_b_token_ids)),
        eos_token_ids=eos_token_ids,
        token_forms=token_forms,
    )


def configure_logging(json_output: bool) -> None:
    """Configure CLI logging."""

    log_format = (
        "%(message)s" if json_output else "%(asctime)s %(levelname)s [verify_models] %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        stream=sys.stdout if json_output else None,
    )


def log_result(result: ModelVerificationResult) -> None:
    """Emit one human-readable verification summary."""

    status = "PASS" if result.supported else "FAIL"
    LOGGER.info(
        "%s model=%s loadable=%s chat_template=%s a_ids=%s b_ids=%s verdict_ids=%s eos_ids=%s",
        status,
        result.model,
        result.loadable,
        result.has_chat_template,
        result.verdict_a_token_ids,
        result.verdict_b_token_ids,
        result.verdict_token_ids,
        result.eos_token_ids,
    )
    LOGGER.info("token forms model=%s %s", result.model, json.dumps(result.token_forms))
    if result.error is not None:
        LOGGER.error("error model=%s %s", result.model, result.error)


def main() -> None:
    """Run model verification for all requested model IDs."""

    args = parse_args()
    configure_logging(args.json)
    results = [verify_model(model_name, args.trust_remote_code) for model_name in args.models]

    if args.json:
        LOGGER.info("%s", json.dumps([result.model_dump() for result in results], indent=2))
    else:
        for result in results:
            log_result(result)

    if not all(result.supported for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
