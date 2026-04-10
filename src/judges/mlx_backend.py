"""Minimal MLX generation helpers for local judge inference."""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

VERDICT_PREFIX = "FINAL VERDICT: "


@dataclass(slots=True)
class LoadedModel:
    """In-memory MLX model plus tokenizer."""

    model: Any
    tokenizer: Any


_MODEL_CACHE: dict[tuple[str, bool], LoadedModel] = {}


def clear_model_cache() -> None:
    """Drop all cached MLX models and tokenizers for the current process."""

    cached_models = list(_MODEL_CACHE.values())
    cleared = len(cached_models)
    _MODEL_CACHE.clear()
    del cached_models
    gc.collect()
    logger.info("cleared mlx model cache entries=%s", cleared)


def get_model(model_name: str, trust_remote_code: bool) -> LoadedModel:
    """Load and cache an MLX-compatible model."""

    from mlx_lm import load

    cache_key = (model_name, trust_remote_code)
    if cache_key not in _MODEL_CACHE:
        logger.info("loading model=%s trust_remote_code=%s", model_name, trust_remote_code)
        tokenizer_config = {"trust_remote_code": True} if trust_remote_code else None
        if tokenizer_config is None:
            model, tokenizer = load(model_name)
        else:
            model, tokenizer = load(model_name, tokenizer_config=tokenizer_config)
        _MODEL_CACHE[cache_key] = LoadedModel(model=model, tokenizer=tokenizer)
        logger.info("model ready=%s", model_name)
    else:
        logger.debug("cache hit model=%s", model_name)
    return _MODEL_CACHE[cache_key]


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Render the prompt with an assistant-side verdict prefix when available."""

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": VERDICT_PREFIX},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            add_generation_prompt=False,
        )
    return f"{prompt} "


def _encode_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Encode text into token IDs without adding special tokens."""

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def _resolve_verdict_token_ids(tokenizer: Any) -> list[int]:
    """Return single-token encodings for the constrained verdict labels."""

    verdict_a_ids: set[int] = set()
    verdict_b_ids: set[int] = set()
    for candidate in ("A", " A"):
        encoded = _encode_token_ids(tokenizer, candidate)
        if len(encoded) == 1:
            verdict_a_ids.add(encoded[0])
    for candidate in ("B", " B"):
        encoded = _encode_token_ids(tokenizer, candidate)
        if len(encoded) == 1:
            verdict_b_ids.add(encoded[0])
    if not verdict_a_ids or not verdict_b_ids:
        raise ValueError("Tokenizer does not provide single-token verdict labels for both A and B.")
    return sorted(verdict_a_ids | verdict_b_ids)


def _resolve_eos_token_ids(tokenizer: Any) -> list[int]:
    """Return EOS token IDs required to stop after one verdict token."""

    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids:
        return [int(token_id) for token_id in eos_token_ids]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return [int(eos_token_id)]
    raise ValueError("Tokenizer does not expose EOS token IDs for constrained generation.")


def make_verdict_processor(tokenizer: Any) -> Callable[[Any, Any], Any]:
    """Constrain generation to one verdict token followed by EOS."""

    import mlx.core as mx

    verdict_token_ids = _resolve_verdict_token_ids(tokenizer)
    eos_token_ids = _resolve_eos_token_ids(tokenizer)
    call_index = 0
    logger.debug(
        "verdict processor configured allowed_verdict_ids=%s eos_ids=%s",
        verdict_token_ids,
        eos_token_ids,
    )

    def processor(_tokens: Any, logits: Any) -> Any:
        nonlocal call_index
        allowed_token_ids = verdict_token_ids if call_index == 0 else eos_token_ids
        call_index += 1
        vocab_ids = mx.arange(logits.shape[-1])[None, :]
        allowed_mask = vocab_ids == allowed_token_ids[0]
        for token_id in allowed_token_ids[1:]:
            allowed_mask = mx.logical_or(allowed_mask, vocab_ids == token_id)
        blocked_logits = mx.full(logits.shape, float("-inf"))
        return mx.where(allowed_mask, logits, blocked_logits)

    return processor


def generate_text(
    *,
    model_name: str,
    prompt: str,
    max_tokens: int,
    trust_remote_code: bool,
) -> str:
    """Generate deterministic text with MLX."""

    from mlx_lm import generate

    loaded_model = get_model(model_name, trust_remote_code)
    rendered_prompt = format_chat_prompt(loaded_model.tokenizer, prompt)
    verdict_processor = make_verdict_processor(loaded_model.tokenizer)
    logger.debug(
        "generating model=%s prompt_chars=%s max_tokens=%s",
        model_name,
        len(rendered_prompt),
        max_tokens,
    )
    response = generate(
        loaded_model.model,
        loaded_model.tokenizer,
        prompt=rendered_prompt,
        max_tokens=max_tokens,
        logits_processors=[verdict_processor],
        verbose=False,
    )
    normalized_response = f"{VERDICT_PREFIX}{response.strip()}"
    logger.debug("normalized constrained response=%r", normalized_response)
    return normalized_response
