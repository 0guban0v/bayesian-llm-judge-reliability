"""Minimal MLX generation helpers for local judge inference."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


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
    """Use the tokenizer chat template when available."""

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


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
    logger.debug(
        "generating model=%s prompt_chars=%s max_tokens=%s",
        model_name,
        len(rendered_prompt),
        max_tokens,
    )
    return generate(
        loaded_model.model,
        loaded_model.tokenizer,
        prompt=rendered_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
