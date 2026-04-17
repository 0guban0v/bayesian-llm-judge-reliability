"""Shared tokenizer constraints for verdict-only judge decoding."""

from __future__ import annotations

from typing import Any


def encode_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Encode text without adding special tokens when tokenizer supports it."""

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def collect_verdict_token_forms(tokenizer: Any) -> dict[str, list[int]]:
    """Collect raw tokenizations for plain and space-prefixed verdict labels."""

    return {
        "A": encode_token_ids(tokenizer, "A"),
        "B": encode_token_ids(tokenizer, "B"),
        " A": encode_token_ids(tokenizer, " A"),
        " B": encode_token_ids(tokenizer, " B"),
    }


def resolve_verdict_label_token_ids(token_forms: dict[str, list[int]]) -> tuple[list[int], list[int]]:
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


def resolve_verdict_token_ids(tokenizer: Any) -> list[int]:
    """Return all single-token encodings accepted for verdict decoding."""

    verdict_a_token_ids, verdict_b_token_ids = resolve_verdict_label_token_ids(collect_verdict_token_forms(tokenizer))
    if not verdict_a_token_ids or not verdict_b_token_ids:
        raise ValueError("Tokenizer does not provide single-token verdict labels for both A and B.")
    return sorted(set(verdict_a_token_ids + verdict_b_token_ids))


def resolve_eos_token_ids(tokenizer: Any) -> list[int]:
    """Return EOS token IDs required to stop after one verdict token."""

    eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_token_ids:
        return [int(token_id) for token_id in eos_token_ids]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        return [int(eos_token_id)]
    raise ValueError("Tokenizer does not expose EOS token IDs for constrained generation.")
