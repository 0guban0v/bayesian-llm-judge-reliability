"""Stable content identities for JudgeBench items."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping

ITEM_CONTENT_FIELDS = (
    "question",
    "response_a",
    "response_b",
    "label",
    "source",
    "split",
)
ITEM_CONTENT_HASH_PATTERN = r"^[0-9a-f]{64}$"


def normalize_item_content(value: object, *, field: str) -> str:
    """Normalize one text field for stable content hashing."""

    if not isinstance(value, str):
        raise TypeError(f"Item content field '{field}' must be a string, found {type(value).__name__}")
    normalized_newlines = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", normalized_newlines).strip()


def item_content_hash(item: Mapping[str, object]) -> str:
    """Return a deterministic SHA-256 identity for judge-relevant item content."""

    missing = [field for field in ITEM_CONTENT_FIELDS if field not in item]
    if missing:
        raise ValueError(f"Item content hash requires fields: {', '.join(missing)}")
    payload = {field: normalize_item_content(item[field], field=field) for field in ITEM_CONTENT_FIELDS}
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def validate_item_content_hash(item: Mapping[str, object]) -> None:
    """Require one item mapping to contain its expected content hash."""

    stored_hash = item.get("item_content_hash")
    if not isinstance(stored_hash, str):
        raise ValueError("Item content field 'item_content_hash' must be a string")
    expected_hash = item_content_hash(item)
    if stored_hash != expected_hash:
        item_key = item.get("item_key", "<unknown>")
        raise ValueError(
            f"Item content hash mismatch for item_key={item_key}: "
            f"stored_hash={stored_hash} expected_hash={expected_hash}"
        )
