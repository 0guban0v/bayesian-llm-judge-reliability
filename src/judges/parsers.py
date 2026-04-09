"""Utilities for extracting pairwise verdicts from judge model output."""

from __future__ import annotations

import re
from typing import Literal

Verdict = Literal["A", "B", "TIE", "UNKNOWN"]
_TERMINAL_VERDICT_PATTERNS = (
    re.compile(r"FINAL\s+VERDICT\s*:\s*([AB])\s*$", re.IGNORECASE),
    re.compile(r"VERDICT\s*:\s*([AB])\s*$", re.IGNORECASE),
    re.compile(r"WINNER\s*:\s*([AB])\s*$", re.IGNORECASE),
    re.compile(r"OUTPUT\s*\((a|b)\)\s*$", re.IGNORECASE),
)
_TERMINAL_TIE_PATTERNS = (
    re.compile(r"\[\[A=B\]\]\s*$", re.IGNORECASE),
    re.compile(r"FINAL\s+VERDICT\s*:\s*TIE\s*$", re.IGNORECASE),
    re.compile(r"VERDICT\s*:\s*TIE\s*$", re.IGNORECASE),
    re.compile(r"TIE\s*$", re.IGNORECASE),
    re.compile(r"NEITHER\s*$", re.IGNORECASE),
)
_TERMINAL_LINE_COUNT = 5


def normalize_label(label: str) -> Literal["A>B", "B>A"]:
    """Normalize ground-truth preference labels."""

    normalized = label.strip().upper()
    if normalized not in {"A>B", "B>A"}:
        raise ValueError(f"Unsupported label: {label}")
    return normalized  # type: ignore[return-value]


def _terminal_window(raw_text: str, max_lines: int = _TERMINAL_LINE_COUNT) -> str:
    """Return the trailing non-empty lines most likely to contain the final verdict."""

    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def _terminal_line(raw_text: str) -> str:
    """Return the final non-empty line from a raw response."""

    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def parse_verdict(raw_text: str) -> Verdict:
    """Parse a judge response into a canonical verdict."""

    if not raw_text.strip():
        return "UNKNOWN"
    terminal_text = _terminal_window(raw_text)
    terminal_line = _terminal_line(raw_text).upper()
    for pattern in _TERMINAL_TIE_PATTERNS:
        if pattern.search(terminal_text):
            return "TIE"
    for pattern in _TERMINAL_VERDICT_PATTERNS:
        match = pattern.search(terminal_text)
        if match is not None:
            return match.group(1).upper()  # type: ignore[return-value]
    if terminal_line in {"A", "B"}:
        return terminal_line  # type: ignore[return-value]
    return "UNKNOWN"


def swap_verdict(verdict: Verdict) -> Verdict:
    """Map a verdict from reversed prompt order back to original order."""

    if verdict == "A":
        return "B"
    if verdict == "B":
        return "A"
    return verdict


def parse_correctness(verdict: Verdict, label: str) -> bool | None:
    """Convert a verdict into correctness against the ground-truth label."""

    canonical_label = normalize_label(label)
    if verdict not in {"A", "B"}:
        return None
    return (verdict == "A" and canonical_label == "A>B") or (
        verdict == "B" and canonical_label == "B>A"
    )
