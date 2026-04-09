"""Utilities for extracting pairwise verdicts from judge model output."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, cast

from pydantic import BaseModel, ValidationError, model_validator

ParsedVerdict = Literal["A", "B", "TIE", "UNKNOWN"]


class Verdict(StrEnum):
    """Strict judge verdict labels accepted from model output."""

    A = "A"
    B = "B"


class JudgeInput(BaseModel):
    """Structured prompt payload for a single judge invocation."""

    question: str
    response_a: str
    response_b: str

    def to_prompt(self, template: str) -> str:
        """Render a prompt template with normalized string fields."""

        return template.format(
            question=self.question.strip(),
            response_a=self.response_a.strip(),
            response_b=self.response_b.strip(),
        )


class JudgeOutput(BaseModel):
    """Strict first-line parser for judge model output."""

    raw: str
    verdict: Verdict

    @model_validator(mode="before")
    @classmethod
    def extract_verdict(cls, values: dict[str, object]) -> dict[str, object]:
        """Extract a verdict from the first non-empty output line."""

        raw = str(values.get("raw", ""))
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Parse failure: empty output")
        first_line = lines[0]
        if first_line == "FINAL VERDICT: A":
            values["verdict"] = Verdict.A
            return values
        if first_line == "FINAL VERDICT: B":
            values["verdict"] = Verdict.B
            return values
        raise ValueError(f"Parse failure — raw output: {raw!r}")


def normalize_label(label: str) -> Literal["A>B", "B>A"]:
    """Normalize ground-truth preference labels."""

    normalized = label.strip().upper()
    if normalized not in {"A>B", "B>A"}:
        raise ValueError(f"Unsupported label: {label}")
    return cast(Literal["A>B", "B>A"], normalized)


def parse_verdict(raw_text: str) -> ParsedVerdict:
    """Parse a judge response using a strict first-line verdict contract."""

    try:
        output = JudgeOutput.model_validate({"raw": raw_text})
    except ValidationError:
        return "UNKNOWN"
    return cast(ParsedVerdict, output.verdict.value)


def swap_verdict(verdict: ParsedVerdict) -> ParsedVerdict:
    """Map a verdict from reversed prompt order back to original order."""

    if verdict == "A":
        return "B"
    if verdict == "B":
        return "A"
    return verdict


def parse_correctness(verdict: ParsedVerdict, label: str) -> bool | None:
    """Convert a verdict into correctness against the ground-truth label."""

    canonical_label = normalize_label(label)
    if verdict not in {"A", "B"}:
        return None
    return (verdict == "A" and canonical_label == "A>B") or (
        verdict == "B" and canonical_label == "B>A"
    )
