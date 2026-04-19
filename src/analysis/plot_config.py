"""Shared plotting configuration and style helpers."""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np
from matplotlib.axes import Axes

EXPORT_DPI = 600

FONT_SIZE_TICK: float = 9.0
FONT_SIZE_ANNOTATION: float = 8.5
FONT_SIZE_TITLE: float = 10.0

COLOR_TEXT_DARK: str = "#202124"
COLOR_TEXT_LIGHT: str = "#ffffff"
COLOR_DIAGNOSTIC: str = "#0f4c81"
COLOR_SURFACE: str = "#ffffff"

JUDGE_COLOR_PINS = {
    "deepseek-r1-distill-qwen-14b": COLOR_DIAGNOSTIC,
    "deepseek-r1-distill-qwen-7b": "#4a4a4a",
    "mistral-7b-instruct-v0-3": "#d95f02",
    "qwen2-5-7b-instruct": "#4daf4a",
    "gemma-2-9b-it": "#984ea3",
}
JUDGE_LABEL_PINS = {
    "deepseek-r1-distill-qwen-14b": "DeepSeek 14B",
    "deepseek-r1-distill-qwen-7b": "DeepSeek 7B",
    "mistral-7b-instruct-v0-3": "Mistral 7B",
    "qwen2-5-7b-instruct": "Qwen 7B",
    "gemma-2-9b-it": "Gemma 9B",
}
SOURCE_COLOR_PINS = {
    "livebench-reasoning": "#1f4e79",
    "livebench-math": "#2f6b55",
    "livecodebench": "#7a3e48",
    "mmlu-pro-computer science": "#5a6f8f",
    "mmlu-pro-math": "#7b5a8c",
    "mmlu-pro-chemistry": "#8a6a43",
    "mmlu-pro-physics": "#3f6f78",
    "mmlu-pro-philosophy": "#6a5b4d",
    "mmlu-pro-health": "#4f7a68",
    "mmlu-pro-economics": "#4e5d6c",
}
SOURCE_LABEL_PINS = {
    "livebench-reasoning": "LB Reasoning",
    "livebench-math": "LB Math",
    "livecodebench": "LiveCodeBench",
    "mmlu-pro-computer science": "MMLU CS",
    "mmlu-pro-math": "MMLU Math",
    "mmlu-pro-chemistry": "MMLU Chemistry",
    "mmlu-pro-physics": "MMLU Physics",
    "mmlu-pro-philosophy": "MMLU Philosophy",
    "mmlu-pro-health": "MMLU Health",
    "mmlu-pro-economics": "MMLU Economics",
}


def judge_display_label(judge_id: str) -> str:
    """Return a compact display label for a judge identifier."""

    return JUDGE_LABEL_PINS.get(judge_id, judge_id)


def source_display_label(source_id: str) -> str:
    """Return a compact display label for a source identifier."""

    if source_id in SOURCE_LABEL_PINS:
        return SOURCE_LABEL_PINS[source_id]
    if source_id.startswith("mmlu-pro-"):
        suffix = source_id.removeprefix("mmlu-pro-").replace("-", " ").title()
        return f"MMLU {suffix}"
    if source_id.startswith("livebench-"):
        suffix = source_id.removeprefix("livebench-").replace("-", " ").title()
        return f"LB {suffix}"
    return source_id.replace("-", " ")


def style_axis(ax: Axes) -> None:
    """Remove unused frame lines and keep a cleaner plotting surface."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def fallback_plot_color(name: str) -> str:
    """Generate a deterministic fallback color for a named series."""

    digest = hashlib.sha256(name.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], byteorder="big") / 65535.0
    saturation = 0.55 + (digest[2] / 255.0) * 0.15
    value = 0.65 + (digest[3] / 255.0) * 0.2
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(red * 255):02x}{int(green * 255):02x}{int(blue * 255):02x}"


def judge_color_map(judge_ids: np.ndarray) -> dict[str, str]:
    """Return stable colors for each judge ID."""

    return {judge_id: JUDGE_COLOR_PINS.get(judge_id, fallback_plot_color(judge_id)) for judge_id in map(str, judge_ids)}


def source_color_map(source_ids: list[str]) -> dict[str, str]:
    """Return stable colors for each source ID."""

    return {source_id: SOURCE_COLOR_PINS.get(source_id, fallback_plot_color(source_id)) for source_id in source_ids}
