"""Shared logging configuration for CLI modules."""

from __future__ import annotations

import logging
from typing import Any

import polars as pl

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure process-wide logging once for command-line entrypoints."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return
    logging.basicConfig(level=level, format=LOG_FORMAT)


def format_table_for_log(frame: pl.DataFrame) -> str:
    """Render a Polars DataFrame as a plain text table without dtype metadata."""

    if frame.height == 0:
        return "<empty>"

    headers = frame.columns

    def stringify(value: Any) -> str:
        return str(value)

    rows = [[stringify(row[header]) for header in headers] for row in frame.to_dicts()]
    widths = [
        max(len(header), *(len(row[column_index]) for row in rows))
        for column_index, header in enumerate(headers)
    ]

    def format_row(values: list[str]) -> str:
        return " | ".join(value.ljust(width) for value, width in zip(values, widths, strict=True))

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)
