"""Shared logging configuration for CLI modules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure process-wide logging once for command-line entrypoints."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        return
    logging.basicConfig(level=level, format=LOG_FORMAT)


def format_table_for_log(frame: "pl.DataFrame") -> str:
    """Render a Polars DataFrame for logs using Polars' built-in formatter."""

    import polars as pl

    if frame.height == 0:
        return "<empty>"

    with pl.Config() as config:
        config.set_ascii_tables(True)
        config.set_fmt_str_lengths(100)
        config.set_tbl_cols(-1)
        config.set_tbl_rows(-1)
        config.set_tbl_width_chars(200)
        config.set_tbl_hide_column_data_types(True)
        config.set_tbl_hide_dataframe_shape(True)
        config.set_tbl_hide_dtype_separator(True)
        return str(frame)
