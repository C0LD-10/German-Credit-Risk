"""
Data loading and schema validation.

Fails loudly and specifically (missing columns named, not a generic
traceback) when the input file doesn't match the expected schema — this is
the first line of defense against silently analyzing the wrong file or a
version of the dataset with a different column layout.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import Config

logger = logging.getLogger(__name__)


class SchemaValidationError(ValueError):
    """Raised when the input data does not match the expected schema."""


def load_raw_data(cfg: Config) -> pd.DataFrame:
    """Load the raw applicant CSV and validate it against `config.yaml:schema`.

    Returns
    -------
    pd.DataFrame indexed by the original row id, with exactly the columns
    declared in `schema.required_columns` (extra columns are dropped with a
    warning rather than silently carried through).
    """
    path: Path = cfg.path("raw_data")
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path, index_col=0)
    validate_schema(df, cfg)

    required = cfg["schema"]["required_columns"]
    extra = [c for c in df.columns if c not in required]
    if extra:
        logger.warning("Dropping unexpected columns not in schema: %s", extra)
        df = df[required]

    logger.info("Loaded raw data: %d rows x %d columns from %s", *df.shape, path)
    return df


def validate_schema(df: pd.DataFrame, cfg: Config) -> None:
    """Raise `SchemaValidationError` with a precise diagnosis if `df` doesn't
    match the columns declared in config.yaml, or contains implausible values
    in numeric fields that would silently corrupt downstream feature
    engineering (e.g. non-positive Duration or Credit amount).
    """
    required = set(cfg["schema"]["required_columns"])
    present = set(df.columns)
    missing = required - present
    if missing:
        raise SchemaValidationError(
            f"Input data is missing required column(s): {sorted(missing)}. "
            f"Expected schema: {sorted(required)}."
        )

    numeric_checks = {"Age": (0, 120), "Credit amount": (0, None), "Duration": (0, None)}
    for col, (lo, hi) in numeric_checks.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SchemaValidationError(f"Column '{col}' is expected to be numeric.")
        bad = df[col] <= lo if lo is not None else pd.Series(False, index=df.index)
        if hi is not None:
            bad = bad | (df[col] > hi)
        if bad.any():
            raise SchemaValidationError(
                f"Column '{col}' has {int(bad.sum())} implausible value(s) "
                f"outside the expected range (> {lo}"
                + (f", <= {hi}" if hi is not None else "")
                + f"). Example offending rows: {df.index[bad][:5].tolist()}"
            )

    if df.duplicated().sum() > 0:
        logger.warning("Input data contains %d exact duplicate rows.", df.duplicated().sum())
