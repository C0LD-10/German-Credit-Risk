"""
Feature engineering.

Each engineered feature is motivated by an explicit, stated hypothesis about
credit risk (see README / notebook Section 12) rather than generated
mechanically — kept as a single, deterministic function so the exact same
transformation can be applied at training time and at inference time
(`app/app.py`), which is the most common source of train/serve skew bugs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config

ENGINEERED_NUMERIC_COLUMNS = [
    "log_credit_amount",
    "installment_proxy",
    "saving_ord",
    "checking_ord",
    "financial_buffer",
    "long_duration_flag",
]


def engineer_features(df: pd.DataFrame, cfg: Config, long_duration_threshold: float | None = None) -> pd.DataFrame:
    """Add engineered features to an already-missing-treated DataFrame.

    Parameters
    ----------
    df:
        Output of `preprocessing.treat_missing_accounts` (must have ordered
        categorical `Saving accounts` / `Checking account` columns).
    cfg:
        Project configuration (used for the long-duration quantile at fit
        time).
    long_duration_threshold:
        Pass the *training-set* threshold explicitly at inference time so a
        single applicant can be scored without recomputing a quantile from a
        dataset of one row. If None, it is computed from `df` itself (fit-time
        usage).
    """
    out = df.copy()

    out["log_credit_amount"] = np.log1p(out["Credit amount"])
    out["installment_proxy"] = out["Credit amount"] / out["Duration"]

    out["age_group"] = pd.cut(
        out["Age"], bins=[18, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"],
    )

    out["saving_ord"] = out["Saving accounts"].cat.codes
    out["checking_ord"] = out["Checking account"].cat.codes
    out["financial_buffer"] = out["saving_ord"] + out["checking_ord"]

    if long_duration_threshold is None:
        long_duration_threshold = out["Duration"].quantile(
            cfg.get("features", "long_duration_quantile", default=0.75)
        )
    out["long_duration_flag"] = (out["Duration"] >= long_duration_threshold).astype(int)
    out.attrs["long_duration_threshold"] = float(long_duration_threshold)

    return out
