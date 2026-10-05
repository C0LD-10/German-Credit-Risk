"""
Fairness audit.

Excluding a protected attribute from model features does not guarantee a
fair outcome: correlated features (e.g. `Job`, `Housing`) can act as proxies
and reintroduce disparate impact. This module checks the model's actual
output distribution across the protected attribute, even though that
attribute was never an input feature.

This is a preliminary, single-attribute audit — see README limitations for
what a production fairness review would additionally require (intersectional
groups, formal parity/equalized-odds metrics, legal/regulatory review).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def audit_by_sensitive_attribute(
    sensitive_values: pd.Series, y_proba: np.ndarray, y_pred: np.ndarray
) -> pd.DataFrame:
    """Return mean predicted risk and flagged-high-risk rate per group of
    `sensitive_values`, aligned by index with `y_proba`/`y_pred`.
    """
    audit_df = pd.DataFrame({
        "sensitive_value": sensitive_values.values,
        "pred_proba": y_proba,
        "pred": y_pred,
    })
    result = audit_df.groupby("sensitive_value").agg(
        mean_predicted_risk=("pred_proba", "mean"),
        flagged_high_risk_rate=("pred", "mean"),
        n=("pred", "size"),
    ).round(4)
    if len(result) == 2:
        rates = result["flagged_high_risk_rate"]
        disparate_impact_ratio = rates.min() / rates.max() if rates.max() > 0 else np.nan
        result.attrs["disparate_impact_ratio"] = float(disparate_impact_ratio)
        # Common (if blunt) rule of thumb from US EEOC guidance: ratios below
        # 0.8 are conventionally flagged for further review.
        result.attrs["flag_for_review"] = bool(disparate_impact_ratio < 0.8)
    return result
