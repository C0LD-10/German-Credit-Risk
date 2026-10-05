"""
Proxy risk score construction.

================================================================================
CRITICAL CAVEAT — read before using anything in this module.
================================================================================
This module builds a HAND-ENGINEERED HEURISTIC, not an empirical estimate of
default probability. It exists purely so the supervised-learning pipeline in
`src/modeling.py` has a target to demonstrate leakage-safe training,
cross-validation, calibration, and cost-sensitive thresholding against. A
model trained to predict `proxy_high_risk` is learning to reconstruct this
formula, not real-world credit risk. Any accuracy/AUC produced downstream
reflects that circularity and must never be reported as real predictive
performance (see README "Limitations").

`Sex` is intentionally never used as an input to this score, and is excluded
from all model features elsewhere in this project — see `src/fairness.py` for
the audit that checks whether *other* features act as a proxy for it anyway.

The moment a real default/repayment label is available, set
`config.yaml: target.mode: "real_label"` and `target.label_column` to point
at it; `src/modeling.py` already reads the target through
`get_target(df, cfg)` below and requires no further code changes.
================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config

PROXY_COMPONENTS = [
    "low_checking", "low_saving", "high_installment",
    "long_duration", "low_job_skill", "young_age_extreme",
]


def _minmax(s: pd.Series) -> pd.Series:
    rng = s.max() - s.min()
    if rng == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rng


def build_proxy_risk_score(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add `proxy_risk_score` (continuous, 0-1) and `proxy_high_risk` (binary)
    columns to `df` using the weights declared in `config.yaml: target.proxy`.
    """
    weights = cfg.get("target", "proxy", "weights")
    quantile_threshold = cfg.get("target", "proxy", "quantile_threshold", default=0.70)
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"Proxy score weights must sum to 1.0, got {sum(weights.values())}")

    out = df.copy()
    components = pd.DataFrame(index=out.index)
    components["low_checking"] = _minmax(-out["checking_ord"])
    components["low_saving"] = _minmax(-out["saving_ord"])
    components["high_installment"] = _minmax(out["installment_proxy"])
    components["long_duration"] = _minmax(out["Duration"])
    components["low_job_skill"] = _minmax(-out["Job"])
    components["young_age_extreme"] = _minmax(np.abs(out["Age"] - out["Age"].median()))

    out["proxy_risk_score"] = sum(components[c] * w for c, w in weights.items())
    threshold = out["proxy_risk_score"].quantile(quantile_threshold)
    out["proxy_high_risk"] = (out["proxy_risk_score"] >= threshold).astype(int)
    out.attrs["proxy_threshold"] = float(threshold)
    out.attrs["proxy_weights"] = dict(weights)

    return out


def get_target(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, str]:
    """Return (df_with_target, y, target_mode) according to `config.yaml:
    target.mode`.

    This is the single switch point between the current proxy-target
    methodology demonstration and a future real-label model: when
    `target.mode == "real_label"`, every other module downstream (modeling,
    evaluation, cost-sensitive thresholding) works unchanged.
    """
    mode = cfg.get("target", "mode", default="proxy")
    if mode == "proxy":
        df_out = build_proxy_risk_score(df, cfg)
        return df_out, df_out["proxy_high_risk"], "proxy"
    elif mode == "real_label":
        label_col = cfg.get("target", "label_column")
        if label_col is None or label_col not in df.columns:
            raise ValueError(
                "config.yaml: target.mode is 'real_label' but target.label_column "
                f"('{label_col}') was not found in the data."
            )
        return df, df[label_col], "real_label"
    else:
        raise ValueError(f"Unknown target.mode '{mode}' in config.yaml")
