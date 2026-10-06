"""
Lightweight test suite covering the correctness-critical, easy-to-silently-break
parts of the pipeline: schema validation, the missing-data treatment decision,
feature-engineering determinism (train/serve skew prevention), and the proxy
score's documented invariants.

Run with:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import features, preprocessing, proxy_scoring
from src.config import load_config
from src.data_loader import SchemaValidationError, validate_schema


@pytest.fixture(scope="module")
def cfg():
    return load_config(PROJECT_ROOT / "config.yaml")


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [25, 40, 60],
        "Sex": ["male", "female", "male"],
        "Job": [2, 3, 1],
        "Housing": ["own", "rent", "free"],
        "Saving accounts": ["little", None, "rich"],
        "Checking account": [None, "moderate", None],
        "Credit amount": [1200, 5000, 800],
        "Duration": [12, 36, 6],
        "Purpose": ["radio/TV", "car", "repairs"],
    })


# ----------------------------------------------------------------------------------
# Schema validation
# ----------------------------------------------------------------------------------

def test_schema_validation_passes_on_valid_data(sample_df, cfg):
    validate_schema(sample_df, cfg)  # should not raise


def test_schema_validation_catches_missing_column(sample_df, cfg):
    bad = sample_df.drop(columns=["Age"])
    with pytest.raises(SchemaValidationError, match="Age"):
        validate_schema(bad, cfg)


def test_schema_validation_catches_implausible_duration(sample_df, cfg):
    bad = sample_df.copy()
    bad.loc[0, "Duration"] = -5
    with pytest.raises(SchemaValidationError, match="Duration"):
        validate_schema(bad, cfg)


# ----------------------------------------------------------------------------------
# Missing-data treatment
# ----------------------------------------------------------------------------------

def test_missing_accounts_become_explicit_none_category(sample_df, cfg):
    treated = preprocessing.treat_missing_accounts(sample_df, cfg)
    assert treated.isna().sum().sum() == 0
    assert "none" in treated["Saving accounts"].cat.categories
    assert treated.loc[1, "Saving accounts"] == "none"
    assert treated.loc[0, "Checking account"] == "none"


def test_account_categories_are_ordered(sample_df, cfg):
    treated = preprocessing.treat_missing_accounts(sample_df, cfg)
    assert treated["Saving accounts"].cat.ordered
    assert treated["Checking account"].cat.ordered
    assert list(treated["Saving accounts"].cat.categories) == cfg["schema"]["saving_account_order"]


# ----------------------------------------------------------------------------------
# Feature engineering determinism (guards against train/serve skew)
# ----------------------------------------------------------------------------------

def test_engineer_features_is_deterministic_given_same_threshold(sample_df, cfg):
    treated = preprocessing.treat_missing_accounts(sample_df, cfg)
    fe_1 = features.engineer_features(treated, cfg, long_duration_threshold=24.0)
    fe_2 = features.engineer_features(treated, cfg, long_duration_threshold=24.0)
    pd.testing.assert_frame_equal(fe_1, fe_2)


def test_engineer_features_single_row_matches_batch_threshold(sample_df, cfg):
    """The exact bug class this test guards against: scoring a single applicant
    (app/app.py) must use the *training-set* long-duration threshold, not
    recompute a quantile from a dataset of one row (which would always flag
    long_duration_flag=1, since a single value is always >= its own quantile).
    """
    treated = preprocessing.treat_missing_accounts(sample_df, cfg)
    batch_fe = features.engineer_features(treated, cfg)
    threshold = batch_fe.attrs["long_duration_threshold"]

    single_applicant = sample_df.iloc[[0]]
    single_treated = preprocessing.treat_missing_accounts(single_applicant, cfg)
    single_fe = features.engineer_features(single_treated, cfg, long_duration_threshold=threshold)

    expected_flag = int(sample_df.iloc[0]["Duration"] >= threshold)
    assert single_fe.iloc[0]["long_duration_flag"] == expected_flag


# ----------------------------------------------------------------------------------
# Proxy risk score invariants
# ----------------------------------------------------------------------------------

def test_proxy_weights_sum_to_one(cfg):
    weights = cfg.get("target", "proxy", "weights")
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_proxy_score_is_bounded_and_target_is_binary(sample_df, cfg):
    treated = preprocessing.treat_missing_accounts(sample_df, cfg)
    fe = features.engineer_features(treated, cfg)
    scored = proxy_scoring.build_proxy_risk_score(fe, cfg)
    assert scored["proxy_risk_score"].between(0, 1).all()
    assert set(scored["proxy_high_risk"].unique()).issubset({0, 1})


def test_proxy_score_excludes_sex(cfg):
    """`Sex` must never appear in the proxy formula's component inputs —
    this is a fairness requirement, not just a style choice."""
    import inspect
    source = inspect.getsource(proxy_scoring.build_proxy_risk_score)
    assert '"Sex"' not in source and "['Sex']" not in source
