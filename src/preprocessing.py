"""
Missing-data diagnosis and treatment.

Encodes the decision process from the project's EDA notebook (Section 7) as
reusable, testable functions: test whether missingness looks random (MCAR)
before deciding how to treat it, rather than imputing by default.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from src.config import Config


def missingness_association_test(
    df: pd.DataFrame,
    target_col: str,
    numeric_predictors: list[str],
    categorical_predictors: list[str],
) -> pd.DataFrame:
    """Test whether missingness in `target_col` is associated with other
    observed columns (Mann-Whitney U for numeric, Chi-square for categorical).

    A low p-value on any predictor is evidence against the data being
    Missing Completely At Random (MCAR) for `target_col`, which argues against
    naive mode/mean imputation and for treating "missing" as its own
    informative category.
    """
    is_missing = df[target_col].isna()
    rows = []
    for col in numeric_predictors:
        u, p = stats.mannwhitneyu(
            df.loc[is_missing, col], df.loc[~is_missing, col], alternative="two-sided"
        )
        rows.append({"predictor": col, "test": "Mann-Whitney U", "statistic": u, "p_value": p})
    for col in categorical_predictors:
        ct = pd.crosstab(df[col], is_missing)
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append({"predictor": col, "test": "Chi-square", "statistic": chi2, "p_value": p})

    out = pd.DataFrame(rows)
    out["significant_at_0.05"] = out["p_value"] < 0.05
    return out.sort_values("p_value").reset_index(drop=True)


def run_missingness_diagnostics(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the missingness-association test for every column with missing
    values, using the remaining columns as predictors. Returns a dict keyed
    by column name, suitable for inclusion in a generated report.
    """
    numeric_cols = [c for c in ("Age", "Credit amount", "Duration") if c in df.columns]
    categorical_cols = [c for c in ("Housing", "Purpose", "Sex") if c in df.columns]

    results = {}
    for col in df.columns[df.isna().any()]:
        num_preds = [c for c in numeric_cols if c != col]
        cat_preds = [c for c in categorical_cols if c != col]
        results[col] = missingness_association_test(df, col, num_preds, cat_preds)
    return results


def treat_missing_accounts(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply the project's documented missingness treatment:

    `NA` in `Saving accounts` / `Checking account` is encoded as an explicit
    "none" category (meaning "applicant holds no account of this type")
    rather than imputed with the mode. See Section 7 of the EDA notebook for
    the statistical justification; this is a documented ASSUMPTION, not a
    verified fact about the source system (see README limitations).
    """
    out = df.copy()

    saving_order = cfg["schema"]["saving_account_order"]
    checking_order = cfg["schema"]["checking_account_order"]

    out["Saving accounts"] = out["Saving accounts"].fillna("none")
    out["Checking account"] = out["Checking account"].fillna("none")

    out["Saving accounts"] = pd.Categorical(
        out["Saving accounts"], categories=saving_order, ordered=True
    )
    out["Checking account"] = pd.Categorical(
        out["Checking account"], categories=checking_order, ordered=True
    )

    job_labels = {int(k): v for k, v in cfg["schema"]["job_labels"].items()}
    out["Job_label"] = out["Job"].map(job_labels)

    residual_na = out.isna().sum().sum()
    if residual_na:
        raise ValueError(f"Unexpected residual missing values after treatment: {residual_na}")

    return out
