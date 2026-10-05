"""
Modeling: leakage-safe preprocessing pipeline, candidate models,
cross-validation, final training, and persistence.

The preprocessing (imputation, scaling, one-hot encoding) is wrapped inside a
single `sklearn.Pipeline` together with the classifier, so every
`fit`/`cross_validate` call fits preprocessing parameters only on the
training fold — the standard, non-negotiable guard against train/test
leakage through preprocessing statistics (e.g. scaling using the full
dataset's mean/std before splitting).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config


def get_feature_columns(cfg: Config) -> tuple[list[str], list[str]]:
    return cfg["features"]["numeric"], cfg["features"]["categorical"]


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Leakage-safe preprocessing: median imputation + standard scaling for
    numeric features, most-frequent imputation + one-hot encoding for
    categorical features. `handle_unknown="ignore"` prevents the pipeline
    from crashing on a categorical value seen at inference but not in the
    training fold.
    """
    return ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])


def get_candidate_models(cfg: Config) -> dict:
    rs = cfg["project"]["random_state"]
    m = cfg["modeling"]["models"]
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=m["logistic_regression"]["max_iter"], random_state=rs
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=m["random_forest"]["n_estimators"], random_state=rs
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
    }


def split_data(X: pd.DataFrame, y: pd.Series, cfg: Config):
    return train_test_split(
        X, y,
        test_size=cfg["modeling"]["test_size"],
        random_state=cfg["project"]["random_state"],
        stratify=y,
    )


def cross_validate_candidates(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: Config
) -> pd.DataFrame:
    """Cross-validate every candidate model with identical preprocessing and
    folds, returning a ranked comparison table (mean/std ROC-AUC, PR-AUC).
    """
    numeric_cols, categorical_cols = get_feature_columns(cfg)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    cv = StratifiedKFold(
        n_splits=cfg["modeling"]["cv_folds"], shuffle=True,
        random_state=cfg["project"]["random_state"],
    )

    rows = []
    for name, clf in get_candidate_models(cfg).items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=["roc_auc", "average_precision"], n_jobs=-1,
        )
        rows.append({
            "model": name,
            "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
            "cv_roc_auc_std": scores["test_roc_auc"].std(),
            "cv_pr_auc_mean": scores["test_average_precision"].mean(),
            "cv_pr_auc_std": scores["test_average_precision"].std(),
        })
    return pd.DataFrame(rows).sort_values(
        f"cv_{cfg['modeling']['primary_metric']}_mean", ascending=False
    ).reset_index(drop=True)


def train_best_model(
    best_model_name: str, X_train: pd.DataFrame, y_train: pd.Series, cfg: Config
) -> Pipeline:
    numeric_cols, categorical_cols = get_feature_columns(cfg)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    clf = get_candidate_models(cfg)[best_model_name]
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def save_model(pipe: Pipeline, cfg: Config, metadata: dict) -> None:
    model_path: Path = cfg.path("model_file")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_path)

    metadata_path: Path = cfg.path("preprocessor_metadata")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_model(cfg: Config) -> Pipeline:
    model_path: Path = cfg.path("model_file")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python -m src.pipeline` first."
        )
    return joblib.load(model_path)


def load_metadata(cfg: Config) -> dict:
    metadata_path: Path = cfg.path("preprocessor_metadata")
    with open(metadata_path, "r") as f:
        return json.load(f)
