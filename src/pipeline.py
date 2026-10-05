"""
End-to-end pipeline orchestrator.

Usage
-----
    python -m src.pipeline [--config config.yaml]

Runs, in order: schema-validated data load -> missing-data diagnosis and
treatment -> feature engineering -> unsupervised segmentation -> target
construction (proxy or real label, per config.yaml) -> model comparison via
cross-validation -> final model training -> test-set evaluation -> fairness
audit -> cost-sensitive threshold optimization -> persists the trained model
to `models/` and a markdown report + figures to `reports/`.

This script is intentionally a thin orchestration layer: every step calls
into a `src/` module that can also be imported and tested independently
(see `app/app.py` for an example of reusing `src.features` /
`src.modeling` outside of this pipeline).
"""

from __future__ import annotations

import argparse
import logging
import sys

from src import evaluation, fairness, features, modeling, preprocessing, proxy_scoring, report, segmentation
from src.config import load_config
from src.data_loader import load_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("pipeline")


def run(config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    cfg.ensure_output_dirs()

    # ---------------------------------------------------------------- 1. Load & validate
    logger.info("Step 1/8 — Loading and validating raw data")
    df_raw = load_raw_data(cfg)
    missingness = df_raw.isna().sum()

    # ---------------------------------------------------------------- 2. Missing-data treatment
    logger.info("Step 2/8 — Diagnosing and treating missing values")
    diagnostics = preprocessing.run_missingness_diagnostics(df_raw)
    for col, result in diagnostics.items():
        n_sig = int(result["significant_at_0.05"].sum())
        logger.info(
            "  Missingness in '%s': %d/%d predictors significantly associated "
            "(evidence against MCAR)", col, n_sig, len(result)
        )
    df_treated = preprocessing.treat_missing_accounts(df_raw, cfg)

    # ---------------------------------------------------------------- 3. Feature engineering
    logger.info("Step 3/8 — Engineering features")
    df_fe = features.engineer_features(df_treated, cfg)
    df_fe.to_parquet(cfg.path("processed_data"))
    logger.info("  Processed data saved to %s", cfg.path("processed_data"))

    # ---------------------------------------------------------------- 4. Unsupervised segmentation
    logger.info("Step 4/8 — Fitting unsupervised risk segmentation")
    seg = segmentation.fit_segmentation(df_fe, cfg)
    logger.info("  Selected k=%d (silhouette=%.3f)", seg["k"], seg["k_diagnostics"]["silhouette"].max())

    # ---------------------------------------------------------------- 5. Target construction
    logger.info("Step 5/8 — Constructing target variable (mode=%s)", cfg.get("target", "mode"))
    df_target, y, target_mode = proxy_scoring.get_target(df_fe, cfg)
    if target_mode == "proxy":
        logger.warning(
            "  Using heuristic PROXY target — downstream metrics validate pipeline "
            "correctness only, NOT real-world predictive performance. See README."
        )

    # ---------------------------------------------------------------- 6. Modeling
    logger.info("Step 6/8 — Cross-validating candidate models")
    numeric_cols, categorical_cols = modeling.get_feature_columns(cfg)
    X = df_target[numeric_cols + categorical_cols]
    X_train, X_test, y_train, y_test = modeling.split_data(X, y, cfg)

    cv_results = modeling.cross_validate_candidates(X_train, y_train, cfg)
    logger.info("\n%s", cv_results.round(3).to_string(index=False))

    best_model_name = cv_results.iloc[0]["model"]
    logger.info("  Training final model: %s", best_model_name)
    best_pipe = modeling.train_best_model(best_model_name, X_train, y_train, cfg)

    # ---------------------------------------------------------------- 7. Evaluation
    logger.info("Step 7/8 — Evaluating on held-out test set")
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluation.compute_test_metrics(y_test, y_proba)
    logger.info(
        "  Test ROC-AUC=%.3f | PR-AUC=%.3f | Brier=%.3f",
        test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["brier_score"],
    )
    evaluation.plot_diagnostic_curves(y_test, y_proba, cfg, best_model_name)
    evaluation.plot_confusion_matrix(y_test, y_proba, cfg["modeling"]["decision_threshold_default"], cfg)
    imp_df = evaluation.compute_permutation_importance(best_pipe, X_test, y_test, cfg)
    evaluation.plot_feature_importance(imp_df, cfg)

    cost_result = evaluation.optimize_cost_sensitive_threshold(y_test, y_proba, cfg)
    logger.info(
        "  Cost-optimal threshold=%.2f (%.1f%% cost reduction vs. default 0.50)",
        cost_result["optimal_threshold"], cost_result["pct_cost_improvement"],
    )

    y_pred_default = (y_proba >= cfg["modeling"]["decision_threshold_default"]).astype(int)
    sensitive_col = cfg["features"]["sensitive_attribute"]
    fairness_table = fairness.audit_by_sensitive_attribute(
        df_raw.loc[X_test.index, sensitive_col], y_proba, y_pred_default
    )
    logger.info("\n%s", fairness_table.to_string())

    # ---------------------------------------------------------------- 8. Persist model & report
    logger.info("Step 8/8 — Saving model artifact and report")
    metadata = {
        "model_name": best_model_name,
        "target_mode": target_mode,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "cv_results": cv_results.to_dict(orient="records"),
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "cost_sensitive_threshold": cost_result["optimal_threshold"],
        "long_duration_threshold": df_fe.attrs.get("long_duration_threshold"),
        "random_state": cfg["project"]["random_state"],
    }
    modeling.save_model(best_pipe, cfg, metadata)

    report_text = report.build_report(
        cfg=cfg, n_rows=len(df_raw), missingness=missingness[missingness > 0],
        cv_results=cv_results, best_model_name=best_model_name, test_metrics=test_metrics,
        cost_result=cost_result, fairness_table=fairness_table,
        segmentation_k=seg["k"], segmentation_silhouette=seg["k_diagnostics"]["silhouette"].max(),
    )
    report_path = report.save_report(report_text, cfg)
    logger.info("  Model saved to %s", cfg.path("model_file"))
    logger.info("  Report saved to %s", report_path)
    logger.info("Pipeline complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the German Credit Risk pipeline end-to-end.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config.yaml override.")
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
