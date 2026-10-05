"""
Model evaluation: metrics, diagnostic plots, and cost-sensitive threshold
optimization.

Plot functions save to `config.yaml: paths.figures_dir` and return the figure
path, so `src/pipeline.py` can both display and catalogue them for the
generated report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)

from src.config import Config

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120, "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    })


def compute_test_metrics(y_test: pd.Series, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "brier_score": brier_score_loss(y_test, y_proba),
        "threshold_used": threshold,
        "classification_report": classification_report(
            y_test, y_pred, target_names=["low/med risk", "high risk"], output_dict=True
        ),
    }


def plot_diagnostic_curves(y_test: pd.Series, y_proba: np.ndarray, cfg: Config, model_name: str) -> Path:
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"AUC={roc_auc_score(y_test, y_proba):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(rec, prec, color=PALETTE[1], lw=2,
                 label=f"AP={average_precision_score(y_test, y_proba):.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=8)
    axes[2].plot(prob_pred, prob_true, marker="o", color=PALETTE[2], label=model_name)
    axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    axes[2].set_xlabel("Mean predicted probability"); axes[2].set_ylabel("Observed frequency")
    axes[2].set_title("Calibration Curve"); axes[2].legend()

    fig.suptitle(f"{model_name} — Test-Set Diagnostics", y=1.03)
    plt.tight_layout()
    out_path = cfg.path("figures_dir") / "diagnostic_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrix(y_test: pd.Series, y_proba: np.ndarray, threshold: float, cfg: Config) -> Path:
    _apply_style()
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["pred: low/med", "pred: high"],
                yticklabels=["true: low/med", "true: high"], ax=ax)
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
    plt.tight_layout()
    out_path = cfg.path("figures_dir") / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def compute_permutation_importance(pipe, X_test: pd.DataFrame, y_test: pd.Series, cfg: Config) -> pd.DataFrame:
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=20,
        random_state=cfg["project"]["random_state"], scoring="roc_auc", n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": list(X_test.columns),
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_feature_importance(imp_df: pd.DataFrame, cfg: Config, top_n: int = 10) -> Path:
    _apply_style()
    top = imp_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"][::-1], top["importance_mean"][::-1],
            xerr=top["importance_std"][::-1], color=PALETTE[4])
    ax.set_xlabel("Permutation importance (Δ ROC-AUC)")
    ax.set_title("Top Feature Importances")
    plt.tight_layout()
    out_path = cfg.path("figures_dir") / "feature_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def optimize_cost_sensitive_threshold(y_test: pd.Series, y_proba: np.ndarray, cfg: Config) -> dict:
    """Sweep the decision threshold to minimize expected cost under the
    asymmetric cost matrix in `config.yaml: cost_sensitive`, and save the
    cost-vs-threshold curve.

    See the module-level caveat in `src/proxy_scoring.py`: on the current
    proxy target the specific numbers are illustrative of the *mechanism*
    only, not a real business cost estimate.
    """
    fn_cost = cfg["cost_sensitive"]["false_negative_cost"]
    fp_cost = cfg["cost_sensitive"]["false_positive_cost"]
    default_threshold = cfg["modeling"]["decision_threshold_default"]

    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        costs.append(fn_cost * fn + fp_cost * fp)
    costs = np.array(costs)

    opt_idx = int(np.argmin(costs))
    opt_threshold = float(thresholds[opt_idx])

    default_pred = (y_proba >= default_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, default_pred).ravel()
    default_cost = fn_cost * fn + fp_cost * fp

    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, costs, color=PALETTE[0], lw=2)
    ax.axvline(opt_threshold, color=PALETTE[3], ls="--", lw=2,
               label=f"Cost-optimal threshold = {opt_threshold:.2f} (cost={costs[opt_idx]:.0f})")
    ax.axvline(default_threshold, color=PALETTE[5], ls=":", lw=2,
               label=f"Default threshold = {default_threshold:.2f} (cost={default_cost:.0f})")
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel(f"Expected cost ({fn_cost}×FN + {fp_cost}×FP)")
    ax.set_title("Cost-Sensitive Threshold Optimization")
    ax.legend()
    plt.tight_layout()
    fig_path = cfg.path("figures_dir") / "cost_threshold_curve.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    pct_improvement = (default_cost - costs[opt_idx]) / default_cost * 100 if default_cost else 0.0

    return {
        "optimal_threshold": opt_threshold,
        "optimal_cost": float(costs[opt_idx]),
        "default_threshold": default_threshold,
        "default_cost": float(default_cost),
        "pct_cost_improvement": float(pct_improvement),
        "figure_path": fig_path,
    }
