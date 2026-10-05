"""
Unsupervised risk segmentation.

With no ground-truth label available (see README), clustering is the only
technique in this project that can make a descriptive claim about "risk
structure" without assuming what risk means in advance. Used for exploratory
insight (notebook Section 13) and surfaced in the generated report — not fed
into the supervised pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import Config

CLUSTER_FEATURES = [
    "Age", "log_credit_amount", "Duration", "installment_proxy",
    "saving_ord", "checking_ord", "Job",
]


def select_k_via_silhouette(X: np.ndarray, cfg: Config) -> tuple[int, pd.DataFrame]:
    """Sweep k over the configured range and pick the silhouette-maximizing k.

    Returns the chosen k and a diagnostics DataFrame (k, inertia, silhouette)
    so the sweep can be plotted/reported rather than hidden inside the
    function.
    """
    lo, hi = cfg["clustering"]["k_search_range"]
    n_init = cfg["clustering"]["n_init"]
    random_state = cfg["project"]["random_state"]

    rows = []
    for k in range(lo, hi + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X, km.labels_),
        })
    diagnostics = pd.DataFrame(rows)
    best_k = int(diagnostics.loc[diagnostics["silhouette"].idxmax(), "k"])
    return best_k, diagnostics


def fit_segmentation(df: pd.DataFrame, cfg: Config) -> dict:
    """Fit a standardized KMeans segmentation and a 2D PCA projection.

    Returns a dict with the fitted `scaler`, `kmeans`, `pca`, cluster labels,
    the PCA projection, the k-selection diagnostics, and a per-cluster
    profile table (mean feature values + size) for reporting.
    """
    X_raw = df[CLUSTER_FEATURES].copy()
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    best_k, diagnostics = select_k_via_silhouette(X, cfg)

    random_state = cfg["project"]["random_state"]
    n_init = cfg["clustering"]["n_init"]
    kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=n_init).fit(X)
    labels = kmeans.labels_

    pca = PCA(n_components=2, random_state=random_state).fit(X)
    projection = pca.transform(X)

    profile = df.assign(cluster=labels).groupby("cluster")[CLUSTER_FEATURES].mean().round(2)
    profile["n_applicants"] = pd.Series(labels).value_counts().sort_index().values

    return {
        "scaler": scaler,
        "kmeans": kmeans,
        "pca": pca,
        "k": best_k,
        "k_diagnostics": diagnostics,
        "labels": labels,
        "projection": projection,
        "explained_variance_ratio": pca.explained_variance_ratio_.sum(),
        "cluster_profile": profile,
    }
