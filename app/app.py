"""
Streamlit scoring app.

Loads the model artifact trained by `src/pipeline.py` and scores a single
applicant entered through a form. Deliberately reuses `src.features.engineer_features`
and the saved `long_duration_threshold` from training — NOT a reimplementation
of the feature logic — so the app can never silently drift out of sync with
how the model was trained (the single most common bug class in ML demo apps).

Run with:
    streamlit run app/app.py
(from the project root, with the project root on PYTHONPATH — see the sys.path
shim below, or `pip install -e .` if you add a pyproject.toml).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src import features, modeling, preprocessing
from src.config import load_config

st.set_page_config(page_title="German Credit — Proxy Risk Scorer", page_icon="🏦", layout="centered")


@st.cache_resource
def _load_artifacts():
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    model = modeling.load_model(cfg)
    metadata = modeling.load_metadata(cfg)
    return cfg, model, metadata


def main() -> None:
    st.title("🏦 German Credit — Risk Scoring Demo")

    st.warning(
        "**Methodology demonstration, not a real credit decision tool.** "
        "The underlying dataset has no true default label. The score below is "
        "produced by a documented heuristic proxy target (see `src/proxy_scoring.py` "
        "and the project README). Do not use this output to make or justify any "
        "real lending decision.",
        icon="⚠️",
    )

    try:
        cfg, model, metadata = _load_artifacts()
    except FileNotFoundError as e:
        st.error(f"{e}\n\nRun `python -m src.pipeline` from the project root first.")
        st.stop()

    st.caption(
        f"Model: **{metadata['model_name']}** · Target mode: **{metadata['target_mode']}** · "
        f"Held-out test ROC-AUC: **{metadata['test_metrics']['roc_auc']:.3f}** "
        "(inflated by proxy-target circularity — see warning above)"
    )

    st.subheader("Applicant details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox(
            "Job skill level", options=[0, 1, 2, 3],
            format_func=lambda j: f"{j} — {cfg['schema']['job_labels'][j]}", index=2,
        )
        housing = st.selectbox("Housing", options=["own", "rent", "free"])
        purpose = st.selectbox(
            "Loan purpose",
            options=["radio/TV", "education", "furniture/equipment", "car",
                     "business", "domestic appliances", "repairs", "vacation/others"],
        )
        sex = st.selectbox(
            "Sex", options=["male", "female"],
            help="Not used as a model input (excluded as a protected attribute). "
                 "Shown here only to mirror the source schema.",
        )
    with col2:
        credit_amount = st.number_input("Credit amount (DM)", min_value=100, max_value=30000, value=3000, step=100)
        duration = st.number_input("Duration (months)", min_value=1, max_value=96, value=24)
        saving = st.selectbox("Saving account status", options=["none", "little", "moderate", "quite rich", "rich"])
        checking = st.selectbox("Checking account status", options=["none", "little", "moderate", "rich"])

    if st.button("Score applicant", type="primary"):
        applicant = pd.DataFrame([{
            "Age": age, "Sex": sex, "Job": job, "Housing": housing,
            "Saving accounts": None if saving == "none" else saving,
            "Checking account": None if checking == "none" else checking,
            "Credit amount": credit_amount, "Duration": duration, "Purpose": purpose,
        }], index=[0])

        df_treated = preprocessing.treat_missing_accounts(applicant, cfg)
        df_fe = features.engineer_features(
            df_treated, cfg, long_duration_threshold=metadata["long_duration_threshold"]
        )

        numeric_cols = metadata["numeric_features"]
        categorical_cols = metadata["categorical_features"]
        X = df_fe[numeric_cols + categorical_cols]

        proba = float(model.predict_proba(X)[0, 1])
        cost_threshold = metadata["cost_sensitive_threshold"]

        st.subheader("Result")
        st.metric("Predicted proxy high-risk probability", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

        if proba >= cost_threshold:
            st.error(
                f"Above the cost-optimal threshold ({cost_threshold:.2f}) from Section 5 of the "
                f"pipeline report — would be flagged 'high risk' under the illustrative 5:1 cost matrix."
            )
        else:
            st.success(
                f"Below the cost-optimal threshold ({cost_threshold:.2f}) — "
                f"would be flagged 'low/medium risk'."
            )

        with st.expander("Engineered features used by the model"):
            st.dataframe(X.T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
