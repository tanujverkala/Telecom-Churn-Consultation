"""
DAY 5 — Streamlit Consultation Dashboard
===========================================
Project : Telecom Customer Retention Consultation Platform
Goal    : A real UI a retention agent could use — enter a customer's
          details, get a risk score, see WHY, get a suggested action.

DESIGN DECISION: this app loads the saved model artifacts directly
(models/xgb_model.joblib etc. from Day 4) rather than calling the FastAPI
backend over HTTP. Both architectures are valid — a separate API is more
"proper" for a multi-client production system, but for a single Streamlit
app, loading the model directly is simpler to run and deploy (one process
instead of two, which matters for Day 6's free-tier deployment). The
FastAPI backend from Day 4 still exists and still demonstrates you can
build a REST API around the model — this file demonstrates the model
serving a UI directly. Both are legitimate, common patterns.

Because this needs to run as a standalone process (not inside your Colab
session), it can't reuse in-memory variables like SCHEMA or X_train — it
redefines the small amount of category/column info it needs, and loads
everything else (the trained model, scaler, feature column order) from
the joblib files Day 4 saved.

HOW TO RUN THIS LOCALLY:
    streamlit run day5_dashboard.py
Then open the URL it prints (usually http://localhost:8501).

CONCEPTS THIS FILE USES:
  1. Streamlit basics — st.write, st.selectbox, st.slider, st.button,
     session-less scripts that RE-RUN TOP TO BOTTOM on every interaction
     (this is Streamlit's core execution model — worth understanding
     before reading further, since it explains why there's no explicit
     "on click" wiring: the whole script just re-executes)
  2. Reusing a trained model + scaler outside of the training script
  3. Basic rule-based recommendations layered on top of ML output
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import shap

# -----------------------------------------------------------------------
# Page config — must be the first Streamlit command in the script
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Retention Consultation Dashboard",
    page_icon="📞",
    layout="wide",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# -----------------------------------------------------------------------
# Category definitions — same source of truth as Day 4's SCHEMA, kept
# here explicitly since this script runs standalone (no shared session
# state with the notebook). If you add/change categories in your data
# audit's SCHEMA, mirror the change here too.
# -----------------------------------------------------------------------
CATEGORIES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": ["Bank transfer (automatic)", "Credit card (automatic)",
                       "Electronic check", "Mailed check"],
}
MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# -----------------------------------------------------------------------
# Load model artifacts ONCE per session, not on every interaction.
# @st.cache_resource tells Streamlit: run this function once, keep the
# result in memory, and reuse it across every re-run of the script
# (Streamlit re-runs the whole file top-to-bottom on every widget
# interaction — without caching, you'd reload the model from disk every
# single time someone moves a slider, which would make the app feel slow).
# -----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    return model, scaler, feature_columns, explainer


model, scaler, feature_columns, explainer = load_artifacts()


def encode_new_customer(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Same fix from Day 4: lock in known categories before one-hot
    encoding, so a single customer's row always produces the full set of
    dummy columns the model was trained on — see DAY4_NOTES.md for why
    this matters (a naive get_dummies on 1 row silently drops columns)."""
    d = raw_df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)

    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        d[col] = d[col].map({"Yes": 1, "No": 0})
    d["gender"] = d["gender"].map({"Male": 1, "Female": 0})

    for col in MULTI_CAT_COLS:
        d[col] = pd.Categorical(d[col], categories=CATEGORIES[col])
    d = pd.get_dummies(d, columns=MULTI_CAT_COLS, drop_first=True)

    bool_cols = d.select_dtypes(include="bool").columns
    d[bool_cols] = d[bool_cols].astype(int)
    d = d.reindex(columns=feature_columns, fill_value=0)
    return d


def get_recommendation(top_factors: pd.Series, risk_level: str) -> list[str]:
    """
    Simple RULE-BASED recommendations layered on top of the ML output.
    This is deliberately NOT another model — it's a lookup table mapping
    "this feature is a top driver of risk" to "here's a concrete action."
    This is what makes the dashboard a CONSULTATION tool instead of just
    a number: it turns "why" into "what to do about it."
    """
    if risk_level == "Low":
        return ["No action needed — customer shows strong retention signals."]

    actions = []
    top_feature_names = top_factors.index.tolist()

    if any("Contract" in f for f in top_feature_names) or "Contract_Two year" in top_feature_names:
        actions.append("Offer a discounted 1-year or 2-year contract upgrade — "
                        "longer contracts are the strongest retention factor in this model.")
    if "tenure" in top_feature_names:
        actions.append("This customer is relatively new — consider a loyalty "
                        "check-in call or an early-tenure engagement offer.")
    if "MonthlyCharges" in top_feature_names or "TotalCharges" in top_feature_names:
        actions.append("Review their plan for a cost-saving bundle or "
                        "loyalty discount — high charges are a key risk driver here.")
    if "InternetService_Fiber optic" in top_feature_names:
        actions.append("Check for service quality complaints — fiber customers "
                        "show elevated churn risk in this dataset, possibly "
                        "price or reliability related.")
    if "PaymentMethod_Electronic check" in top_feature_names:
        actions.append("Suggest switching to automatic payment (bank transfer "
                        "or credit card) — associated with lower churn risk.")

    if not actions:
        actions.append("Flag for manual review — risk is elevated but doesn't "
                        "match a standard playbook pattern.")
    return actions


# =============================================================================
# UI LAYOUT
# =============================================================================
st.title("📞 Telecom Retention Consultation Dashboard")
st.caption("Enter a customer's details to get a churn risk score, the "
           "factors driving it, and a suggested retention action.")

col_form, col_results = st.columns([1, 1.3])

with col_form:
    st.subheader("Customer Details")

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", CATEGORIES["gender"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", CATEGORIES["Partner"])
        dependents = st.selectbox("Has Dependents", CATEGORIES["Dependents"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with c2:
        contract = st.selectbox("Contract", CATEGORIES["Contract"])
        payment = st.selectbox("Payment Method", CATEGORIES["PaymentMethod"])
        paperless = st.selectbox("Paperless Billing", CATEGORIES["PaperlessBilling"])
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                         float(monthly_charges * max(tenure, 1)))

    st.markdown("**Services**")
    c3, c4, c5 = st.columns(3)
    with c3:
        phone = st.selectbox("Phone Service", CATEGORIES["PhoneService"])
        multiple_lines = st.selectbox("Multiple Lines", CATEGORIES["MultipleLines"])
        internet = st.selectbox("Internet Service", CATEGORIES["InternetService"])
    with c4:
        online_security = st.selectbox("Online Security", CATEGORIES["OnlineSecurity"])
        online_backup = st.selectbox("Online Backup", CATEGORIES["OnlineBackup"])
        device_protection = st.selectbox("Device Protection", CATEGORIES["DeviceProtection"])
    with c5:
        tech_support = st.selectbox("Tech Support", CATEGORIES["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", CATEGORIES["StreamingTV"])
        streaming_movies = st.selectbox("Streaming Movies", CATEGORIES["StreamingMovies"])

    run_prediction = st.button("Get Risk Assessment", type="primary", use_container_width=True)

with col_results:
    st.subheader("Risk Assessment")

    if run_prediction:
        customer = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multiple_lines,
            "InternetService": internet, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])

        encoded = encode_new_customer(customer)
        scaled = encoded.copy()
        scaled[NUMERIC_COLS] = scaler.transform(encoded[NUMERIC_COLS])

        proba = float(model.predict_proba(scaled)[0, 1])
        risk_level = "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low"
        risk_color = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}[risk_level]

        # --- Risk gauge ---
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            title={"text": f"Churn Risk — {risk_level}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 40], "color": "#eafaf1"},
                    {"range": [40, 70], "color": "#fef5e7"},
                    {"range": [70, 100], "color": "#fdedec"},
                ],
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # --- SHAP explanation for this customer ---
        shap_vals = explainer.shap_values(scaled)[0]
        contributions = pd.Series(shap_vals, index=scaled.columns)
        top5 = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(5)

        st.markdown("**Top factors driving this prediction**")
        factor_fig = go.Figure(go.Bar(
            x=top5.values,
            y=top5.index,
            orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in top5.values],
        ))
        factor_fig.update_layout(
            height=250, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Impact (red = increases risk, green = decreases risk)",
        )
        st.plotly_chart(factor_fig, use_container_width=True)

        # --- Recommended actions ---
        st.markdown("**Recommended actions**")
        for action in get_recommendation(top5, risk_level):
            st.info(action)

    else:
        st.info("Fill in the customer's details and click **Get Risk Assessment** "
                "to see their churn risk, the driving factors, and a suggested action.")
