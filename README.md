# Telecom Customer Retention Consultation Platform

A machine learning tool that predicts telecom customer churn risk and
explains *why* — turning a black-box prediction into an actionable
consultation an agent can use, not just a number.

**Live app:** _add your Streamlit Community Cloud URL here once deployed_

## What it does

Given a customer's account details (contract type, tenure, charges,
services), the dashboard returns:
- A churn risk score (0–100%) with a Low/Medium/High rating
- The specific factors driving *that customer's* risk, via SHAP
  explainability — not just global feature importance, but a
  per-customer breakdown
- A rule-based recommended action (e.g. contract upgrade offer, loyalty
  outreach) based on which factors are driving the risk

## Why this problem

Acquiring a new telecom customer costs 5–7x more than retaining an
existing one. Retention teams need to know not just *who* is likely to
leave, but *why*, so they can act with a specific, relevant offer instead
of a generic one. This project builds that decision-support layer.

## Dataset

IBM's public Telco Customer Churn dataset — 7,043 customers, 21 features
(demographics, contract details, services subscribed, billing info).

## Approach

1. **Data audit** — schema validation to catch data quality issues before
   any analysis (e.g. a blank-string bug in `TotalCharges` affecting new
   customers with 0 tenure).
2. **EDA** — identified contract type and tenure as the strongest visible
   churn drivers (month-to-month customers churn at 42.7% vs. 2.8% for
   two-year contracts).
3. **Modeling** — compared Logistic Regression, Random Forest, and
   XGBoost. All three land around ROC-AUC 0.84; the real difference is
   each model's precision/recall tradeoff, which maps to a real business
   decision (cost of a false alarm vs. cost of a missed churner).
4. **Explainability** — SHAP values layered on the XGBoost model, both
   globally (which features matter most overall) and locally (why this
   one customer scored the way they did).
5. **Serving** — a FastAPI backend (`/predict` endpoint) demonstrating a
   production-style REST API, and a Streamlit dashboard as the actual
   user-facing consultation tool.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.806 | 0.659 | 0.559 | 0.605 | 0.842 |
| Random Forest | 0.759 | 0.532 | 0.778 | 0.632 | 0.843 |
| XGBoost | 0.752 | 0.521 | 0.794 | 0.629 | 0.842 |

Top predictive features (consistent across EDA, feature importance, and
SHAP): **contract length, tenure, and monthly/total charges.**

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, FastAPI, Streamlit, Plotly

## Running it locally

```
pip install -r requirements.txt
streamlit run day5_dashboard.py
```
Requires the `models/` folder (trained model, scaler, feature column
list) to sit alongside `day5_dashboard.py`.

## Project structure

```
├── day5_dashboard.py       # Streamlit consultation dashboard
├── requirements.txt
├── models/
│   ├── xgb_model.joblib
│   ├── scaler.joblib
│   └── feature_columns.joblib
└── README.md
```

## A note on data leakage / real-world use

This model is trained on historical churn outcomes to predict *future*
risk for *current* customers — the same customer's outcome is never
known in advance in real use, so predictions are always genuinely
prospective, not a lookup of already-known results.
