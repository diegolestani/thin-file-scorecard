"""
app.py — Gradio interface for Hugging Face Spaces deployment
------------------------------------------------------------
This file is the entry point for Hugging Face Spaces.
HF Spaces expects either app.py (Gradio) or main.py (FastAPI) at root level.

The interface lets a user enter applicant characteristics and receive:
- A scorecard points total
- Estimated probability of default
- Risk band classification
- Per-variable score contributions (explainability)

Design note on explainability
------------------------------
A key regulatory requirement in credit decisions is adverse action explanation:
if an applicant is declined, they have a right to know which factors most
negatively affected their score. This interface demonstrates that capability
by showing per-variable point contributions, which factors helped the score,
and which factors hurt it. This is only possible because we use a WOE +
logistic regression architecture rather than a black-box model.
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Allow imports from src/ when running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.binning import engineer_features
from src.scorecard import score_from_log_odds

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
ARTIFACT_DIR = "api/model_artifacts"

def load_artifacts():
    try:
        return {
            "binning_process": joblib.load(f"{ARTIFACT_DIR}/binning_process.pkl"),
            "model": joblib.load(f"{ARTIFACT_DIR}/logistic_model.pkl"),
            "selected_vars": joblib.load(f"{ARTIFACT_DIR}/selected_vars.pkl"),
            "scaling_params": joblib.load(f"{ARTIFACT_DIR}/scaling_params.pkl"),
            "scorecard_df": pd.read_csv(f"{ARTIFACT_DIR}/scorecard_table.csv"),
        }
    except FileNotFoundError:
        return None

artifacts = load_artifacts()


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------
def score_applicant(
    age: float,
    years_employed: float,
    income: float,
    credit_amount: float,
    annuity: float,
    goods_price: float,
    gender: str,
    education: str,
    income_type: str,
    housing_type: str,
    owns_car: str,
    owns_realty: str,
    flag_document_3: bool,
    flag_document_6: bool,
    flag_mobil: bool,
    flag_email: bool,
    ext_source_2: float,
    ext_source_3: float,
):
    """Score a single applicant and return detailed breakdown."""

    if artifacts is None:
        return (
            "Model not yet trained",
            "Run the training notebook first to generate model artifacts.",
            pd.DataFrame(),
        )

    # Build raw applicant dict (matching Home Credit field names)
    applicant = {
        "DAYS_BIRTH": int(-age * 365),
        "DAYS_EMPLOYED": int(-years_employed * 365) if years_employed > 0 else 365243,
        "DAYS_ID_PUBLISH": int(-age * 365 * 0.3),   # proxy
        "DAYS_REGISTRATION": int(-age * 365 * 0.5), # proxy
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "CODE_GENDER": gender,
        "NAME_EDUCATION_TYPE": education,
        "NAME_INCOME_TYPE": income_type,
        "NAME_HOUSING_TYPE": housing_type,
        "FLAG_OWN_CAR": "Y" if owns_car == "Yes" else "N",
        "FLAG_OWN_REALTY": "Y" if owns_realty == "Yes" else "N",
        "FLAG_DOCUMENT_3": int(flag_document_3),
        "FLAG_DOCUMENT_6": int(flag_document_6),
        "FLAG_MOBIL": int(flag_mobil),
        "FLAG_EMAIL": int(flag_email),
        "EXT_SOURCE_2": ext_source_2,
        "EXT_SOURCE_3": ext_source_3,
        # Defaults for fields not in interface
        "FLAG_EMP_PHONE": 1,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 0,
        "FLAG_DOCUMENT_8": 0,
        "FLAG_DOCUMENT_16": 0,
        "FLAG_DOCUMENT_18": 0,
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 2.0,
        "REGION_POPULATION_RELATIVE": 0.02,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "REG_CITY_NOT_WORK_CITY": 0,
        "REG_CITY_NOT_LIVE_CITY": 0,
        "LIVE_CITY_NOT_WORK_CITY": 0,
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_FAMILY_STATUS": "Married",
        "WEEKDAY_APPR_PROCESS_START": "TUESDAY",
        "HOUR_APPR_PROCESS_START": 12,
        "LIVE_REGION_NOT_WORK_REGION": 0,
        "REG_REGION_NOT_LIVE_REGION": 0,
        "REG_REGION_NOT_WORK_REGION": 0,
    }

    try:
        bp = artifacts["binning_process"]
        model = artifacts["model"]
        selected_vars = artifacts["selected_vars"]
        scaling_params = artifacts["scaling_params"]
        scorecard_df = artifacts["scorecard_df"]
        factor = scaling_params["factor"]
        offset = scaling_params["offset"]

        # Engineer features
        df = pd.DataFrame([applicant])
        df = engineer_features(df)

        # Get available features
        available_raw = [v for v in selected_vars if v in df.columns]
        if not available_raw:
            return "Error", "No matching features in applicant profile.", pd.DataFrame()

        X = df[available_raw]
        X_woe = bp.transform(X, metric="woe")
        X_model = X_woe.reindex(columns=selected_vars, fill_value=0)

        # Predict
        log_odds = model.decision_function(X_model)[0]
        prob_default = model.predict_proba(X_model)[0][1]
        score = float(score_from_log_odds(np.array([log_odds]), factor, offset)[0])

        # Risk band
        if score >= 620:
            risk_band = "🟢 Low Risk"
            band_color = "low"
        elif score >= 560:
            risk_band = "🟡 Medium Risk"
            band_color = "medium"
        else:
            risk_band = "🔴 High Risk"
            band_color = "high"

        # Score summary text
        summary = (
            f"**Score: {score:.0f} points**\n\n"
            f"Risk Band: {risk_band}\n\n"
            f"Estimated Probability of Default: {prob_default:.1%}"
        )

        # Per-variable breakdown
        breakdown_records = []
        for var in available_raw:
            var_woe = X_woe[var].values[0] if var in X_woe.columns else 0
            var_bins = scorecard_df[scorecard_df["variable"] == var]
            # Find closest WOE match for points
            if len(var_bins) > 0:
                closest = var_bins.iloc[(var_bins["woe"] - var_woe).abs().argmin()]
                points_contribution = closest["points"]
                bin_label = closest["bin"]
            else:
                points_contribution = 0
                bin_label = "N/A"

            breakdown_records.append({
                "Variable": var,
                "Bin": bin_label,
                "WOE": round(var_woe, 3),
                "Points": round(points_contribution, 1),
            })

        breakdown_df = pd.DataFrame(breakdown_records).sort_values("Points")

        return summary, f"Score: {score:.0f}", breakdown_df

    except Exception as e:
        return f"Error: {str(e)}", "Could not compute score.", pd.DataFrame()


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Thin-File Credit Scorecard",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("""
    # Thin-File Credit Scorecard
    **Inference under data scarcity: scoring creditworthiness without credit history**

    This scorecard was built using Weight-of-Evidence (WOE) binning, LASSO-based
    variable selection, and logistic regression — the standard architecture for
    interpretable, regulatorily-defensible credit scoring.

    The model is trained on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
    dataset, which covers unbanked borrowers in Southeast Europe and Asia —
    a population with limited or no formal credit history. Alternative variables
    (behavioral signals, document compliance, employment proxies) substitute for
    bureau data.

    > **Methodology note:** This is a demonstration project. Threshold settings,
    > cost parameters, and score ranges should be calibrated to actual portfolio
    > data before production deployment. See the [GitHub repo](https://github.com/diegolestani/thin-file-scorecard)
    > for full documentation.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Demographics & Employment")
            age = gr.Slider(18, 70, value=35, step=1, label="Age (years)")
            years_employed = gr.Slider(0, 40, value=5, step=0.5, label="Years employed (0 = unemployed/pensioner)")
            gender = gr.Radio(["M", "F"], value="M", label="Gender")
            education = gr.Dropdown(
                ["Secondary / secondary special", "Higher education",
                 "Incomplete higher", "Lower secondary", "Academic degree"],
                value="Secondary / secondary special",
                label="Education level"
            )
            income_type = gr.Dropdown(
                ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"],
                value="Working",
                label="Income type"
            )

        with gr.Column(scale=1):
            gr.Markdown("### Financial Profile")
            income = gr.Number(value=150000, label="Annual income")
            credit_amount = gr.Number(value=500000, label="Loan amount requested")
            annuity = gr.Number(value=25000, label="Monthly annuity payment")
            goods_price = gr.Number(value=450000, label="Goods price")
            housing_type = gr.Dropdown(
                ["House / apartment", "Rented apartment", "With parents",
                 "Municipal apartment", "Office apartment", "Co-op apartment"],
                value="House / apartment",
                label="Housing type"
            )
            owns_car = gr.Radio(["Yes", "No"], value="No", label="Owns a car")
            owns_realty = gr.Radio(["Yes", "No"], value="No", label="Owns real estate")

        with gr.Column(scale=1):
            gr.Markdown("### Behavioral & External Signals")
            ext_source_2 = gr.Slider(0, 1, value=0.5, step=0.01,
                                      label="External score 2 (normalized, 0=high risk, 1=low risk)")
            ext_source_3 = gr.Slider(0, 1, value=0.5, step=0.01,
                                      label="External score 3 (normalized)")
            gr.Markdown("**Document submissions:**")
            flag_document_3 = gr.Checkbox(value=True, label="Document 3 submitted")
            flag_document_6 = gr.Checkbox(value=False, label="Document 6 submitted")
            gr.Markdown("**Contact information:**")
            flag_mobil = gr.Checkbox(value=True, label="Mobile phone registered")
            flag_email = gr.Checkbox(value=False, label="Email registered")

    score_btn = gr.Button("Compute Score", variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=1):
            summary_output = gr.Markdown(label="Score Summary")
        with gr.Column(scale=1):
            score_output = gr.Textbox(label="Score", interactive=False)

    gr.Markdown("### Score Breakdown by Variable")
    gr.Markdown("*Positive points improve the score (lower risk). Negative points reduce the score (higher risk).*")
    breakdown_output = gr.Dataframe(
        headers=["Variable", "Bin", "WOE", "Points"],
        label="Per-variable contributions",
    )

    score_btn.click(
        fn=score_applicant,
        inputs=[
            age, years_employed, income, credit_amount, annuity, goods_price,
            gender, education, income_type, housing_type,
            owns_car, owns_realty,
            flag_document_3, flag_document_6,
            flag_mobil, flag_email,
            ext_source_2, ext_source_3,
        ],
        outputs=[summary_output, score_output, breakdown_output],
    )

    gr.Markdown("""
    ---
    **About this project** | [GitHub](https://github.com/diegolestani/thin-file-scorecard) |
    Built by Diego Lestani · Methodology: WOE binning + LASSO selection + logistic regression scorecard
    """)

if __name__ == "__main__":
    demo.launch()
