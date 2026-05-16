"""
scorecard.py
------------
Convert logistic regression log-odds to a points-based scorecard.

The scaling logic
-----------------
A scorecard translates predicted log-odds (the linear output of the
logistic regression) into a score on a human-readable scale (typically
300–850 for consumer credit, or 500–700 for internal scorecards).

The scaling is defined by three parameters:

    points0 : score assigned at the target odds
    odds0   : odds of good:bad at the target score (e.g. 50 means 50 goods
              for every 1 bad)
    pdo     : points to double the odds (how many score points correspond
              to halving the default probability)

From these, we derive factor and offset:

    factor = pdo / ln(2)
    offset = points0 - factor * ln(odds0)

Then for each observation:

    score = offset + factor * ln(odds)
          = offset - factor * log_odds_of_default

Why these specific parameters?
-------------------------------
points0 = 600 : industry convention inherited from FICO scaling.
                Scores cluster around 600 for an average-risk applicant.
odds0   = 50  : 50:1 good-to-bad odds at the target point. This means
                the target score of 600 corresponds to a ~2% default rate,
                appropriate for an emerging market micro-lending context.
pdo     = 20  : a 20-point increase doubles the odds of being a good
                borrower. This is a moderate sensitivity — lower PDO means
                the score is more sensitive to risk differences.

These are starting points, not fixed parameters. A production deployment
would calibrate odds0 to the observed portfolio bad rate.

Variable-level points
----------------------
The scorecard decomposes the total score into per-variable contributions.
For each variable and each bin:

    points_ij = -(WOE_ij * coef_j * factor) - (intercept/n_vars * factor)

This decomposition allows a loan officer to see exactly which attributes
of an applicant contributed to their score — a regulatory requirement in
many jurisdictions (adverse action explanation).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from optbinning import BinningProcess
import joblib
import os


# ---------------------------------------------------------------------------
# Scaling parameters — documented above
# ---------------------------------------------------------------------------
POINTS0 = 600    # target score
ODDS0 = 50       # good:bad odds at target score
PDO = 20         # points to double the odds


def compute_scaling_params(
    points0: int = POINTS0,
    odds0: int = ODDS0,
    pdo: int = PDO,
) -> tuple[float, float]:
    """
    Compute factor and offset from scorecard scaling parameters.

    Returns
    -------
    (factor, offset) tuple
    """
    factor = pdo / np.log(2)
    offset = points0 - factor * np.log(odds0)
    return factor, offset


def score_from_log_odds(
    log_odds: np.ndarray,
    factor: float,
    offset: float,
) -> np.ndarray:
    """
    Convert log-odds of default to scorecard points.

    Note the sign convention: higher log-odds of default = lower score.
    This matches consumer credit convention where higher score = better risk.
    """
    return offset - factor * log_odds


def build_scorecard_table(
    binning_process: BinningProcess,
    model: LogisticRegression,
    selected_vars: list[str],
    factor: float,
    offset: float,
) -> pd.DataFrame:
    """
    Build a per-variable, per-bin scorecard points table.

    This is the human-readable scorecard — for each variable, each bin
    gets a point contribution. A borrower's total score is the sum of
    base points plus one contribution per variable.

    Parameters
    ----------
    binning_process : fitted BinningProcess
    model : unpenalized logistic regression fitted on selected WOE variables
    selected_vars : variables in the model
    factor, offset : scaling parameters from compute_scaling_params

    Returns
    -------
    DataFrame with columns: variable, bin, woe, points
    """
    n_vars = len(selected_vars)
    intercept = model.intercept_[0]
    coefs = dict(zip(selected_vars, model.coef_[0]))

    records = []

    # Base points (intercept contribution, split evenly across variables)
    base_points_per_var = -(intercept / n_vars) * factor

    for var in selected_vars:
        # Strip _woe suffix if present to match BinningProcess names
        var_clean = var.replace("_woe", "")

        try:
            bt = binning_process.get_binning(var_clean)
            bin_table = bt.binning_table.build()
        except Exception:
            continue

        coef = coefs[var]

        for _, row in bin_table.iterrows():
            bin_label = row.get("Bin", row.get("bin", ""))
            woe_val = row.get("WoE", row.get("woe", np.nan))

            if pd.isna(woe_val) or str(bin_label) in ["Special", "Missing", "Totals"]:
                continue

            points = -(woe_val * coef * factor) + base_points_per_var

            records.append({
                "variable": var_clean,
                "bin": str(bin_label),
                "woe": round(woe_val, 4),
                "points": round(points, 1),
            })

    scorecard_df = pd.DataFrame(records)
    return scorecard_df


def score_applicant(
    applicant: dict,
    scorecard_df: pd.DataFrame,
    binning_process: BinningProcess,
    model: LogisticRegression,
    selected_vars: list[str],
    factor: float,
    offset: float,
) -> dict:
    """
    Score a single applicant given their raw feature values.

    This is the function called by the API endpoint.

    Parameters
    ----------
    applicant : dict of raw feature values (pre-engineering)
    scorecard_df : points table from build_scorecard_table
    binning_process : fitted binning
    model : fitted logistic regression
    selected_vars : model variables
    factor, offset : scaling parameters

    Returns
    -------
    dict with: score, probability_of_default, risk_band, score_breakdown
    """
    from src.binning import engineer_features

    # Convert to single-row DataFrame and engineer features
    df = pd.DataFrame([applicant])
    df = engineer_features(df)

    # Get available features
    available = [v.replace("_woe", "") for v in selected_vars if v.replace("_woe", "") in df.columns]

    if not available:
        raise ValueError("No matching features found in applicant data.")

    # Transform to WOE
    X = df[available]
    X_woe = binning_process.transform(X, metric="woe")

    # Align columns to model expectations
    woe_cols = [v for v in selected_vars if v in X_woe.columns or v.replace("_woe", "") in available]
    X_model = X_woe.reindex(columns=selected_vars, fill_value=0)

    # Predict
    log_odds = model.decision_function(X_model)[0]
    prob_default = model.predict_proba(X_model)[0][1]
    score = float(score_from_log_odds(np.array([log_odds]), factor, offset)[0])

    # Risk band
    if score >= 620:
        risk_band = "Low Risk"
    elif score >= 580:
        risk_band = "Medium Risk"
    else:
        risk_band = "High Risk"

    return {
        "score": round(score, 1),
        "probability_of_default": round(prob_default, 4),
        "risk_band": risk_band,
    }


def save_artifacts(
    binning_process: BinningProcess,
    model: LogisticRegression,
    selected_vars: list[str],
    scorecard_df: pd.DataFrame,
    factor: float,
    offset: float,
    output_dir: str = "api/model_artifacts",
) -> None:
    """Save all model artifacts needed for deployment."""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(binning_process, f"{output_dir}/binning_process.pkl")
    joblib.dump(model, f"{output_dir}/logistic_model.pkl")
    joblib.dump(selected_vars, f"{output_dir}/selected_vars.pkl")
    joblib.dump({"factor": factor, "offset": offset}, f"{output_dir}/scaling_params.pkl")
    scorecard_df.to_csv(f"{output_dir}/scorecard_table.csv", index=False)
    print(f"Artifacts saved to {output_dir}/")


def load_artifacts(artifact_dir: str = "api/model_artifacts") -> dict:
    """Load all model artifacts for inference."""
    return {
        "binning_process": joblib.load(f"{artifact_dir}/binning_process.pkl"),
        "model": joblib.load(f"{artifact_dir}/logistic_model.pkl"),
        "selected_vars": joblib.load(f"{artifact_dir}/selected_vars.pkl"),
        "scaling_params": joblib.load(f"{artifact_dir}/scaling_params.pkl"),
        "scorecard_df": pd.read_csv(f"{artifact_dir}/scorecard_table.csv"),
    }
