"""
binning.py
----------
Weight-of-Evidence (WOE) binning with monotonicity constraints.

Design rationale
----------------
The original R pipeline used the `scorecard` package's interactive woebin_adj
tool, which requires manual human intervention for each variable and produces
bins stored as a hardcoded list. This has two problems:

1. Non-reproducibility: the pipeline cannot run end-to-end without a human.
2. Selection bias: manual adjustment introduces judgment calls that are hard
   to audit or defend.

This module uses OptBinning, which enforces monotonicity of the WOE curve
automatically via a constrained optimization. The result is bins that are:
- Monotone in WOE (higher risk = consistently higher/lower WOE direction)
- Reproducible: same data always produces same bins
- Defensible: the monotonicity constraint mirrors the domain assumption that
  creditworthiness signals should be directionally consistent

What WOE binning does
---------------------
For each predictor X and binary outcome Y (default=1):

    WOE_i = ln( P(X in bin_i | Y=1) / P(X in bin_i | Y=0) )

A positive WOE means the bin is overrepresented among defaulters.
A negative WOE means the bin is underrepresented among defaulters.

The Information Value (IV) summarizes the total discriminatory power of X:

    IV = sum_i (P(Y=1 in bin_i) - P(Y=0 in bin_i)) * WOE_i

IV thresholds (industry convention):
    < 0.02  : unpredictive — exclude
    0.02-0.1: weak
    0.1-0.3 : medium
    > 0.3   : strong (also check for data leakage)
"""

import pandas as pd
import numpy as np
from optbinning import BinningProcess
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Variable definitions — mirrors the cluster logic from the original pipeline
# ---------------------------------------------------------------------------

# These are the variable groups used for correlation screening and
# incremental model building. Each group captures a distinct dimension
# of creditworthiness signal available at application time.

VARIABLE_CLUSTERS = {
    "demographics": [
        "CODE_GENDER",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "AGE_YEARS",          # engineered from DAYS_BIRTH
    ],
    "employment": [
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "YEARS_EMPLOYED",      # engineered from DAYS_EMPLOYED
        "YEARS_ID_PUBLISH",    # engineered from DAYS_ID_PUBLISH
        "YEARS_REGISTRATION",  # engineered from DAYS_REGISTRATION
    ],
    "financials": [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "INCOME_CREDIT_RATIO",     # engineered
        "ANNUITY_INCOME_RATIO",    # engineered
        "CREDIT_GOODS_RATIO",      # engineered
    ],
    "housing_assets": [
        "NAME_HOUSING_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "CNT_FAM_MEMBERS",
    ],
    "document_compliance": [
        # Binary flags for documents submitted — analogous to photo submission
        # variables in the original thin-file lending pipeline. The inference is that
        # willingness to submit documentation is a behavioral signal of
        # creditworthiness, independent of the document content.
        "FLAG_DOCUMENT_3",
        "FLAG_DOCUMENT_6",
        "FLAG_DOCUMENT_8",
        "FLAG_DOCUMENT_16",
        "FLAG_DOCUMENT_18",
    ],
    "contact_reachability": [
        # Contact information completeness as a proxy for borrower stability
        # and willingness to be reached — analogous to phone/device variables.
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_CONT_MOBILE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
    ],
    "external_scores": [
        # Normalized scores from external sources — where available.
        # These are the closest thing to bureau data in this dataset.
        # Including them allows comparison of a bureau-augmented model
        # vs a proxy-only model.
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ],
    "region": [
        "REGION_POPULATION_RELATIVE",
        "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY",
        "REG_CITY_NOT_WORK_CITY",
        "REG_CITY_NOT_LIVE_CITY",
        "LIVE_CITY_NOT_WORK_CITY",
    ],
}

# All variables used in the pipeline (flat list)
ALL_VARIABLES = [v for cluster in VARIABLE_CLUSTERS.values() for v in cluster]

# Minimum IV to include a variable in model candidates
IV_THRESHOLD = 0.02

# Variables that should not be binned (target and ID)
SKIP_VARS = ["TARGET", "SK_ID_CURR"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw Home Credit fields.

    Days-based fields in Home Credit are stored as negative integers
    (days before application). We convert to positive years for
    interpretability.

    We also create financial ratios that capture capacity to repay
    relative to credit exposure — the core financial underwriting logic.
    """
    df = df.copy()

    # Age in years (DAYS_BIRTH is negative)
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365).round(1)

    # Years employed — DAYS_EMPLOYED has a known anomaly: value 365243
    # indicates pensioners/unemployed. We cap at 0 to avoid negative years.
    df["YEARS_EMPLOYED"] = np.where(
        df["DAYS_EMPLOYED"] == 365243,
        np.nan,
        (-df["DAYS_EMPLOYED"] / 365).round(1)
    )

    # Years since ID document was published
    df["YEARS_ID_PUBLISH"] = (-df["DAYS_ID_PUBLISH"] / 365).round(1)

    # Years since registration
    df["YEARS_REGISTRATION"] = (-df["DAYS_REGISTRATION"] / 365).round(1)

    # Financial ratios
    df["INCOME_CREDIT_RATIO"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)

    return df


def fit_binning(
    df: pd.DataFrame,
    target: str = "TARGET",
    monotonic_trend: str = "auto",
    min_bin_size: float = 0.05,
) -> BinningProcess:
    """
    Fit WOE binning on all candidate variables.

    Parameters
    ----------
    df : DataFrame with engineered features and target
    target : name of the binary target column
    monotonic_trend : passed to OptBinning. 'auto' detects direction
                      from data; 'auto_heuristic' is faster for large datasets
    min_bin_size : minimum fraction of observations per bin (guards against
                   bins driven by outliers)

    Returns
    -------
    Fitted BinningProcess object (can be serialized with joblib)
    """
    feature_names = [v for v in ALL_VARIABLES if v in df.columns and v not in SKIP_VARS]

    # Separate categorical and numerical for OptBinning
    categorical_vars = [
        v for v in feature_names
        if df[v].dtype == object or df[v].nunique() <= 10
    ]

    binning_process = BinningProcess(
        variable_names=feature_names,
        categorical_variables=categorical_vars,
        min_bin_size=min_bin_size,
        monotonic_trend=monotonic_trend,  # key improvement over original
    )

    X = df[feature_names]
    y = df[target]

    binning_process.fit(X, y)

    return binning_process


def get_iv_table(binning_process: BinningProcess) -> pd.DataFrame:
    """
    Extract Information Value for all binned variables, sorted descending.

    This is the primary variable selection input — variables below IV_THRESHOLD
    are excluded from model candidates before LASSO selection runs.
    """
    summary = binning_process.summary()

    iv_table = (
        summary[["name", "iv"]]
        .rename(columns={"name": "variable", "iv": "information_value"})
        .sort_values("information_value", ascending=False)
        .reset_index(drop=True)
    )

    iv_table["predictive_power"] = pd.cut(
        iv_table["information_value"],
        bins=[-np.inf, 0.02, 0.1, 0.3, np.inf],
        labels=["Unpredictive", "Weak", "Medium", "Strong"]
    )

    return iv_table


def transform_to_woe(
    binning_process: BinningProcess,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply fitted bins to transform raw features into WOE values.

    The resulting dataframe has one WOE column per variable, plus TARGET.
    This is the input to the logistic regression models.
    """
    feature_names = [v for v in ALL_VARIABLES if v in df.columns and v not in SKIP_VARS]
    X = df[feature_names]
    X_woe = binning_process.transform(X, metric="woe")

    # Keep only variables above IV threshold
    iv_table = get_iv_table(binning_process)
    good_vars = iv_table[iv_table["information_value"] >= IV_THRESHOLD]["variable"].tolist()
    woe_cols = [c for c in X_woe.columns if any(v in c for v in good_vars)]

    X_woe_filtered = X_woe[woe_cols].copy()
    X_woe_filtered["TARGET"] = df["TARGET"].values

    return X_woe_filtered
