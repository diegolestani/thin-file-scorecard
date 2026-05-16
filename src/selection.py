"""
selection.py
------------
Variable selection via LASSO logistic regression.

Why LASSO instead of stepwise-by-IV
-------------------------------------
The original pipeline selected variables by:
1. Ranking within clusters by IV (univariate)
2. Adding sequentially to models
3. Keeping significant coefficients from the final model

This is problematic for two reasons:

1. IV is univariate. It ignores multicollinearity — a variable with low IV
   can dominate a multivariate model if it is orthogonal to everything else.
   Conversely, a high-IV variable may contribute nothing after conditioning
   on correlated predictors.

2. Stepwise selection by p-value is statistically invalid: it inflates Type I
   error because the same data is used for selection and inference. The final
   p-values cannot be interpreted at face value.

LASSO (L1-penalized logistic regression) solves both problems:
- The penalty simultaneously considers all variables, respecting correlations
- The regularization path provides a principled selection mechanism
- Cross-validated lambda selection prevents overfitting

Post-LASSO refitting
--------------------
After LASSO selects variables, we refit an unpenalized logistic regression
on the selected subset. This recovers unbiased coefficient estimates and
valid standard errors for inference — LASSO coefficients are shrunk toward
zero by construction and should not be interpreted directly.

This two-stage approach (LASSO select → OLS/GLM refit) is standard in
applied econometrics (Belloni & Chernozhukov, 2013).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


def select_variables_lasso(
    X_woe: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    max_iter: int = 1000,
    random_state: int = 42,
) -> list[str]:
    """
    Use LASSO logistic regression with cross-validated lambda to select
    variables from the WOE-transformed feature matrix.

    Parameters
    ----------
    X_woe : WOE-transformed feature matrix (output of transform_to_woe)
    y : binary target (1 = default)
    cv : number of cross-validation folds for lambda selection
    max_iter : solver iterations (increase if convergence warnings appear)
    random_state : for reproducibility

    Returns
    -------
    List of selected variable names (non-zero LASSO coefficients)
    """
    # WOE features are already on a comparable scale (log-odds units),
    # but standardizing ensures the L1 penalty is applied uniformly
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_woe)

    lasso_cv = LogisticRegressionCV(
        Cs=20,                    # grid of 20 regularization strengths
        cv=cv,
        penalty="l1",
        solver="saga",            # saga handles L1 and large datasets
        scoring="roc_auc",        # optimize for discrimination, not accuracy
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )

    lasso_cv.fit(X_scaled, y)

    # Extract selected variables (non-zero coefficients)
    coef = lasso_cv.coef_[0]
    selected_mask = coef != 0
    selected_vars = X_woe.columns[selected_mask].tolist()

    print(f"  LASSO selected {len(selected_vars)} of {X_woe.shape[1]} variables")
    print(f"  Optimal C (inverse regularization): {lasso_cv.C_[0]:.4f}")

    return selected_vars


def refit_logistic(
    X_woe: pd.DataFrame,
    y: pd.Series,
    selected_vars: list[str],
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Refit unpenalized logistic regression on LASSO-selected variables.

    This recovers unbiased coefficient estimates after LASSO selection.
    The resulting model is what gets converted to a scorecard.

    Parameters
    ----------
    X_woe : full WOE feature matrix
    y : binary target
    selected_vars : variables selected by LASSO
    """
    X_selected = X_woe[selected_vars]

    model = LogisticRegression(
        penalty=None,        # no regularization — unpenalized refit
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(X_selected, y)

    return model


def get_coefficient_table(
    model: LogisticRegression,
    selected_vars: list[str],
) -> pd.DataFrame:
    """
    Return a clean table of model coefficients with odds ratios.

    In a logistic regression on WOE inputs, coefficients are not directly
    interpretable as in a standard logistic regression. The WOE
    transformation has already encoded the relationship between each bin
    and the outcome — the coefficients reflect residual marginal contributions
    after accounting for the WOE encoding.

    Odds ratios (exp(coef)) indicate direction:
    - OR > 1: variable increases default probability
    - OR < 1: variable decreases default probability
    """
    coef_df = pd.DataFrame({
        "variable": selected_vars,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0]),
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)

    coef_df["intercept"] = model.intercept_[0]

    return coef_df
