"""
evaluation.py
-------------
Model evaluation: discrimination, calibration, stability.

Three distinct questions
------------------------
1. DISCRIMINATION: Does the model correctly rank borrowers by risk?
   Measured by AUC (Area Under the ROC Curve).
   AUC = 0.5 → random; AUC = 1.0 → perfect ranking.

2. CALIBRATION: Are the predicted probabilities accurate in absolute terms?
   A model can have excellent AUC but poor calibration — it might rank
   borrowers correctly while systematically overestimating or underestimating
   default rates.
   Measured by: reliability diagrams, Hosmer-Lemeshow test, Brier score.

   This was missing from the scorecard implementations. For a lending scorecard,
   calibration matters because:
   - Pricing decisions depend on expected loss = PD × LGD × EAD
   - If PD is systematically biased, pricing is wrong even if ranking is right

3. STABILITY: Does the score distribution shift over time?
   Measured by Population Stability Index (PSI).
   PSI < 0.10 → stable; 0.10-0.25 → some shift; > 0.25 → significant drift.

   In production, PSI would be computed monthly on new applicant cohorts
   against the development population. Here we demonstrate it on a
   time-based holdout split.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    brier_score_loss, log_loss,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Discrimination
# ---------------------------------------------------------------------------

def cross_validated_auc(
    X_woe: pd.DataFrame,
    y: pd.Series,
    selected_vars: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Estimate AUC via stratified k-fold cross-validation.

    Stratified folds preserve the class imbalance ratio in each fold,
    which matters when the positive class (default) is rare.

    Returns
    -------
    dict with mean_auc, std_auc, fold_aucs
    """
    X = X_woe[selected_vars]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_aucs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        fold_aucs.append(auc)

        print(f"  Fold {fold_idx + 1}: AUC = {auc:.4f}")

    result = {
        "mean_auc": np.mean(fold_aucs),
        "std_auc": np.std(fold_aucs),
        "fold_aucs": fold_aucs,
        "n_folds": n_folds,
    }

    print(f"\n  Mean AUC: {result['mean_auc']:.4f} ± {result['std_auc']:.4f}")
    return result


def plot_roc_curve(
    y_true: pd.Series,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    ax=None,
) -> plt.Axes:
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    return ax


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------

def hosmer_lemeshow_test(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Hosmer-Lemeshow goodness-of-fit test.

    Groups observations into deciles of predicted probability and compares
    observed vs expected default rates within each group using a chi-squared test.

    H0: the model is well-calibrated (predicted probabilities match observed rates)
    H1: there is significant miscalibration

    A p-value > 0.05 means we cannot reject H0 — the model is acceptably calibrated.
    A p-value < 0.05 indicates significant miscalibration.

    Note: HL test has known limitations — it is sensitive to sample size
    (large samples almost always reject H0) and to the choice of n_bins.
    Always interpret alongside the reliability diagram.
    """
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["decile"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")

    grouped = df.groupby("decile").agg(
        observed=("y_true", "sum"),
        expected=("y_prob", "sum"),
        total=("y_true", "count"),
    )

    # Chi-squared statistic
    hl_stat = (
        (grouped["observed"] - grouped["expected"]) ** 2 /
        (grouped["expected"] * (1 - grouped["expected"] / grouped["total"]) + 1e-10)
    ).sum()

    df_dof = len(grouped) - 2
    p_value = 1 - stats.chi2.cdf(hl_stat, df=df_dof)

    return {
        "hl_statistic": round(hl_stat, 4),
        "degrees_of_freedom": df_dof,
        "p_value": round(p_value, 4),
        "interpretation": "Well calibrated (p > 0.05)" if p_value > 0.05 else "Miscalibrated (p < 0.05)",
        "grouped_table": grouped.reset_index(),
    }


def plot_reliability_diagram(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
    ax=None,
) -> plt.Axes:
    """
    Reliability diagram (calibration plot).

    Plots observed default rate vs mean predicted probability within
    equal-frequency bins. A perfectly calibrated model lies on the diagonal.

    Systematic deviations indicate:
    - Points above diagonal: model underestimates risk
    - Points below diagonal: model overestimates risk
    - S-shape: probability compression (common with logistic regression)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df["bin"] = pd.qcut(df["y_prob"], q=n_bins, duplicates="drop")

    cal = df.groupby("bin").agg(
        mean_pred=("y_prob", "mean"),
        obs_rate=("y_true", "mean"),
        count=("y_true", "count"),
    ).reset_index()

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect calibration")
    ax.scatter(
        cal["mean_pred"], cal["obs_rate"],
        s=cal["count"] / cal["count"].max() * 200,
        color="steelblue", alpha=0.8, zorder=5
    )
    ax.plot(cal["mean_pred"], cal["obs_rate"], color="steelblue", lw=1.5, alpha=0.6)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Default Rate")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.legend()
    ax.set_xlim([0, max(cal["mean_pred"].max() * 1.1, 0.3)])
    ax.set_ylim([0, max(cal["obs_rate"].max() * 1.1, 0.3)])

    return ax


def calibration_summary(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    """Summary of calibration metrics."""
    return {
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "mean_predicted_prob": round(y_prob.mean(), 4),
        "observed_default_rate": round(y_true.mean(), 4),
        "prediction_bias": round(y_prob.mean() - y_true.mean(), 4),
    }


# ---------------------------------------------------------------------------
# 3. Stability (PSI)
# ---------------------------------------------------------------------------

def population_stability_index(
    scores_dev: np.ndarray,
    scores_val: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Population Stability Index.

    Compares the score distribution between a development (training) population
    and a validation (holdout or new cohort) population.

    PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i)

    where expected_i is the share of development population in bin i,
    and actual_i is the share of validation population in bin i.

    Thresholds:
        PSI < 0.10  : No significant shift — model is stable
        0.10-0.25   : Moderate shift — investigate
        PSI > 0.25  : Significant shift — model should be recalibrated

    In production: compute monthly using new applicant cohorts vs the
    development population used to build the model.
    """
    # Define bins on development population
    breakpoints = np.percentile(scores_dev, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    def get_distribution(scores, breakpoints):
        counts = np.histogram(scores, bins=breakpoints)[0]
        # Clip to avoid log(0) — add small constant to empty bins
        proportions = np.maximum(counts / len(scores), 1e-6)
        return proportions

    expected = get_distribution(scores_dev, breakpoints)
    actual = get_distribution(scores_val, breakpoints)

    psi_bins = (actual - expected) * np.log(actual / expected)
    psi = psi_bins.sum()

    if psi < 0.10:
        interpretation = "Stable (PSI < 0.10)"
    elif psi < 0.25:
        interpretation = "Moderate shift (0.10 ≤ PSI < 0.25) — monitor"
    else:
        interpretation = "Significant shift (PSI ≥ 0.25) — recalibrate model"

    return {
        "psi": round(psi, 4),
        "interpretation": interpretation,
        "bin_contributions": psi_bins.tolist(),
    }


# ---------------------------------------------------------------------------
# 4. Cost-weighted threshold optimization
# ---------------------------------------------------------------------------

def optimize_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> dict:
    """
    Find the classification threshold that minimizes total misclassification cost.

    Parameters
    ----------
    cost_fp : cost of a false positive (rejecting a good borrower)
              This is the opportunity cost — foregone interest income
    cost_fn : cost of a false negative (approving a bad borrower)
              This is the expected loss — principal × LGD

    In a thin-file lending context, cost_fn >> cost_fp because:
    - Approving a bad loan loses the principal (high cost)
    - Rejecting a good loan loses the margin (lower cost)

    A reasonable starting ratio for micro-lending: cost_fn / cost_fp ≈ 3-5x.
    This should be calibrated to actual portfolio loss data in production.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []

    n_neg = (y_true == 0).sum()
    n_pos = (y_true == 1).sum()

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        total_cost = fp * cost_fp + fn * cost_fn
        costs.append(total_cost)

    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    optimal_cost = costs[optimal_idx]

    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_optimal)
    tn, fp, fn, tp = cm.ravel()

    return {
        "optimal_threshold": round(optimal_threshold, 3),
        "total_cost_at_threshold": round(optimal_cost, 2),
        "true_positive_rate": round(tp / (tp + fn), 4),
        "false_positive_rate": round(fp / (fp + tn), 4),
        "approval_rate": round((tp + fp) / len(y_true), 4),
        "cost_fp_assumed": cost_fp,
        "cost_fn_assumed": cost_fn,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def full_evaluation_report(
    y_true: pd.Series,
    y_prob: np.ndarray,
    scores: np.ndarray,
    scores_dev: np.ndarray = None,
    cost_fp: float = 100,
    cost_fn: float = 300,
    save_path: str = None,
) -> dict:
    """
    Generate a complete evaluation report with all plots.

    Parameters
    ----------
    y_true : observed outcomes on test set
    y_prob : predicted default probabilities on test set
    scores : scorecard points on test set
    scores_dev : scores on training set (for PSI)
    cost_fp : false positive cost
    cost_fn : false negative cost (default 3x FP for micro-lending)
    save_path : if provided, save the figure to this path

    Returns
    -------
    dict of all evaluation metrics
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Thin-File Scorecard — Evaluation Report", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ROC curve
    ax1 = fig.add_subplot(gs[0, 0])
    plot_roc_curve(y_true, y_prob, title="ROC Curve", ax=ax1)

    # Reliability diagram
    ax2 = fig.add_subplot(gs[0, 1])
    plot_reliability_diagram(y_true, y_prob, ax=ax2)

    # Score distribution by outcome
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(scores[y_true == 0], bins=30, alpha=0.6, color="steelblue", label="Good (no default)", density=True)
    ax3.hist(scores[y_true == 1], bins=30, alpha=0.6, color="firebrick", label="Bad (default)", density=True)
    ax3.set_xlabel("Scorecard Points")
    ax3.set_ylabel("Density")
    ax3.set_title("Score Distribution by Outcome")
    ax3.legend(fontsize=8)

    # Cost curve
    ax4 = fig.add_subplot(gs[1, 0])
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        costs.append(fp * cost_fp + fn * cost_fn)
    ax4.plot(thresholds, costs, color="darkorange", lw=2)
    opt_t = thresholds[np.argmin(costs)]
    ax4.axvline(opt_t, color="black", linestyle="--", alpha=0.7, label=f"Optimal: {opt_t:.2f}")
    ax4.set_xlabel("Classification Threshold")
    ax4.set_ylabel("Total Misclassification Cost")
    ax4.set_title("Cost-Weighted Threshold Optimization")
    ax4.legend()

    # PSI (if dev scores provided)
    ax5 = fig.add_subplot(gs[1, 1])
    if scores_dev is not None:
        psi_result = population_stability_index(scores_dev, scores)
        bins = range(len(psi_result["bin_contributions"]))
        ax5.bar(bins, psi_result["bin_contributions"], color="steelblue", alpha=0.7)
        ax5.axhline(0, color="black", lw=0.5)
        ax5.set_xlabel("Score Bin")
        ax5.set_ylabel("PSI Contribution")
        ax5.set_title(f"PSI by Bin\nTotal PSI = {psi_result['psi']:.4f}")
    else:
        ax5.text(0.5, 0.5, "PSI requires\ndevelopment scores",
                 ha="center", va="center", transform=ax5.transAxes, color="gray")
        ax5.set_title("Population Stability Index")
        psi_result = None

    # Metrics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    hl = hosmer_lemeshow_test(y_true, y_prob)
    cal = calibration_summary(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    metrics_text = (
        f"AUC:              {auc:.4f}\n"
        f"Brier Score:      {cal['brier_score']:.4f}\n"
        f"Log Loss:         {cal['log_loss']:.4f}\n"
        f"Pred. Bias:       {cal['prediction_bias']:+.4f}\n"
        f"\nHosmer-Lemeshow:\n"
        f"  Stat:           {hl['hl_statistic']:.4f}\n"
        f"  p-value:        {hl['p_value']:.4f}\n"
        f"  {hl['interpretation']}\n"
    )
    if psi_result:
        metrics_text += f"\nPSI: {psi_result['psi']:.4f}\n{psi_result['interpretation']}"

    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax6.set_title("Summary Statistics")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Evaluation report saved to {save_path}")

    plt.tight_layout()

    return {
        "auc": roc_auc_score(y_true, y_prob),
        "calibration": cal,
        "hosmer_lemeshow": hl,
        "psi": psi_result,
        "threshold_optimization": optimize_threshold(y_true, y_prob, cost_fp, cost_fn),
    }
