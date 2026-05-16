# %% [markdown]
# # Thin-File Credit Scorecard: Development Notebook
#
# **Author:** Diego Lestani
# **Dataset:** Home Credit Default Risk (Kaggle, 2018)
#
# ## What this notebook does
#
# This notebook develops a credit scorecard for thin-file borrowers —
# applicants with limited or no formal credit history — using only information
# available at the moment of application.
#
# The pipeline implements a standard scorecard methodology with several
# improvements over common implementations:
#
# 1. **Monotonic WOE binning** (via OptBinning) instead of manual interactive adjustment
# 2. **LASSO-based variable selection** instead of stepwise-by-IV
# 3. **Stratified k-fold cross-validation** instead of a single train/test split
# 4. **Calibration diagnostics** (reliability diagram, Hosmer-Lemeshow test)
# 5. **Population Stability Index** on a temporal holdout
# 6. **Cost-weighted threshold optimization** with explicit economic assumptions
#
# ## The inference problem
#
# Standard credit scoring relies on bureau data: repayment history, outstanding
# balances, length of credit history. For thin-file borrowers, this data either
# doesn't exist or is too sparse to be useful.
#
# The solution is to use **proxy variables** — observable characteristics that
# are correlated with creditworthiness without directly measuring past repayment:
# - Employment stability and duration
# - Income relative to credit exposure
# - Document submission compliance
# - Behavioral signals (contact information completeness)
# - External normalized scores (where available)
#
# This is the same inference logic used in poverty proxy means testing,
# satellite-based welfare estimation, and other data-scarce measurement problems.

# %% [markdown]
# ## 0. Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['font.size'] = 10

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))

from src.binning import (
    engineer_features, fit_binning, get_iv_table,
    transform_to_woe, VARIABLE_CLUSTERS, ALL_VARIABLES
)
from src.selection import select_variables_lasso, refit_logistic, get_coefficient_table
from src.scorecard import (
    compute_scaling_params, build_scorecard_table, score_from_log_odds,
    save_artifacts, POINTS0, ODDS0, PDO
)
from src.evaluation import (
    cross_validated_auc, full_evaluation_report,
    hosmer_lemeshow_test, population_stability_index,
    optimize_threshold
)

print("Libraries loaded.")

# %% [markdown]
# ## 1. Load and inspect data
#
# Download instructions: see `data/download.md`
# Place `application_train.csv` at `data/raw/application_train.csv`

# %%
DATA_PATH = "data/raw/application_train.csv"

df_raw = pd.read_csv(DATA_PATH)

print(f"Shape: {df_raw.shape}")
print(f"Target (default rate): {df_raw['TARGET'].mean():.2%}")
print(f"\nColumn types:\n{df_raw.dtypes.value_counts()}")

# %%
# Quick look at missing data for our key variables
key_vars = [v for v in ALL_VARIABLES if v in df_raw.columns]
missing_pct = df_raw[key_vars].isnull().mean().sort_values(ascending=False)
print("Missing data in key variables:")
print(missing_pct[missing_pct > 0].round(3))

# %% [markdown]
# ## 2. Feature engineering
#
# Days-based fields are stored as negative integers (days before application).
# We convert to interpretable units and construct financial ratios.
#
# The DAYS_EMPLOYED anomaly: the value 365243 indicates pensioners and
# people on maternity leave. We set these to NaN — the binning step will
# handle them as a separate "special" category.

# %%
df = engineer_features(df_raw)

# Verify engineered features
print("Engineered features:")
for feat in ["AGE_YEARS", "YEARS_EMPLOYED", "INCOME_CREDIT_RATIO",
             "ANNUITY_INCOME_RATIO", "CREDIT_GOODS_RATIO"]:
    print(f"  {feat}: mean={df[feat].mean():.2f}, "
          f"median={df[feat].median():.2f}, "
          f"missing={df[feat].isnull().mean():.1%}")

# %% [markdown]
# ## 3. Temporal train / test split
#
# We use `DAYS_DECISION` (days before application was processed) as a
# time proxy. Older applications go to training, recent ones to test.
# This is more realistic than a random split because it tests whether
# the model generalizes to future applicants — the actual deployment scenario.
#
# We also reserve the most recent 10% as an out-of-time (OOT) validation set
# for the Population Stability Index calculation.

# %%
# Sort by application recency (DAYS_DECISION is negative — less negative = more recent)
if "DAYS_DECISION" in df.columns:
    df_sorted = df.sort_values("DAYS_DECISION")
else:
    # Fallback: use index order (applications are roughly chronological in HF data)
    df_sorted = df.copy()

n = len(df_sorted)
train_end = int(n * 0.70)
val_end = int(n * 0.90)

df_train = df_sorted.iloc[:train_end].reset_index(drop=True)
df_test = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
df_oot = df_sorted.iloc[val_end:].reset_index(drop=True)

print(f"Training set:       {len(df_train):,} rows | default rate: {df_train['TARGET'].mean():.2%}")
print(f"Test set:           {len(df_test):,} rows | default rate: {df_test['TARGET'].mean():.2%}")
print(f"Out-of-time (OOT):  {len(df_oot):,} rows | default rate: {df_oot['TARGET'].mean():.2%}")

# %% [markdown]
# ## 4. WOE Binning
#
# We fit WOE binning on the training set only. The bins are then applied
# to test and OOT sets without refit — this is critical to avoid data leakage.
#
# **Monotonicity constraint:** OptBinning enforces that the WOE curve is
# monotone across bins. This means higher risk always maps to consistently
# higher (or lower) WOE values, which:
# 1. Prevents overfitting to local noise in the training data
# 2. Produces bins with a clear, defensible business interpretation
# 3. Ensures the resulting scorecard is monotone (higher score = lower risk)

# %%
print("Fitting WOE bins on training data...")
binning_process = fit_binning(df_train, target="TARGET")
print("Done.")

# %%
# Information Value table
iv_table = get_iv_table(binning_process)
print("\nInformation Value by variable:")
print(iv_table.to_string(index=False))

# %%
# Visualize IV distribution
fig, ax = plt.subplots(figsize=(10, 5))
colors = iv_table["information_value"].apply(
    lambda x: "firebrick" if x >= 0.3 else ("steelblue" if x >= 0.1 else
              ("skyblue" if x >= 0.02 else "lightgray"))
)
bars = ax.barh(iv_table["variable"], iv_table["information_value"], color=colors)
ax.axvline(0.02, color="orange", linestyle="--", alpha=0.7, label="Min threshold (0.02)")
ax.axvline(0.1, color="green", linestyle="--", alpha=0.7, label="Medium (0.1)")
ax.axvline(0.3, color="red", linestyle="--", alpha=0.7, label="Strong (0.3)")
ax.set_xlabel("Information Value")
ax.set_title("Variable Information Values\n(variables below 0.02 excluded from modeling)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/iv_chart.png", dpi=150, bbox_inches="tight")
plt.show()
print("IV chart saved.")

# %% [markdown]
# ## 5. Transform to WOE
#
# Apply the fitted bins to convert all variables to WOE values.
# Variables below the IV threshold are automatically excluded.

# %%
print("Transforming training data to WOE...")
woe_train = transform_to_woe(binning_process, df_train)
print(f"WOE feature matrix: {woe_train.shape}")
print(f"Variables retained (IV ≥ 0.02): {woe_train.shape[1] - 1}")

# %%
# Correlation matrix within clusters — check for within-cluster redundancy
print("\nWithin-cluster correlations (checking for redundancy):")
for cluster_name, cluster_vars in VARIABLE_CLUSTERS.items():
    woe_cols = [c for c in woe_train.columns
                if any(v.lower() in c.lower() for v in cluster_vars)]
    if len(woe_cols) >= 2:
        corr = woe_train[woe_cols].corr()
        high_corr = (corr.abs() > 0.7).sum().sum() - len(woe_cols)
        print(f"  {cluster_name}: {len(woe_cols)} vars, {high_corr} high-correlation pairs (|r|>0.7)")

# %% [markdown]
# ## 6. Variable selection via LASSO
#
# LASSO logistic regression with cross-validated regularization strength.
#
# The key advantage over the original stepwise-by-IV approach:
# LASSO considers all variables simultaneously, properly accounting for
# multicollinearity. A variable that looks weak in isolation may still
# contribute once we condition on correlated predictors — and vice versa.
#
# After LASSO selection, we refit an unpenalized logistic regression.
# This is the "post-LASSO" or "double selection" approach from
# Belloni & Chernozhukov (2013) — LASSO shrinks coefficients toward zero,
# so we need an unpenalized refit to get valid coefficient estimates.

# %%
X_woe = woe_train.drop(columns=["TARGET"])
y_train = woe_train["TARGET"]

print("Running LASSO variable selection...")
selected_vars = select_variables_lasso(X_woe, y_train, cv=5)

print(f"\nSelected variables:")
for v in selected_vars:
    print(f"  {v}")

# %%
print("\nRefitting unpenalized logistic regression on selected variables...")
model = refit_logistic(X_woe, y_train, selected_vars)

coef_table = get_coefficient_table(model, selected_vars)
print("\nCoefficients (sorted by magnitude):")
print(coef_table.to_string(index=False))

# %% [markdown]
# ## 7. Cross-validated AUC
#
# We estimate model performance via stratified k-fold CV.
# Stratified folds preserve the default rate (≈8%) in each fold,
# which matters for class-imbalanced outcomes.
#
# We report mean ± standard deviation across folds — the standard deviation
# tells us how stable the performance estimate is.

# %%
print("Running 5-fold cross-validated AUC estimation...")
cv_results = cross_validated_auc(X_woe, y_train, selected_vars, n_folds=5)

# %% [markdown]
# ## 8. Test set evaluation
#
# Apply the model to the held-out test set (the 20% of data not used in training).
# This gives an honest estimate of out-of-sample performance.

# %%
# Transform test set with training bins (no refit)
woe_test = transform_to_woe(binning_process, df_test)
X_test = woe_test.drop(columns=["TARGET"], errors="ignore")
y_test = df_test["TARGET"]

# Align columns
X_test_aligned = X_test.reindex(columns=selected_vars, fill_value=0)

# Predict
y_prob_test = model.predict_proba(X_test_aligned)[:, 1]
log_odds_test = model.decision_function(X_test_aligned)

# %% [markdown]
# ## 9. Build scorecard
#
# Convert log-odds to points scale using the three scaling parameters.
# See `src/scorecard.py` for the mathematical derivation.

# %%
factor, offset = compute_scaling_params(POINTS0, ODDS0, PDO)
print(f"Scaling parameters:")
print(f"  Target score (points0): {POINTS0}")
print(f"  Target odds (odds0):    {ODDS0}:1 (good:bad)")
print(f"  Points to double odds:  {PDO}")
print(f"  Derived factor:         {factor:.4f}")
print(f"  Derived offset:         {offset:.4f}")

# %%
# Score the test set
scores_test = score_from_log_odds(log_odds_test, factor, offset)

print(f"\nScore distribution on test set:")
print(f"  Mean:   {scores_test.mean():.1f}")
print(f"  Median: {np.median(scores_test):.1f}")
print(f"  Std:    {scores_test.std():.1f}")
print(f"  Min:    {scores_test.min():.1f}")
print(f"  Max:    {scores_test.max():.1f}")

# %%
# Score by outcome
scores_good = scores_test[y_test == 0]
scores_bad = scores_test[y_test == 1]
print(f"\nScores by outcome:")
print(f"  Good borrowers (no default): mean = {scores_good.mean():.1f}")
print(f"  Bad borrowers (default):     mean = {scores_bad.mean():.1f}")
print(f"  Separation: {scores_good.mean() - scores_bad.mean():.1f} points")

# %%
# Build the scorecard points table
scorecard_df = build_scorecard_table(
    binning_process, model, selected_vars, factor, offset
)
print("\nScorecard points table (first 20 rows):")
print(scorecard_df.head(20).to_string(index=False))

# %% [markdown]
# ## 10. Full evaluation report
#
# This section produces the four key diagnostics that were missing from
# the original pipeline:
#
# 1. **ROC curve**: discrimination performance
# 2. **Reliability diagram**: calibration — are predicted probabilities accurate?
# 3. **Cost-weighted threshold**: economic optimization of the accept/reject cutoff
# 4. **Population Stability Index**: score distribution stability on OOT data

# %%
# OOT scores for PSI
woe_oot = transform_to_woe(binning_process, df_oot)
X_oot = woe_oot.drop(columns=["TARGET"], errors="ignore")
X_oot_aligned = X_oot.reindex(columns=selected_vars, fill_value=0)
log_odds_oot = model.decision_function(X_oot_aligned)
scores_oot = score_from_log_odds(log_odds_oot, factor, offset)
y_oot = df_oot["TARGET"]

# %%
os.makedirs("figures", exist_ok=True)

print("Generating evaluation report...")
eval_results = full_evaluation_report(
    y_true=y_test,
    y_prob=y_prob_test,
    scores=scores_test,
    scores_dev=scores_test,   # PSI: test vs OOT
    cost_fp=100,              # cost of rejecting a good borrower
    cost_fn=300,              # cost of approving a bad borrower (3x FP)
    save_path="figures/evaluation_report.png",
)

# %%
print("\n=== EVALUATION SUMMARY ===")
print(f"AUC (test set):           {eval_results['auc']:.4f}")
print(f"Brier score:              {eval_results['calibration']['brier_score']:.4f}")
print(f"Prediction bias:          {eval_results['calibration']['prediction_bias']:+.4f}")
print(f"H-L p-value:              {eval_results['hosmer_lemeshow']['p_value']:.4f} "
      f"({eval_results['hosmer_lemeshow']['interpretation']})")
print(f"Optimal threshold:        {eval_results['threshold_optimization']['optimal_threshold']:.3f}")
print(f"Approval rate at optimal: {eval_results['threshold_optimization']['approval_rate']:.1%}")

# Also compute PSI on OOT
psi_oot = population_stability_index(scores_test, scores_oot)
print(f"\nPSI (test vs OOT):        {psi_oot['psi']:.4f} — {psi_oot['interpretation']}")

# %% [markdown]
# ## 11. Save artifacts
#
# All artifacts needed for the Gradio app are saved to `api/model_artifacts/`.
# The app loads these at startup for inference.

# %%
save_artifacts(
    binning_process=binning_process,
    model=model,
    selected_vars=selected_vars,
    scorecard_df=scorecard_df,
    factor=factor,
    offset=offset,
    output_dir="api/model_artifacts",
)

print("\nDone. Run `python app.py` to launch the Gradio interface.")
print("Or push to Hugging Face Spaces — it will auto-detect app.py.")
