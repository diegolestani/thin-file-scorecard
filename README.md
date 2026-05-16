# Thin-File Credit Scorecard

**Inference under data scarcity: scoring creditworthiness without credit history**

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/diegolestani/thin-file-scorecard)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## The problem

Standard credit scoring relies on bureau data: repayment history, outstanding balances, length of credit history. For a large share of the global population — informal workers, first-time borrowers, recent migrants, small business owners in emerging markets — this data simply doesn't exist.

This project builds a credit scorecard for exactly this population, using only information observable at the moment of application: demographic characteristics, employment profile, income and credit exposure ratios, document submission compliance, and external normalized scores where available.

This is fundamentally an **inference under data scarcity** problem. The methodology mirrors how poverty economists construct proxy means tests, how epidemiologists impute outcomes from administrative records, and how small area estimation infers local statistics from sparse surveys. The domain changes; the statistical challenge does not.

---

## Dataset

**Home Credit Default Risk** — Kaggle, 2018  
[kaggle.com/competitions/home-credit-default-risk](https://www.kaggle.com/competitions/home-credit-default-risk)

Home Credit serves unbanked populations in Southeast Europe and Asia. The dataset contains ~307,000 loan applications with a binary default outcome, rich in alternative variables (employment type, document flags, behavioral signals) and deliberately sparse in traditional bureau data.

**Key statistics:**
- 307,511 applications | 8% default rate (class imbalance is the norm)
- 122 variables in the application table; this project uses ~25 engineered features
- Only `application_train.csv` is used — information available at application time only

See [`data/download.md`](data/download.md) for download instructions.

---

## Methodology

The pipeline implements a standard scorecard architecture with several improvements over typical implementations.

### Architecture

```
Raw data
  ↓
Feature engineering (ratios, time conversions, proxy variables)
  ↓
WOE binning with monotonicity constraint (OptBinning)
  ↓
IV-based variable filtering (IV < 0.02 excluded)
  ↓
LASSO logistic regression for variable selection
  ↓
Post-LASSO refit (unpenalized logistic regression)
  ↓
Scorecard scaling (log-odds → points)
  ↓
Evaluation: discrimination + calibration + stability
  ↓
Deployment (Gradio / Hugging Face Spaces)
```

### Key design decisions

**WOE binning with monotonicity constraint**  
Weight-of-Evidence binning transforms both numerical and categorical variables into a common log-odds scale, making them directly comparable as inputs to logistic regression. The monotonicity constraint (enforced via OptBinning's constrained optimization) requires that the WOE curve is directionally consistent across bins — higher age is either consistently associated with lower risk or higher risk, not erratically alternating. This prevents overfitting to noise and produces bins with a clear business interpretation.

**LASSO for variable selection (not stepwise-by-IV)**  
A common approach builds models by incrementally adding variables in order of Information Value, then retaining those with significant coefficients. This is statistically problematic: IV is a univariate measure that ignores multicollinearity, and stepwise selection by p-value inflates Type I error because the same data is used for selection and inference.

LASSO (L1-penalized logistic regression) selects variables simultaneously, properly accounting for correlations. The regularization strength is chosen by 5-fold cross-validation optimizing AUC. After selection, an unpenalized logistic regression is refit on the selected subset — this recovers unbiased coefficient estimates and valid standard errors (Belloni & Chernozhukov, 2013).

**Calibration diagnostics**  
AUC measures ranking ability: whether the model correctly orders borrowers by risk. It says nothing about whether predicted probabilities are accurate in absolute terms. A model with AUC = 0.75 can still predict a 10% default rate for a group where the true rate is 25%. For a lending scorecard — where pricing depends on expected loss = PD × LGD × EAD — calibration matters as much as discrimination.

This project includes: reliability diagrams, Hosmer-Lemeshow goodness-of-fit test, and Brier score.

**Temporal train/test split + Population Stability Index**  
Data is split by application date (older → training, recent → test, most recent → out-of-time). This tests generalization to future applicants, the actual deployment scenario. PSI is computed between test and out-of-time sets to check for score distribution shift.

**Cost-weighted threshold optimization**  
The accept/reject threshold is chosen to minimize total misclassification cost, not maximize accuracy. The cost parameters reflect the economics of micro-lending: the cost of approving a bad borrower (expected loss) is approximately 3× the cost of rejecting a good borrower (foregone interest margin). These are explicit assumptions, not magic numbers.

---

## Scorecard scaling

The model's log-odds output is converted to a points scale using three parameters:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `points0` | 600 | Score assigned at the target odds |
| `odds0` | 50 | 50 good borrowers for every 1 bad at score 600 |
| `pdo` | 20 | 20-point increase doubles the odds of being a good borrower |

This follows the FICO convention. The target odds of 50:1 corresponds to a ~2% default rate at the reference score — appropriate for an emerging market micro-lending context.

The decomposition into per-variable, per-bin points allows adverse action explanation: if a borrower is declined, the scorecard can report which specific attributes reduced their score — a regulatory requirement in many jurisdictions.

---

## Results

| Metric | Value | Benchmark |
|--------|-------|-----------|
| Cross-validated AUC (5-fold) | *see notebook* | >0.70 acceptable |
| Brier score | *see notebook* | Lower is better |
| Hosmer-Lemeshow p-value | *see notebook* | >0.05 = well calibrated |
| PSI (test vs OOT) | *see notebook* | <0.10 = stable |

*Results populate after running the training notebook.*

---

## Repository structure

```
thin-file-scorecard/
├── app.py                          ← Gradio app (Hugging Face Spaces entry point)
├── requirements.txt
├── data/
│   └── download.md                 ← Instructions to get the data
├── notebooks/
│   └── scorecard_development.py    ← Training pipeline (run as Jupytext notebook)
├── src/
│   ├── binning.py                  ← WOE binning with monotonicity constraint
│   ├── selection.py                ← LASSO variable selection + post-LASSO refit
│   ├── scorecard.py                ← Scaling, points table, inference
│   └── evaluation.py               ← AUC, calibration, PSI, cost optimization
├── api/
│   └── model_artifacts/            ← Saved model files (generated by notebook)
└── figures/                        ← Evaluation plots (generated by notebook)
```

---

## Running locally

```bash
# 1. Clone
git clone https://github.com/diegolestani/thin-file-scorecard.git
cd thin-file-scorecard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download data (see data/download.md)
mkdir -p data/raw
# Place application_train.csv in data/raw/

# 4. Run the training notebook
# Option A: as a Python script
python notebooks/scorecard_development.py

# Option B: as a Jupyter notebook (requires jupytext)
jupytext --to notebook notebooks/scorecard_development.py
jupyter notebook notebooks/scorecard_development.ipynb

# 5. Launch the Gradio app
python app.py
```

---

## Deploying to Hugging Face Spaces

```bash
# Create a new Space at huggingface.co/new-space
# Select Gradio as the SDK

# Add the Space as a remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/thin-file-scorecard

# Push (model artifacts must be committed or fetched separately)
git push space main
```

The Space will auto-detect `app.py` and launch the Gradio interface.

**Note:** Model artifacts (`api/model_artifacts/`) are not committed to git by default (they can exceed 100MB). For Hugging Face deployment, either:
- Commit the artifacts directly (if <100MB each)
- Use [Hugging Face Hub](https://huggingface.co/docs/hub/models-uploading) to store artifacts and load them at runtime

---

## Limitations and production considerations

This is a demonstration project. Before production deployment:

1. **Recalibrate** `odds0` to the observed portfolio default rate
2. **Retrain** on a larger, more recent cohort
3. **Validate** the monotonicity assumptions against domain knowledge for each variable
4. **Set** cost parameters from actual loss data, not assumed ratios
5. **Implement** monthly PSI monitoring on new applicant cohorts
6. **Add** fairness diagnostics: test for disparate impact across protected groups

---

## References

- Belloni, A. & Chernozhukov, V. (2013). *Least squares after model selection in high-dimensional sparse models.* Bernoulli.
- Siddiqi, N. (2012). *Credit Risk Scorecards.* Wiley.
- Naeem, M. et al. (2019). *OptBinning: Optimal Binning for Scorecard Development.* arXiv.

---

## Author

**Diego Lestani**  
Quantitative economist · World Bank (Distributional Impact of Policies Unit)  
[linkedin.com/in/diegolestani](https://linkedin.com/in/diegolestani)
