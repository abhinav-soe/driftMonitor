# ML Model Monitoring & Drift Detection

Monitors a deployed churn model: validates incoming data, detects feature and prediction drift,
tracks performance when labels arrive, and raises actionable alerts in a Streamlit dashboard.

## Run
```bash
python -m pip install -r requirements.txt
# copy your Kaggle Telco CSV from the churn project to data/churn.csv
python baseline.py          # Phase 1: train model, save reference data + baseline metrics
python simulate.py          # Phase 2: build 5 simulated production batches
python run_monitoring.py    # (optional) CLI summary of every batch
python -m streamlit run app.py
python -m pytest            # Phase 5: unit tests
```

## What is monitored
| Layer | Method | Needs labels? |
|---|---|---|
| Data quality | schema, missing-rate increase, duplicates, unseen categories, out-of-range values | No |
| Feature drift | KS test (numeric), chi-square (categorical), PSI as effect size | No |
| Prediction drift | KS + PSI on predicted probabilities, predicted churn rate | No |
| Performance | precision, recall, F1, ROC-AUC vs baseline | **Yes** |

A feature is flagged only if the test is significant (p < 0.05) **and** PSI >= 0.10 (moderate) / 0.25 (high).
With thousands of rows, p-values flag trivial changes; PSI supplies the effect size. All thresholds live in `config.py`.

## The simulated experiment
Batches are drawn from a held-out pool the model never saw. **This is a controlled experiment, not evidence of real-world degradation.**

| Batch | What changes | Expected result |
|---|---|---|
| 1 Normal | nothing | healthy |
| 2 Mild drift | more high-charge customers | moderate drift, performance holds |
| 3 Strong drift | more annual contracts + high charges | strong drift; performance may hold (covariate shift) |
| 4 Concept drift | same inputs, different churn behaviour | **no input drift, performance drops** |
| 5 Data quality | 30% missing charges, duplicates, new payment method | quality alerts |

Key lessons: drift is a warning, not proof of failure (batch 3), and stable inputs do not guarantee good
predictions (batch 4) - which is why label-based performance monitoring is still needed.

## Design decisions & limitations
- Drift tests are written with SciPy so every statistic is explainable; Evidently could be added for extra HTML reports.
- Quality checks run on raw data *before* cleaning so imputation cannot hide missing values.
- Batches of ~600 rows give noisy metrics; the dashboard warns when labelled samples are small.
- No alert de-duplication, scheduling or database yet - natural next steps (cron/Airflow, SQLite history, email/Slack alerts).
