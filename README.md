# ML Model Monitoring & Drift Detection

An ML monitoring system for a customer churn prediction model. The project simulates production data batches, validates incoming data, detects feature and prediction drift, tracks model performance when ground-truth labels become available, and surfaces actionable alerts through a Streamlit dashboard.

It demonstrates an important distinction in production machine learning: **detecting a change in the data is not the same as proving that a model's performance has degraded.**

## Features

- **Data quality monitoring:** Check schema, missing-value rates, duplicate records, unseen categories, and out-of-range values.
- **Feature drift detection:** Apply Kolmogorov–Smirnov (KS) tests to numeric features and chi-square tests to categorical features.
- **Effect-size monitoring:** Use the Population Stability Index (PSI) to distinguish statistically significant changes from changes that are practically meaningful.
- **Prediction drift detection:** Monitor predicted probabilities using KS tests and PSI, as well as changes in the predicted churn rate.
- **Delayed-label performance monitoring:** Compare precision, recall, F1-score, and ROC-AUC against baseline metrics when actual labels become available.
- **Simulated production batches:** Generate five controlled scenarios covering normal operation, mild drift, strong drift, concept drift, and data-quality problems.
- **Interactive dashboard:** Review monitoring results and alerts in Streamlit.
- **Command-line reporting:** Summarize monitoring results across batches from the terminal.
- **Automated tests:** Run the project's unit tests using pytest.

## How It Works

The monitoring pipeline follows a baseline-and-observe workflow.

### Phase 1: Establish a baseline

`baseline.py` trains the churn prediction model using the Telco customer churn dataset. It saves the model, reference data, and baseline metrics used by subsequent monitoring runs.

The reference data describes the expected input distribution, while the baseline metrics provide a point of comparison for later performance evaluation.

### Phase 2: Simulate incoming production data

`simulate.py` creates five simulated production batches from a held-out pool that the model did not train on. Each batch represents a different operating condition.

### Phase 3: Validate incoming data

Before cleaning or imputation, the monitoring pipeline checks raw data for quality issues, including:

- Missing values and increases in missing-value rates.
- Duplicate records.
- Unseen categorical values.
- Values outside expected ranges.
- Schema inconsistencies.

Running quality checks before cleaning helps prevent preprocessing from concealing problems in incoming data.

### Phase 4: Detect feature and prediction drift

The monitoring system compares each batch with the reference distribution.

| Monitoring layer | Method | Requires ground-truth labels? |
|---|---|---|
| Data quality | Schema, missingness, duplicates, categories, ranges | No |
| Numeric feature drift | KS test and PSI | No |
| Categorical feature drift | Chi-square test and PSI | No |
| Prediction drift | KS test, PSI, predicted churn rate | No |
| Model performance | Precision, recall, F1-score, ROC-AUC | Yes |

A feature is flagged when the statistical test is significant at `p < 0.05` **and** PSI reaches the configured threshold:

- **PSI ≥ 0.10:** Moderate drift.
- **PSI ≥ 0.25:** High drift.

The thresholds are configured in `config.py`.

Using both statistical significance and PSI helps distinguish evidence of a distribution change from the practical magnitude of that change. P-values can flag small differences when sample sizes are large, while PSI provides an additional measure of distribution shift.

### Phase 5: Monitor performance when labels arrive

Input and prediction drift can be monitored without knowing the eventual outcomes. However, evaluating whether the model's predictions remain effective requires ground-truth labels.

When labels become available, the system compares precision, recall, F1-score, and ROC-AUC with the baseline metrics. The dashboard also warns when the number of labelled examples is too small for reliable interpretation.

### Phase 6: Review alerts

The Streamlit dashboard presents monitoring results and alerts so users can investigate changes in data quality, input distributions, predictions, and model performance.

## The Simulated Experiment

The project uses controlled scenarios to demonstrate why drift monitoring needs multiple layers.

| Batch | Scenario | Expected observation |
|---|---|---|
| 1 | Normal | No intentional changes; expected healthy operation |
| 2 | Mild drift | More high-charge customers; moderate feature drift while performance may hold |
| 3 | Strong drift | More annual contracts and high charges; stronger drift, although performance may remain stable |
| 4 | Concept drift | Input distributions remain the same while churn behaviour changes; performance may decline |
| 5 | Data-quality issues | 30% missing charges, duplicate records, and a new payment method trigger quality checks |

These are designed experimental scenarios, not results from a real production deployment.

**Key lesson:** Drift is a warning signal, not proof of model failure. A model can tolerate input drift, and model performance can deteriorate even when the input distribution remains stable.

## Tech Stack

- **Language:** Python
- **Machine learning:** scikit-learn
- **Data processing:** pandas and NumPy
- **Statistical monitoring:** SciPy
- **Dashboard:** Streamlit
- **Testing:** pytest

## Getting Started

### Prerequisites

- Python installed in a version compatible with the project dependencies.
- pip.
- A Telco customer churn CSV file from your churn prediction project or another compatible source.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

Replace the placeholders with your repository URL and directory name.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Prepare the dataset

Copy the Telco customer churn CSV from your churn prediction project into:

```text
data/churn.csv
```

Make sure the dataset's columns and types match the expectations of the baseline training pipeline.

### 4. Train the baseline model

```bash
python baseline.py
```

This is Phase 1: it trains the model and saves reference data and baseline metrics for monitoring.

### 5. Generate simulated production batches

```bash
python simulate.py
```

This is Phase 2: it generates five simulated batches representing different monitoring scenarios.

### 6. Run the monitoring summary

```bash
python run_monitoring.py
```

This optional command-line step summarizes monitoring results across the batches.

### 7. Launch the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to explore the monitoring dashboard.

### 8. Run the tests

```bash
python -m pytest
```

## Project Structure

The following is a high-level overview of the main scripts described in this project.

| File or directory | Purpose |
|---|---|
| `baseline.py` | Train the churn model and save reference data and baseline metrics |
| `simulate.py` | Generate five simulated production batches |
| `run_monitoring.py` | Produce a command-line summary of monitoring results |
| `app.py` | Streamlit dashboard for monitoring and alerts |
| `config.py` | Configure drift thresholds and monitoring settings |
| `data/churn.csv` | Input Telco churn dataset |
| `requirements.txt` | Python dependencies |
| `tests/` | Unit tests, if present |

Additional files may be used for individual quality checks, drift calculations, and metric comparisons. Refer to the repository's actual file tree for the complete structure.

## Design Decisions

- **Explicit statistical tests:** Drift detection uses SciPy so the underlying statistical calculations are inspectable and explainable.
- **Statistical significance plus effect size:** Combining p-values with PSI thresholds helps avoid treating every statistically significant change as operationally important.
- **Raw-data quality checks:** Quality checks run before cleaning so that imputation does not conceal missingness or other incoming-data problems.
- **Separate drift and performance monitoring:** The system distinguishes distribution changes from actual predictive performance changes.
- **Configurable thresholds:** Monitoring thresholds are centralized in `config.py`.

## Limitations

- **Controlled simulation:** The generated batches demonstrate expected monitoring behaviour but do not establish that the system has detected degradation in a real production environment.
- **Small batches:** Batches of approximately 600 rows can produce noisy metric estimates. Performance conclusions should account for the number of labelled examples.
- **No automated scheduling:** Monitoring runs are initiated manually rather than through a scheduler.
- **No alert deduplication:** Repeated alerts are not yet consolidated.
- **No persistent monitoring database:** Historical results are not stored in a database for long-term tracking.
- **No external notifications:** Email and Slack alerts are potential future extensions, not current capabilities.

## Future Improvements

- Schedule recurring monitoring runs using cron or Airflow.
- Persist monitoring results and alert history in SQLite or another database.
- Add alert deduplication and severity-based notification rules.
- Integrate email or Slack notifications.
- Evaluate additional drift methods or use a framework such as Evidently for supplementary reports.
- Add monitoring for data pipelines, model versions, and changes in prediction thresholds.
- Validate detection behaviour on real production data with delayed ground-truth labels.

## License and Attribution

Add the appropriate license for this repository and retain the required attribution for the dataset and any third-party dependencies you use.
