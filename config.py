"""All monitoring thresholds in one place - tune these for your own use case."""
from features import SERVICE_COLS

THRESHOLD = 0.5            # decision threshold of the churn model
ALPHA = 0.05               # significance level for KS / chi-square tests
PSI_MODERATE = 0.10        # PSI >= 0.10 -> moderate shift   (common industry rule of thumb)
PSI_HIGH = 0.25            # PSI >= 0.25 -> major shift
SHARE_WARN = 0.20          # share of drifted features that triggers a warning
SHARE_CRIT = 0.50          # ... and a critical alert
PERF_WARN_DROP = 0.05      # absolute drop vs baseline metric -> warning
PERF_CRIT_DROP = 0.10      # absolute drop vs baseline metric -> critical
MISSING_WARN = 0.02        # increase in missing-value rate vs reference
MISSING_CRIT = 0.10
DUP_WARN = 0.01            # share of duplicate rows
RANGE_WARN = 0.02          # share of values outside the reference min/max
MIN_LABELS = 200           # below this, metrics are flagged as noisy

REQUIRED_COLUMNS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                    "MultipleLines", "InternetService", "Contract", "PaperlessBilling",
                    "PaymentMethod", "MonthlyCharges", "TotalCharges"] + SERVICE_COLS
