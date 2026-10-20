"""Data-quality checks: schema, missing values, duplicates, unseen categories, out-of-range values."""
import numpy as np
import pandas as pd

import config as C
from features import CATEGORICAL

CONTINUOUS = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "charge_per_service"]


def _issue(check, column, value, severity, message):
    return dict(check=check, column=column, value=value, severity=severity, message=message)


def check_raw(raw: pd.DataFrame, baseline: dict):
    issues = []
    missing = [c for c in C.REQUIRED_COLUMNS if c not in raw.columns]
    for c in missing:
        issues.append(_issue("schema", c, None, "critical", f"Required column '{c}' is missing"))
    if missing:
        return issues
    extra = [c for c in raw.columns if c not in C.REQUIRED_COLUMNS + ["customerID", "Churn"]]
    if extra:
        issues.append(_issue("schema", ", ".join(extra), None, "info", "Unexpected extra columns (ignored)"))

    blanks = raw[C.REQUIRED_COLUMNS].replace(r"^\s*$", np.nan, regex=True)
    ref_missing = baseline.get("raw_missing_rate", {})
    for c in C.REQUIRED_COLUMNS:
        rate = float(blanks[c].isna().mean())
        inc = rate - ref_missing.get(c, 0.0)
        if inc >= C.MISSING_CRIT:
            sev = "critical"
        elif inc >= C.MISSING_WARN:
            sev = "warning"
        else:
            continue
        issues.append(_issue("missing values", c, rate, sev,
                             f"{rate:.1%} of '{c}' is missing (reference: {ref_missing.get(c, 0):.1%})"))
    dup = float(raw.duplicated().mean())
    if dup >= C.DUP_WARN:
        issues.append(_issue("duplicates", "-", dup, "warning", f"{dup:.1%} of rows are exact duplicates"))
    return issues


def check_features(X: pd.DataFrame, ref: pd.DataFrame):
    issues = []
    for c in CATEGORICAL:
        unseen = set(X[c].dropna().astype(str)) - set(ref[c].astype(str))
        if unseen:
            share = float(X[c].astype(str).isin(unseen).mean())
            issues.append(_issue("unseen category", c, share, "warning",
                                 f"{share:.1%} of '{c}' has categories never seen in training: {sorted(unseen)}"))
    for c in CONTINUOUS:
        lo, hi = ref[c].min(), ref[c].max()
        share = float(((X[c] < lo) | (X[c] > hi)).mean())
        if share >= C.RANGE_WARN:
            issues.append(_issue("out of range", c, share, "warning",
                                 f"{share:.1%} of '{c}' lies outside the reference range [{lo:.1f}, {hi:.1f}]"))
    return issues
