import numpy as np
import pandas as pd

import config as C
from monitor.drift import detect_drift, psi_categorical, psi_numeric
from monitor.performance import compare, compute_metrics
from monitor.quality import check_raw

rng = np.random.default_rng(0)


def make(n, mc_mean=60, p_mtm=0.5):
    return pd.DataFrame({"MonthlyCharges": rng.normal(mc_mean, 15, n),
                         "Contract": rng.choice(["Month-to-month", "One year"], n, p=[p_mtm, 1 - p_mtm])})


def status(ref, cur):
    df, _ = detect_drift(ref, cur, ["MonthlyCharges", "Contract"])
    return dict(zip(df.feature, df.status))


def test_psi_zero_for_identical_data():
    x = rng.normal(size=2000)
    assert psi_numeric(x, x) < 1e-9
    s = pd.Series(["a", "b"] * 100)
    assert psi_categorical(s, s) < 1e-9


def test_no_drift_when_same_distribution():
    assert set(status(make(3000), make(600)).values()) == {"none"}


def test_numeric_shift_detected():
    assert status(make(3000), make(600, mc_mean=80))["MonthlyCharges"] == "high"


def test_categorical_shift_detected():
    assert status(make(3000), make(600, p_mtm=0.2))["Contract"] == "high"


def _raw(n=100):
    return pd.DataFrame({c: ["x"] * n for c in C.REQUIRED_COLUMNS})


def test_quality_flags_missing_column_and_nulls():
    assert check_raw(_raw().drop(columns=["tenure"]), {})[0]["severity"] == "critical"
    raw = _raw()
    raw.loc[:29, "MonthlyCharges"] = np.nan
    assert any(i["check"] == "missing values" and i["severity"] == "critical" for i in check_raw(raw, {}))


def test_quality_flags_duplicates():
    raw = pd.concat([_raw(), _raw().head(10)])
    assert any(i["check"] == "duplicates" for i in check_raw(raw, {}))


def test_performance_drop_levels():
    base = dict(precision=.8, recall=.8, f1=.8, roc_auc=.9)
    cur = dict(precision=.78, recall=.6, f1=.74, roc_auc=.9)
    s = {r["metric"]: r["status"] for r in compare(cur, base)}
    assert s == {"precision": "ok", "recall": "critical", "f1": "warning", "roc_auc": "ok"}


def test_perfect_predictions_metrics():
    m = compute_metrics([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert m["recall"] == 1 and m["roc_auc"] == 1
