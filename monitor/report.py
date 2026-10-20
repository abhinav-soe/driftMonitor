"""Orchestrates one monitoring run: quality -> drift -> predictions -> performance -> alerts."""
import json
import os

import joblib
import numpy as np
import pandas as pd

import config as C
from features import FEATURES, clean, engineer
from monitor import drift, performance, quality


def load_assets():
    base = json.load(open("models/baseline.json"))
    return dict(model=joblib.load("models/model.joblib"), baseline=base,
                ref=pd.read_csv("data/reference.csv"), ref_scores=pd.read_csv("models/reference_scores.csv"))


def _alert(level, title, detail, action=""):
    return dict(level=level, title=title, detail=detail, action=action)


def _labels(raw):
    s = raw["Churn"].astype(str).str.strip().str.lower()
    return s.map({"yes": 1, "1": 1, "true": 1, "no": 0, "0": 0, "false": 0, "1.0": 1, "0.0": 0})


def build_report(raw: pd.DataFrame, assets: dict, use_labels: bool = True):
    base = assets["baseline"]
    q_raw = quality.check_raw(raw, base)
    if any(i["check"] == "schema" and i["severity"] == "critical" for i in q_raw):
        return dict(status="invalid", n_rows=len(raw), quality=q_raw, drift=None, drift_summary=None,
                    prediction=None, performance=None, labels_status="none", X=None, proba=None,
                    alerts=[_alert("critical", "Schema check failed", q_raw[0]["message"],
                                   "Fix the upstream data feed - the model cannot score this batch.")])

    raw1 = raw.drop_duplicates().reset_index(drop=True)
    y = _labels(raw1) if (use_labels and "Churn" in raw1.columns) else None
    work = raw1.drop(columns=["Churn"], errors="ignore").copy()
    work["_row"] = np.arange(len(work))           # keeps feature rows aligned with labels
    X = engineer(clean(work))[FEATURES]
    proba = assets["model"].predict_proba(X)[:, 1]

    ref = assets["ref"]
    q_feat = quality.check_features(X, ref)
    drift_df, dsum = drift.detect_drift(ref[FEATURES], X, FEATURES)
    pred = drift.prediction_drift(assets["ref_scores"]["proba"].values, proba)

    # ---- performance (only if labels are usable) ----
    lstat = performance.label_status(y)
    perf = None
    if lstat == "ok":
        mask = y.notna().values
        cur = performance.compute_metrics(y[mask].values, proba[mask])
        perf = dict(current=cur, rows=performance.compare(cur, base["metrics"]), n=cur["n"])

    # ---- alerts ----
    alerts = []
    for i in q_raw + q_feat:
        lvl = i["severity"]
        alerts.append(_alert(lvl, f"Data quality: {i['check']} ({i['column']})", i["message"],
                             "Check the upstream pipeline / data source before trusting this batch."))
    share = dsum["share"]
    high = drift_df[drift_df.status == "high"].feature.tolist()
    mod = drift_df[drift_df.status == "moderate"].feature.tolist()
    if share >= C.SHARE_CRIT or share >= C.SHARE_WARN:
        alerts.append(_alert("critical" if share >= C.SHARE_CRIT else "warning", "Dataset-level drift",
                             f"{dsum['n_drifted']} of {dsum['n_features']} features drifted ({share:.0%}).",
                             "Investigate upstream changes (new segment, pricing change, pipeline bug). "
                             "If confirmed, plan a retrain on recent data."))
    if high:
        alerts.append(_alert("warning", "Strong feature drift", "High drift in: " + ", ".join(high[:8]),
                             "Compare distributions on the 'Feature drift' tab."))
    if mod:
        alerts.append(_alert("info", "Moderate feature drift", "Moderate drift in: " + ", ".join(mod[:8])))
    if pred["status"] != "none":
        alerts.append(_alert("warning" if pred["status"] == "high" else "info", "Prediction drift",
                             f"Predicted churn rate {pred['ref_rate']:.1%} -> {pred['cur_rate']:.1%} "
                             f"(PSI {pred['psi']:.2f}).",
                             "May reflect a real population change, a data problem or a model issue."))
    perf_bad = False
    if perf:
        for r in perf["rows"]:
            if r["status"] != "ok":
                perf_bad = True
                alerts.append(_alert(r["status"], f"Performance drop: {r['metric']}",
                                     f"{r['metric']} {r['baseline']:.3f} -> {r['current']:.3f} ({r['change']:+.3f}).",
                                     "Check label quality and recent business changes; "
                                     "retrain or recalibrate on recent labelled data."))
        if perf["n"] < C.MIN_LABELS:
            alerts.append(_alert("info", "Small labelled sample",
                                 f"Only {perf['n']} labelled rows - metrics are noisy; treat drops with care."))
        if perf_bad and share < C.SHARE_WARN and not high:
            alerts.append(_alert("critical", "Performance dropped WITHOUT input drift",
                                 "Inputs look stable but predictions got worse - a sign of concept drift "
                                 "(the relationship between features and churn changed).",
                                 "Input-drift monitoring alone would have missed this; retrain on recent labels."))
        if (high or share >= C.SHARE_WARN) and not perf_bad:
            alerts.append(_alert("info", "Inputs drifted, performance held up",
                                 "Drift is a warning, not proof of failure: this looks like covariate shift "
                                 "the model currently handles. Keep monitoring."))
    elif lstat == "none":
        alerts.append(_alert("info", "No labels available",
                             "Performance cannot be measured yet - only drift signals are shown. "
                             "Re-run once actual churn outcomes are recorded."))
    elif lstat == "single_class":
        alerts.append(_alert("info", "Labels contain a single class", "Metrics such as ROC-AUC are undefined."))

    levels = {a["level"] for a in alerts}
    status = "critical" if "critical" in levels else "warning" if "warning" in levels else "healthy"
    return dict(status=status, n_rows=len(raw), n_scored=len(X), quality=q_raw + q_feat, drift=drift_df,
                drift_summary=dsum, prediction=pred, performance=perf, labels_status=lstat,
                alerts=alerts, X=X, proba=proba)


def summarize(name, rep):
    row = dict(batch=name, status=rep["status"], rows=rep["n_rows"])
    if rep["drift_summary"]:
        row.update(drift_share=rep["drift_summary"]["share"], n_drifted=rep["drift_summary"]["n_drifted"],
                   predicted_churn_rate=rep["prediction"]["cur_rate"])
    if rep["performance"]:
        row.update({r["metric"]: r["current"] for r in rep["performance"]["rows"]})
    row["critical_alerts"] = sum(a["level"] == "critical" for a in rep["alerts"])
    row["warning_alerts"] = sum(a["level"] == "warning" for a in rep["alerts"])
    return row


def export_json(rep):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)
    out = {k: v for k, v in rep.items() if k not in ("X", "proba", "drift")}
    out["drift"] = None if rep["drift"] is None else rep["drift"].to_dict("records")
    return json.dumps(out, indent=2, default=conv)
