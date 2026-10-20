import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
from monitor.report import build_report, export_json, load_assets, summarize

st.set_page_config(page_title="Model Monitoring", page_icon="🛰️", layout="wide")
BADGE = {"healthy": "🟢 Healthy", "warning": "🟠 Warning", "critical": "🔴 Critical", "invalid": "⛔ Invalid input"}
COLORS = {"none": "#4c9a6a", "moderate": "#e0a030", "high": "#d9534f"}
SHOW = {"critical": st.error, "warning": st.warning, "info": st.info}


@st.cache_resource
def assets():
    return load_assets()


@st.cache_data
def batch_report(file, use_labels):
    return build_report(pd.read_csv(f"data/batches/{file}"), assets(), use_labels)


@st.cache_data
def history(use_labels):
    return pd.DataFrame([summarize(m["title"], batch_report(m["file"], use_labels)) for m in manifest])


try:
    A = assets()
    manifest = json.load(open("data/batches/manifest.json"))
except FileNotFoundError:
    st.error("Run `python baseline.py` and `python simulate.py` first.")
    st.stop()

# ------------------------------ sidebar ------------------------------
st.sidebar.title("🛰️ Model Monitor")
mode = st.sidebar.radio("Data source", ["Simulated batches", "Upload CSV"])
use_labels = not st.sidebar.checkbox("Pretend labels are not available yet", value=False)
if mode == "Simulated batches":
    titles = [m["title"] for m in manifest]
    pick = st.sidebar.selectbox("Production batch", titles)
    meta = manifest[titles.index(pick)]
    st.sidebar.caption(meta["description"])
    rep = batch_report(meta["file"], use_labels)
    name = pick
else:
    up = st.sidebar.file_uploader("Production data (raw Telco-format CSV, 'Churn' column optional)", type="csv")
    if up is None:
        st.info("Upload a CSV with the same columns as the training data to monitor it.")
        st.stop()
    rep = build_report(pd.read_csv(up), A, use_labels)
    name = up.name

st.sidebar.markdown("---")
st.sidebar.caption(f"Thresholds: α={C.ALPHA}, PSI moderate ≥{C.PSI_MODERATE}, high ≥{C.PSI_HIGH}. Edit `config.py`.")

# ------------------------------ header ------------------------------
st.title("Model Monitoring & Drift Detection")
st.subheader(f"{name} — {BADGE[rep['status']]}")
st.caption("Drift is a warning, not proof the model is wrong. Stable inputs do not guarantee good predictions.")

if rep["status"] == "invalid":
    for a in rep["alerts"]:
        st.error(f"**{a['title']}** — {a['detail']}")
    st.stop()

tabs = st.tabs(["🚨 Summary & alerts", "🧹 Data quality", "📊 Feature drift", "🎯 Prediction drift",
                "📈 Performance", "📤 Export"])
base = A["baseline"]

# ------------------------------ summary ------------------------------
with tabs[0]:
    k = st.columns(4)
    k[0].metric("Rows scored", f"{rep['n_scored']:,}")
    k[1].metric("Drifted features", f"{rep['drift_summary']['n_drifted']} / {rep['drift_summary']['n_features']}")
    k[2].metric("Predicted churn rate", f"{rep['prediction']['cur_rate']:.1%}",
                f"{rep['prediction']['cur_rate'] - rep['prediction']['ref_rate']:+.1%} vs ref", delta_color="off")
    if rep["performance"]:
        r = next(x for x in rep["performance"]["rows"] if x["metric"] == "roc_auc")
        k[3].metric("ROC-AUC", f"{r['current']:.3f}", f"{r['change']:+.3f} vs baseline")
    else:
        k[3].metric("ROC-AUC", "n/a", "no labels", delta_color="off")

    st.markdown("#### Alerts")
    if not rep["alerts"]:
        st.success("No alerts — this batch looks like the reference data.")
    order = {"critical": 0, "warning": 1, "info": 2}
    for a in sorted(rep["alerts"], key=lambda a: order[a["level"]]):
        SHOW[a["level"]](f"**{a['title']}** — {a['detail']}" + (f"\n\n*Suggested action:* {a['action']}" if a["action"] else ""))

    if mode == "Simulated batches":
        st.markdown("#### History across batches")
        h = history(use_labels)
        cols = [c for c in ["drift_share", "predicted_churn_rate", "recall", "roc_auc"] if c in h.columns]
        fig = px.line(h.melt("batch", cols, "metric", "value"), x="batch", y="value", color="metric", markers=True)
        fig.update_layout(xaxis_title=None, yaxis_title=None, legend_title=None, height=380)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(h.round(3), hide_index=True, width="stretch")

# ------------------------------ quality ------------------------------
with tabs[1]:
    if rep["quality"]:
        q = pd.DataFrame(rep["quality"])[["severity", "check", "column", "message"]]
        st.dataframe(q, hide_index=True, width="stretch")
    else:
        st.success("All data-quality checks passed (schema, missing values, duplicates, unseen categories, ranges).")
    st.caption("Raw-data checks run before cleaning, so imputation cannot hide missing values.")

# ------------------------------ feature drift ------------------------------
with tabs[2]:
    d = rep["drift"]
    fig = px.bar(d, x="psi", y="feature", color="status", color_discrete_map=COLORS, orientation="h",
                 category_orders={"feature": d.feature.tolist()}, height=520)
    fig.add_vline(x=C.PSI_MODERATE, line_dash="dot"); fig.add_vline(x=C.PSI_HIGH, line_dash="dash")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    show = d[["feature", "type", "test", "statistic", "p_value", "psi", "status", "change"]]
    st.dataframe(show.style.format({"statistic": "{:.3f}", "p_value": "{:.2e}", "psi": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.markdown("#### Compare distributions")
    f = st.selectbox("Feature", d.feature.tolist())
    ref, cur = A["ref"][f], rep["X"][f]
    if d.loc[d.feature == f, "type"].iloc[0] == "numerical":
        df = pd.concat([pd.DataFrame({"value": ref, "dataset": "Reference"}),
                        pd.DataFrame({"value": cur, "dataset": "Current batch"})])
        fig = px.histogram(df, x="value", color="dataset", barmode="overlay", histnorm="probability density",
                           opacity=0.6, nbins=40, color_discrete_sequence=["#4c72b0", "#d9534f"])
    else:
        df = pd.concat([ref.astype(str).value_counts(normalize=True).rename("share").reset_index().assign(dataset="Reference"),
                        cur.astype(str).value_counts(normalize=True).rename("share").reset_index().assign(dataset="Current batch")])
        df.columns = ["category", "share", "dataset"]
        fig = px.bar(df, x="category", y="share", color="dataset", barmode="group",
                     color_discrete_sequence=["#4c72b0", "#d9534f"])
    st.plotly_chart(fig, width="stretch")
    st.caption("Status needs BOTH a significant test (KS / chi-square) and a meaningful PSI — with big samples "
               "p-values alone flag trivial changes.")

# ------------------------------ prediction drift ------------------------------
with tabs[3]:
    p = rep["prediction"]
    c = st.columns(3)
    c[0].metric("Reference predicted churn rate", f"{p['ref_rate']:.1%}")
    c[1].metric("Current predicted churn rate", f"{p['cur_rate']:.1%}", f"{p['cur_rate'] - p['ref_rate']:+.1%}", delta_color="off")
    c[2].metric("Prediction PSI", f"{p['psi']:.3f}", p["status"], delta_color="off")
    df = pd.concat([pd.DataFrame({"p": A["ref_scores"]["proba"], "dataset": "Reference"}),
                    pd.DataFrame({"p": rep["proba"], "dataset": "Current batch"})])
    fig = px.histogram(df, x="p", color="dataset", barmode="overlay", histnorm="probability density", opacity=0.6,
                       nbins=30, color_discrete_sequence=["#4c72b0", "#d9534f"])
    fig.add_vline(x=C.THRESHOLD, line_dash="dash")
    st.plotly_chart(fig, width="stretch")
    st.caption("Needs no labels. A shift can mean a real population change, a data problem, or a model issue.")

# ------------------------------ performance ------------------------------
with tabs[4]:
    perf = rep["performance"]
    if not perf:
        st.info({"none": "No labels available for this batch, so performance cannot be measured. "
                         "Drift signals are the only early warning until outcomes are recorded.",
                 "single_class": "Labels contain only one class; ROC-AUC is undefined."}[rep["labels_status"]])
    else:
        t = pd.DataFrame(perf["rows"])
        st.dataframe(t.style.format({"baseline": "{:.3f}", "current": "{:.3f}", "change": "{:+.3f}"}),
                     hide_index=True, width="stretch")
        a, b = st.columns(2)
        fig = go.Figure([go.Bar(name="Baseline", x=t.metric, y=t.baseline, marker_color="#4c72b0"),
                         go.Bar(name="Current batch", x=t.metric, y=t.current, marker_color="#d9534f")])
        fig.update_layout(barmode="group", height=350, yaxis_range=[0, 1])
        a.plotly_chart(fig, width="stretch")
        cm = perf["current"]["confusion"]
        fig = px.imshow(cm, text_auto=True, x=["Stay", "Churn"], y=["Stay", "Churn"], color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual"), height=350)
        b.plotly_chart(fig, width="stretch")
        st.caption(f"Based on {perf['n']} labelled rows at threshold {C.THRESHOLD}. Small batches give noisy metrics.")

# ------------------------------ export ------------------------------
with tabs[5]:
    st.download_button("⬇️ Full report (JSON)", export_json(rep), "monitoring_report.json", "application/json")
    st.download_button("⬇️ Feature drift table (CSV)", rep["drift"].to_csv(index=False), "feature_drift.csv", "text/csv")
