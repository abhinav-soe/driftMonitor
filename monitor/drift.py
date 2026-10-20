"""Feature drift (KS / chi-square + PSI) and prediction drift."""
import numpy as np
import pandas as pd
from scipy import stats

import config as C
from features import CATEGORICAL

EPS = 1e-4


def psi_numeric(ref, cur, bins=10):
    """Population Stability Index using quantile bins of the reference data."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur):
    cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
    r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    c = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    return float(((c - r) * np.log(c / r)).sum())


def classify(p_value, psi):
    """Drift needs BOTH statistical significance and a meaningful effect size.
    With large samples p-values become tiny for trivial changes, so PSI acts as the effect size."""
    if p_value < C.ALPHA and psi >= C.PSI_HIGH:
        return "high"
    if p_value < C.ALPHA and psi >= C.PSI_MODERATE:
        return "moderate"
    return "none"


def feature_kind(name, ref_col):
    if name in CATEGORICAL or ref_col.nunique() <= 2:
        return "categorical"
    return "numerical"


def _numeric(name, r, c):
    r = pd.to_numeric(r, errors="coerce").dropna()
    c = pd.to_numeric(c, errors="coerce").dropna()
    ks = stats.ks_2samp(r, c)
    psi = psi_numeric(r, c)
    rel = (c.mean() - r.mean()) / abs(r.mean()) if r.mean() else 0.0
    return dict(feature=name, type="numerical", test="KS", statistic=float(ks.statistic),
                p_value=float(ks.pvalue), psi=psi,
                change=f"mean {r.mean():.2f} -> {c.mean():.2f} ({rel:+.0%})")


def _categorical(name, r, c):
    r, c = r.astype(str), c.astype(str)
    cats = sorted(set(r) | set(c))
    rc = r.value_counts().reindex(cats, fill_value=0)
    cc = c.value_counts().reindex(cats, fill_value=0)
    table = np.array([rc.values, cc.values])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        chi, p = 0.0, 1.0
    else:
        chi, p = stats.chi2_contingency(table)[:2]
    rs, cs = rc / rc.sum(), cc / cc.sum()
    k = (cs - rs).abs().idxmax()
    return dict(feature=name, type="categorical", test="Chi-square", statistic=float(chi),
                p_value=float(p), psi=psi_categorical(r, c),
                change=f"{k}: {rs[k]:.0%} -> {cs[k]:.0%}")


def detect_drift(ref: pd.DataFrame, cur: pd.DataFrame, features):
    rows = []
    for f in features:
        kind = feature_kind(f, ref[f])
        row = (_numeric if kind == "numerical" else _categorical)(f, ref[f], cur[f])
        row["status"] = classify(row["p_value"], row["psi"])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    n_dr = int((df.status != "none").sum())
    return df, dict(n_features=len(df), n_drifted=n_dr, share=n_dr / max(len(df), 1))


def prediction_drift(ref_proba, cur_proba):
    ks = stats.ks_2samp(ref_proba, cur_proba)
    psi = psi_numeric(ref_proba, cur_proba)
    return dict(ks_statistic=float(ks.statistic), p_value=float(ks.pvalue), psi=psi,
                ref_rate=float((np.asarray(ref_proba) >= C.THRESHOLD).mean()),
                cur_rate=float((np.asarray(cur_proba) >= C.THRESHOLD).mean()),
                status=classify(ks.pvalue, psi))
