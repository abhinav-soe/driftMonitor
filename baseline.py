"""Phase 1: train the churn model, save it with its reference data + baseline metrics.
Also holds out a 'production pool' of rows the model has never seen, used to simulate new batches."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from features import CATEGORICAL, FEATURES, NUMERIC, clean, engineer
from monitor.performance import compute_metrics

SEED = 42
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

raw = pd.read_csv("data/churn.csv").drop_duplicates()
dev_raw, pool = train_test_split(raw, test_size=0.35, stratify=raw["Churn"], random_state=SEED)
pool.to_csv("data/production_pool.csv", index=False)             # never used for training

missing = dev_raw[C.REQUIRED_COLUMNS].replace(r"^\s*$", np.nan, regex=True).isna().mean().to_dict()
dev = engineer(clean(dev_raw))
X, y = dev[FEATURES], dev["Churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.23, stratify=y, random_state=SEED)

model = Pipeline([
    ("prep", ColumnTransformer([("num", StandardScaler(), NUMERIC),
                                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
]).fit(X_tr, y_tr)

proba = model.predict_proba(X_te)[:, 1]
metrics = compute_metrics(y_te.values, proba)
metrics.pop("confusion"); metrics.pop("n")

dev[FEATURES + ["Churn"]].to_csv("data/reference.csv", index=False)       # reference dataset
pd.DataFrame({"y_true": y_te.values, "proba": proba}).to_csv("models/reference_scores.csv", index=False)
joblib.dump(model, "models/model.joblib")
json.dump(dict(metrics=metrics, churn_rate=float(y.mean()), positive_rate=float((proba >= C.THRESHOLD).mean()),
               raw_missing_rate=missing, n_reference=len(dev)), open("models/baseline.json", "w"), indent=2)
print("Reference rows:", len(dev), "| production pool:", len(pool))
print("Baseline metrics:", {k: round(v, 3) for k, v in metrics.items()})
