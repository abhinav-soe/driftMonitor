"""Phase 2: simulate production batches from the held-out pool (a CONTROLLED experiment)."""
import json
import os

import numpy as np
import pandas as pd

N, OUT = 600, "data/batches"
os.makedirs(OUT, exist_ok=True)
pool = pd.read_csv("data/production_pool.csv")
mc = pool.MonthlyCharges.fillna(pool.MonthlyCharges.median())
high_charge = (mc > mc.quantile(0.66)).astype(float)
annual = (pool.Contract != "Month-to-month").astype(float)


def draw(w=None, seed=0):
    return pool.sample(N, weights=w, random_state=seed).reset_index(drop=True)


b1 = draw(seed=1)
b2 = draw(w=1 + 1.5 * high_charge, seed=2)                                  # more high-charge customers
b3 = draw(w=(1 + 4 * high_charge) * (1 + 3 * annual), seed=3)               # more annual + high charges
b4 = draw(seed=4)                                                           # normal inputs, changed behaviour
r = np.random.default_rng(4)
mtm, churn = b4.Contract == "Month-to-month", b4.Churn == "Yes"
b4.loc[mtm & churn & (r.random(N) < 0.5), "Churn"] = "No"                   # retention campaign works
b4.loc[~mtm & ~churn & (r.random(N) < 0.12), "Churn"] = "Yes"               # competitor poaches annual users
b5 = draw(seed=5)
r = np.random.default_rng(5)
b5.loc[r.random(N) < 0.30, "MonthlyCharges"] = np.nan                       # broken billing feed
b5.loc[r.random(N) < 0.05, "PaymentMethod"] = "Crypto wallet"               # new, unseen category
b5 = pd.concat([b5, b5.sample(int(0.08 * N), random_state=5)])             # duplicated rows

batches = [
    ("batch_1_normal.csv", b1, "Batch 1 - Normal", "Same population as training. Expect: no alerts."),
    ("batch_2_mild_drift.csv", b2, "Batch 2 - Mild drift", "More high-charge customers. Expect: moderate drift in charge features."),
    ("batch_3_strong_drift.csv", b3, "Batch 3 - Strong drift", "Many more annual contracts + high charges. Expect: strong drift; performance may or may not drop."),
    ("batch_4_concept_drift.csv", b4, "Batch 4 - Concept drift", "Inputs look normal but churn behaviour changed. Expect: no input drift, performance drops."),
    ("batch_5_data_quality.csv", b5, "Batch 5 - Data quality issues", "30% missing charges, duplicates, unseen payment method. Expect: quality alerts."),
]
manifest = []
for fn, df, title, desc in batches:
    df.to_csv(f"{OUT}/{fn}", index=False)
    manifest.append(dict(file=fn, title=title, description=desc))
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
print("Wrote", len(batches), "batches to", OUT)
