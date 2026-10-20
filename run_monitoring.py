"""CLI: monitor every simulated batch and write data/monitoring_history.csv."""
import json
import pandas as pd
from monitor.report import build_report, load_assets, summarize

assets = load_assets()
rows = []
for m in json.load(open("data/batches/manifest.json")):
    rep = build_report(pd.read_csv(f"data/batches/{m['file']}"), assets)
    rows.append(summarize(m["title"], rep))
hist = pd.DataFrame(rows)
hist.to_csv("data/monitoring_history.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(hist.round(3).to_string(index=False))
