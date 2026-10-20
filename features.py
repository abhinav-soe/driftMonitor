"""Shared cleaning + feature engineering (used by training AND the dashboard)."""
import numpy as np
import pandas as pd

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                "StreamingTV", "StreamingMovies"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
           "num_services", "charge_per_service", "SeniorCitizen"]
CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "InternetService",
               "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group",
               "is_autopay", "new_month_to_month", "MultipleLines"] + SERVICE_COLS
FEATURES = NUMERIC + CATEGORICAL


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # new customers (tenure 0) have no total yet -> 0; other gaps -> median
    df.loc[df.tenure == 0, "TotalCharges"] = 0
    for c in ["MonthlyCharges", "TotalCharges"]:
        df[c] = df[c].fillna(df[c].median())
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return df.reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_monthly_spend"] = df.TotalCharges / df.tenure.clip(lower=1)
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)
    df["charge_per_service"] = df.MonthlyCharges / (df.num_services + 1)
    df["tenure_group"] = pd.cut(df.tenure, [-1, 12, 24, 48, 100],
                                labels=["0-1y", "1-2y", "2-4y", "4y+"]).astype(str)
    df["is_autopay"] = df.PaymentMethod.str.contains("automatic").map({True: "Yes", False: "No"})
    df["new_month_to_month"] = ((df.Contract == "Month-to-month") & (df.tenure <= 12)).map(
        {True: "Yes", False: "No"})
    return df
