import pandas as pd
import numpy as np

TARGET = "SLA_Breached"
FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete",
    "SLA_Days", "Submission_Weekday", "Queue_Category"
]
REQUIRED_COLUMNS = ["Request_ID", *FEATURES, TARGET]

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    if "Submission_Date" in df.columns:
        dates = pd.to_datetime(df["Submission_Date"], errors="coerce")
        if "Submission_Weekday" not in df.columns:
            df["Submission_Weekday"] = dates.dt.day_name()
    if "Queue_Length" in df.columns and "Queue_Category" not in df.columns:
        q = pd.to_numeric(df["Queue_Length"], errors="coerce")
        df["Queue_Category"] = pd.cut(
            q, bins=[-np.inf, 5, 15, np.inf],
            labels=["Low", "Medium", "High"]
        ).astype("object")
    for col in ["Queue_Length", "SLA_Days", TARGET]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def validate_dataset(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if df.empty:
        return False, ["The uploaded dataset contains no rows."], missing
    if missing:
        return False, [f"Missing required columns: {', '.join(missing)}"], missing
    return True, [], []

def data_quality_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_request_ids": int(df["Request_ID"].duplicated().sum()) if "Request_ID" in df else 0,
        "missing_by_column": df.isna().sum().to_dict()
    }
