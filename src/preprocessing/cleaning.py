import pandas as pd
import numpy as np
from .utils import parse_items

def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.lower()
        df = df[df["status"].isin(["success","succeeded","completed"])]

    for col in ["transaction_ts", "created_at", "updated_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        cap = df["amount"].quantile(0.99)
        df["amount"] = np.where(df["amount"] > cap, cap, df["amount"])

    df["items_parsed"] = df["items"].apply(parse_items) if "items" in df.columns else [[] for _ in range(len(df))]

    key_cols = [c for c in ["customer_id","merchant_id","transaction_ts"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols)
    return df

def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    df = users.copy()
    if "signup_date" in df.columns:
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("Unknown").astype(str).str.title()
    return df

def clean_merchants(merchants: pd.DataFrame) -> pd.DataFrame:
    df = merchants.copy()
    for col in ["category","subcategory","city","state","country"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df["rating"] = df["rating"].fillna(df["rating"].median())
    return df