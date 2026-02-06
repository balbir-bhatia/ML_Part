import pandas as pd
import numpy as np

def build_user_features(transactions_clean: pd.DataFrame) -> pd.DataFrame:
    df = transactions_clean.copy()
    df["transaction_ts"] = pd.to_datetime(df["transaction_ts"], errors="coerce")
    now = df["transaction_ts"].max() if "transaction_ts" in df.columns else pd.Timestamp.utcnow()

    agg = df.groupby("customer_id").agg(
        frequency=("transaction_ts","count"),
        monetary=("amount","sum"),
        avg_amount=("amount","mean"),
        last_txn=("transaction_ts","max")
    ).reset_index()

    agg["recency_days"] = (now - agg["last_txn"]).dt.days.clip(lower=0)
    agg["price_sensitivity"] = (agg["avg_amount"] / (agg["avg_amount"].median() or 1)).fillna(1.0)

    # Top category from items
    def extract_cats(items_list):
        cats = []
        for it in items_list:
            cat = (it.get("category") or "").lower()
            if cat: cats.append(cat)
        return cats

    cats_by_user = df.groupby("customer_id")["items_parsed"].apply(lambda rows: [it for sub in rows for it in sub])
    top_cat = cats_by_user.apply(lambda items: pd.Series(extract_cats(items)).value_counts().idxmax() if len(extract_cats(items)) else "unknown")

    agg = agg.merge(top_cat.rename("top_category"), on="customer_id", how="left")
    agg["top_category"] = agg["top_category"].fillna("unknown")
    return agg