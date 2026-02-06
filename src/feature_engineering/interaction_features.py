import pandas as pd

def build_user_service_interactions(transactions_clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in transactions_clean.iterrows():
        for it in r["items_parsed"]:
            sid = str(it.get("service_id") or it.get("sku") or "").strip()
            if not sid: continue
            qty = it.get("qty", 1)
            price = it.get("price", 0.0)
            rows.append({
                "customer_id": str(r.get("customer_id")),
                "service_id": sid,
                "qty": qty,
                "spent": price * qty,
                "transaction_ts": r.get("transaction_ts")
            })
    if not rows:
        return pd.DataFrame(columns=["customer_id","service_id","interactions","total_spent","avg_spent","last_ts","recency_days"])

    df = pd.DataFrame(rows)
    df["transaction_ts"] = pd.to_datetime(df["transaction_ts"], errors="coerce")
    now = df["transaction_ts"].max()
    agg = df.groupby(["customer_id","service_id"]).agg(
        interactions=("qty","sum"),
        total_spent=("spent","sum"),
        avg_spent=("spent","mean"),
        last_ts=("transaction_ts","max")
    ).reset_index()
    agg["recency_days"] = (now - agg["last_ts"]).dt.days.clip(lower=0)
    return agg