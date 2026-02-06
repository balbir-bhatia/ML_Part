import pandas as pd
from collections import Counter

def build_service_catalog(transactions_clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in transactions_clean.iterrows():
        for it in r["items_parsed"]:
            sid = str(it.get("service_id") or it.get("sku") or "").strip()
            if not sid: continue
            rows.append({
                "service_id": sid,
                "service_name": it.get("name") or sid,
                "category": (it.get("category") or "").lower().strip(),
                "subcategory": (it.get("subcategory") or "").lower().strip(),
                "price": it.get("price"),
                "merchant_id": r.get("merchant_id"),
                "transaction_ts": r.get("transaction_ts"),
                "amount": r.get("amount"),
            })
    if not rows:
        return pd.DataFrame(columns=["service_id","service_name","category","subcategory","avg_price","popularity","merchant_count"])

    df = pd.DataFrame(rows)
    cat = df.groupby("service_id").agg(
        service_name=("service_name","last"),
        category=("category", lambda x: Counter(x).most_common(1)[0][0] if len(x)>0 else "unknown"),
        subcategory=("subcategory", lambda x: Counter(x).most_common(1)[0][0] if len(x)>0 else "unknown"),
        avg_price=("price","mean"),
        popularity=("service_id","count"),
        merchant_count=("merchant_id", lambda x: x.nunique())
    ).reset_index()

    cat["avg_price"] = cat["avg_price"].fillna(cat["avg_price"].median() if not cat["avg_price"].dropna().empty else 0.0)
    cat["category"] = cat["category"].fillna("unknown")
    cat["subcategory"] = cat["subcategory"].fillna("unknown")
    return cat