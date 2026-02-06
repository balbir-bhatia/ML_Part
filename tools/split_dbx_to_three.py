# tools/split_dbx_to_three.py
import os, json, numpy as np, pandas as pd

SRC_FILE = "data\output_fixed_DBX.csv"  # your uploaded file
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if "__file__" in globals() else "."
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def as_str(x):
    if pd.isna(x): return None
    if isinstance(x, float) and np.isfinite(x): return f"{int(x):d}"
    return str(x).strip()

def load_raw(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().replace(" ", "_").replace("/", "_").lower() for c in df.columns]
    for col in ["date_of_account_opening","last_transaction_date","transaction_date",
                "payment_due_date","last_credit_card_payment_date","approval_rejection_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    for col in ["account_balance","transaction_amount","account_balance_after_transaction",
                "credit_limit","credit_card_balance","minimum_payment_due","interest_rate","loan_amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "contact_number" in df.columns:
        df["contact_number"] = df["contact_number"].apply(as_str)
    return df

def latest_snapshot_per_customer(df):
    if "transaction_date" in df.columns:
        df = df.copy()
        df["__rank"] = df.groupby("customer_id")["transaction_date"].rank(method="first", ascending=False)
        return df.loc[df["__rank"] == 1].copy()
    return df.drop_duplicates("customer_id").copy()

raw = load_raw(os.path.join(BASE_DIR, SRC_FILE))

# ---------------- Customers ----------------
latest = latest_snapshot_per_customer(raw)
cust_cols = ["customer_id","first_name","last_name","age","gender","address","city",
             "contact_number","email","account_type","date_of_account_opening"]
cust_cols = [c for c in cust_cols if c in latest.columns]
customers = latest[cust_cols].drop_duplicates().reset_index(drop=True)
customers["current_balance"] = latest.get("account_balance_after_transaction",
                                 pd.Series([np.nan]*len(latest))).fillna(latest.get("account_balance"))
if "transaction_date" in raw.columns:
    customers = customers.merge(
        raw.groupby("customer_id")["transaction_date"].max().rename("last_transaction_date"),
        on="customer_id", how="left"
    )
customers["country"] = "italy"
customers["segment"] = np.where(customers["current_balance"]>=10000,"high_value",
                         np.where(customers["current_balance"]>=3000,"regular","low_value"))

# ---------------- Merchants (synthetic but deterministic) ----------------
categories = [
  ("electronics","mobile"),("electronics","audio"),("fashion","apparel"),("grocery","staples"),
  ("grocery","fruits"),("home","kitchen"),("books","education"),("sports","fitness"),
  ("beauty","skincare"),("pharmacy","otc"),("fuel","petrol"),("travel","tickets"),
  ("utilities","electricity"),("utilities","water"),("food","restaurant")
]
city_pool = sorted(customers["city"].dropna().astype(str).str.lower().unique().tolist()) or ["rome","milan","turin"]
np.random.seed(42)
n_merchants = max(30, len(categories)*2)
merchant_rows = []
for i in range(n_merchants):
    cat, sub = categories[i % len(categories)]
    merchant_rows.append({
        "merchant_id": f"M{i+1:03d}",
        "merchant_name": f"{cat.title()} {sub.title()} Store {i+1:02d}",
        "category": cat, "subcategory": sub,
        "city": np.random.choice(city_pool), "state": np.random.choice(city_pool),
        "country": "italy",
        "onboarded_at": (pd.Timestamp("2023-01-01")+pd.to_timedelta(int(np.random.randint(0,730)), unit="D")).date(),
        "rating": float(np.round(np.random.uniform(3.8,4.9),1)),
        "is_active": True
    })
merchants = pd.DataFrame(merchant_rows)

# Deterministic merchant assignment per txn
raw = raw.sort_values(["customer_id","transaction_date","transactionid"]).reset_index(drop=True)
def assign_merchant_id(row):
    t = str(row.get("transaction_type","")).lower()
    txid = row.get("transactionid")
    base = (abs(hash(str(txid))) % len(merchants))
    if "debit" in t: return merchants.iloc[base]["merchant_id"]
    if "net" in t or "neft" in t: return merchants.iloc[(base+7)%len(merchants)]["merchant_id"]
    if "credit" in t:
        if (abs(hash(str(txid)+"R")) % 100) < 40:
            return merchants.iloc[(base+3)%len(merchants)]["merchant_id"]
        subset = merchants[merchants["category"].isin(["utilities","travel"])]
        if subset.empty: return merchants.iloc[(base+5)%len(merchants)]["merchant_id"]
        idx = (abs(hash(str(txid)+"U")) % len(subset))
        return subset.iloc[idx]["merchant_id"]
    return merchants.iloc[base]["merchant_id"]
raw["merchant_id"] = raw.apply(assign_merchant_id, axis=1)

# ---------------- Transactions ----------------
rename_map = {
    "transactionid":"transaction_id", "customer_id":"customer_id",
    "transaction_date":"transaction_ts", "transaction_type":"transaction_type",
    "transaction_amount":"amount","account_balance":"balance_before",
    "account_balance_after_transaction":"balance_after","branch_id":"branch_id",
    "cardid":"card_id","card_type":"card_type","credit_limit":"credit_limit",
    "credit_card_balance":"card_balance"
}
txn = raw[list(rename_map.keys())].rename(columns=rename_map).copy()
txn["currency"] = "EUR"
txn["merchant_id"] = raw["merchant_id"]

# Fill derived balance_after where missing & inferable (keeps math consistent)
tt = txn["transaction_type"].astype(str).str.lower()
mask_debit  = tt.str.contains("debit") | tt.str.contains("neft") | tt.str.contains("net")
mask_credit = tt.str.contains("credit")
can_fill = txn["balance_after"].isna() & txn["balance_before"].notna() & txn["amount"].notna()
txn.loc[mask_debit & can_fill,  "balance_after"] = txn["balance_before"] - txn["amount"]
txn.loc[mask_credit & can_fill, "balance_after"] = txn["balance_before"] + txn["amount"]

# Single item per txn (JSON) so recommender can use services
merch_map = merchants.set_index("merchant_id")[["category","subcategory"]]
svc_name_by_cat = {
  ("electronics","mobile"):"Smartphone Accessory",
  ("electronics","audio"):"Wireless Headphones",
  ("fashion","apparel"):"T-Shirt",
  ("grocery","staples"):"Rice 5kg",
  ("grocery","fruits"):"Apples 1kg",
  ("home","kitchen"):"Non-stick Pan",
  ("books","education"):"Exam Guide",
  ("sports","fitness"):"Yoga Mat",
  ("beauty","skincare"):"Face Moisturizer",
  ("pharmacy","otc"):"OTC Medicine",
  ("fuel","petrol"):"Fuel Purchase",
  ("travel","tickets"):"Train Ticket",
  ("utilities","electricity"):"Electricity Bill",
  ("utilities","water"):"Water Bill",
  ("food","restaurant"):"Restaurant Bill",
}
items = []
for _, r in txn.iterrows():
    cat, sub = merch_map.loc[r["merchant_id"]].values if r["merchant_id"] in merch_map.index else ("misc","misc")
    nm = svc_name_by_cat.get((cat, sub), f"{cat.title()} {sub.title()} Item")
    price = float(r["amount"]) if pd.notna(r["amount"]) else 0.0
    items.append(json.dumps([{
        "service_id": f"SVC_{cat[:3]}_{sub[:3]}",
        "name": nm, "category": cat, "subcategory": sub, "qty": 1, "price": price
    }]))
txn["items"] = items

# Write
customers.to_csv(os.path.join(RAW_DIR,"customers.csv"), index=False)
merchants.to_csv(os.path.join(RAW_DIR,"merchants.csv"), index=False)
txn.to_csv(os.path.join(RAW_DIR,"transactions.csv"), index=False)

# Optional sanity check (counts)
print(f"customers={len(customers)}, merchants={len(merchants)}, transactions={len(txn)}")