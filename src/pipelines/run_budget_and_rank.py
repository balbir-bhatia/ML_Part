# ================================================================
#  src/pipelines/run_budget_and_rank.py  (ML ranking + NaN-safe)
# ================================================================
"""
Pipeline (ML ranking version):
  1) Load tuned LSTM checkpoint → forecast next-period spend per user
  2) Add predicted_next_spend to customers → customers_with_budget.csv
  3) Build non-PII user features
  4) TRAIN or LOAD ML rankers:
        - CategoryRanker (GradientBoostingRegressor)
        - ProductRanker  (GradientBoostingRegressor)
     (both with in-pipeline imputers so NaNs never break training)
  5) Predict ML scores + rank:
        → data/features/user_category_rank_ml.csv
        → data/features/user_product_rank_ml.csv

TRAIN:
  python src/pipelines/run_budget_and_rank.py --train

PREDICT ONLY:
  python src/pipelines/run_budget_and_rank.py
"""

import os, sys, argparse, warnings
from typing import Tuple
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# LSTM utilities import
# ---------------------------------------------------------------------
try:
    from src.models.lstm_spend_tune import (
        CFG, set_seed, load_data, build_period_features_daily, LSTMRegressor
    )
except Exception:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.models.lstm_spend_tune import (
        CFG, set_seed, load_data, build_period_features_daily, LSTMRegressor
    )

# ---------------- OHE wrapper for sklearn compatibility ----------------
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------------- Safe min/max on possibly-NaN series ------------------
def _safe_minmax(series: pd.Series):
    arr = pd.to_numeric(series, errors="coerce").values
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    return float(np.min(arr)), float(np.max(arr))

# ---------------- Load tuned LSTM checkpoint ---------------------------
def load_ckpt(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    scaler = StandardScaler()
    scaler.mean_  = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    feature_cols = ckpt["feature_cols"]
    cfg_dict = ckpt["cfg"]
    return ckpt, scaler, feature_cols, cfg_dict

def cfg_from_ckpt(cfg_dict: dict) -> CFG:
    c = CFG()
    for k, v in cfg_dict.items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c

# ================================================================
# 1) FORECAST NEXT-PERIOD SPEND PER USER (LSTM)
# ================================================================
def forecast_per_user(cfg: CFG, model_path: str):
    set_seed(cfg.seed)
    tx, customers, merchants = load_data(cfg.data_dir)

    feats, feature_cols, _ = build_period_features_daily(
        tx,
        top_k_categories=cfg.top_k_categories,
        freq=cfg.freq,
        outlier_cap_pct=cfg.outlier_cap_pct,
        global_cap=cfg.global_cap,
        cap_floor=cfg.cap_floor,
    )
    feats = feats.sort_values(["customer_id","period_start"]).reset_index(drop=True)

    X_list, users = [], []
    for u, g in feats.groupby("customer_id"):
        g = g.sort_values("period_start")
        if len(g) < cfg.lookback:
            continue
        X_last = g[feature_cols].values[-cfg.lookback:].astype(np.float32)
        X_list.append(X_last); users.append(u)

    if not X_list:
        return pd.DataFrame(columns=["customer_id","predicted_next_spend"]), feats, feature_cols

    X = np.stack(X_list, axis=0)
    ckpt, scaler, feature_cols_ckpt, _ = load_ckpt(model_path)
    if feature_cols_ckpt != feature_cols:
        raise RuntimeError("Feature cols mismatch with checkpoint. Retrain or align configs.")

    B, T, F = X.shape
    X = scaler.transform(X.reshape(B*T, F)).reshape(B, T, F).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        n_features=len(feature_cols),
        hidden=cfg.hidden_size,
        layers=cfg.num_layers,
        dropout=cfg.dropout,
        horizon=cfg.horizon
    ).to(device)
    model.load_state_dict(ckpt["state"], strict=True)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, B, 512):
            batch = torch.from_numpy(X[i:i+512]).to(device)
            y_log = model(batch).cpu().numpy().ravel()
            preds.append(np.expm1(y_log))
    preds = np.concatenate(preds)

    df = pd.DataFrame({
        "customer_id": [str(u) for u in users],
        "predicted_next_spend": preds.round(2)
    })
    return df, feats, feature_cols

# ================================================================
# 2) UPDATE CUSTOMERS WITH BUDGET
# ================================================================
def update_customers_with_budget(customers_path, forecasts_df, out_path):
    customers = pd.read_csv(customers_path)
    customers["customer_id"] = customers["customer_id"].astype(str)
    forecasts_df["customer_id"] = forecasts_df["customer_id"].astype(str)
    merged = customers.merge(forecasts_df, on="customer_id", how="left")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    return merged

# ================================================================
# 3) LOAD PRODUCTS
# ================================================================
def load_products(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    for c in ["product_category","sub_category"]:
        if c in df: df[c] = df[c].astype(str)
    for c in ["example_amount","account_balance","current_credit_limit","interest_rate","loan_term_months","rewards_points"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ================================================================
# 4) BUILD NON-PII USER PROFILE
# ================================================================
def build_user_profile(customers_with_budget: pd.DataFrame,
                       transactions_path: str,
                       lookback_days: int = 30):
    keep = ["customer_id","age","gender","city","segment","current_balance","predicted_next_spend"]
    keep = [c for c in keep if c in customers_with_budget.columns]
    users = customers_with_budget[keep].copy()
    users["customer_id"] = users["customer_id"].astype(str)

    tx = pd.read_csv(transactions_path)
    if "transaction_ts" in tx.columns:
        tx["transaction_ts"] = pd.to_datetime(tx["transaction_ts"], errors="coerce")
    else:
        tx["transaction_ts"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
    tx["customer_id"] = tx["customer_id"].astype(str)
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)

    cutoff = tx["transaction_ts"].max() - pd.Timedelta(days=lookback_days)
    tx = tx[tx["transaction_ts"] >= cutoff]

    ag = tx.groupby("customer_id")["amount"].agg(
        txn_count_30d="count",
        total_30d="sum",
        max_amt_30d="max"
    ).reset_index()
    ag["avg_amt_30d"] = (ag["total_30d"] / ag["txn_count_30d"]).replace([np.inf,-np.inf],0)

    users = users.merge(ag, on="customer_id", how="left")
    for c in ["txn_count_30d","total_30d","avg_amt_30d","max_amt_30d","current_balance","predicted_next_spend","age"]:
        if c in users: users[c] = pd.to_numeric(users[c], errors="coerce").fillna(0)

    if "city" in users:
        top = users["city"].value_counts().head(30).index
        users["city"] = users["city"].where(users["city"].isin(top), "other")

    for c in ["gender","city","segment"]:
        if c in users: users[c] = users[c].fillna("unknown").astype(str)

    return users

# ================================================================
# 5) TEACHER (HEURISTICS) FOR PSEUDO-LABELS
# ================================================================
def _anchor_amount(row):
    for k in ["example_amount","account_balance","current_credit_limit"]:
        v = row.get(k, np.nan)
        if pd.notna(v): return float(v)
    return np.nan

def _normalize_closeness(x, ref, eps=1e-6):
    if ref is None or np.isnan(ref): return 0
    return max(0.0, 1 - abs(float(x)-float(ref)) / (abs(float(ref))+eps))

def teacher_category_score(u, cat_anchor, category):
    ref = cat_anchor.get(category, np.nan)
    def clos(x): return _normalize_closeness(x, ref)

    u_b  = float(u.get("predicted_next_spend",0))
    u_bal= float(u.get("current_balance",0))
    u_tx = float(u.get("txn_count_30d",0))
    u_av = float(u.get("avg_amt_30d",0))
    u_mx = float(u.get("max_amt_30d",0))
    u_ag = float(u.get("age",0))

    budget_fit = clos(u_b)
    bal_fit    = clos(u_bal)
    activity   = min(1.0,u_tx/30)*0.5 + min(1.0,u_av/(ref+1e-6))*0.5

    cat_l = category.lower()
    if cat_l.startswith("loan"):       return 0.45*clos(u_mx) + 0.35*(1-bal_fit) + 0.20*activity
    if cat_l.startswith("investment"): return 0.65*bal_fit + 0.25*budget_fit + 0.10*activity
    if cat_l.startswith("savings"):    return 0.50*bal_fit + 0.30*activity + 0.20*budget_fit
    if cat_l.startswith("insurance"):  return 0.60*min(1.0,u_ag/70) + 0.20*activity + 0.20*budget_fit
    if cat_l.startswith("credit card"):return 0.50*budget_fit + 0.50*min(1.0,u_av/(ref+1e-6))
    return 0.6*budget_fit + 0.4*activity

# ================================================================
# BUILD CANDIDATES
# ================================================================
def build_cat_candidates(user_features, products):
    pr = products.copy()
    pr["anchor"] = products.apply(_anchor_amount, axis=1)
    cat_anchor = pr.groupby("product_category")["anchor"].median().to_dict()

    rows = []
    for _, u in user_features.iterrows():
        for c in sorted(products["product_category"].unique()):
            rows.append({
                "customer_id": u["customer_id"],
                "product_category": c,
                "predicted_next_spend": u["predicted_next_spend"],
                "current_balance": u["current_balance"],
                "txn_count_30d": u["txn_count_30d"],
                "avg_amt_30d": u["avg_amt_30d"],
                "total_30d": u["total_30d"],
                "max_amt_30d": u["max_amt_30d"],
                "age": u["age"],
                "gender": u["gender"],
                "city": u["city"],
                "segment": u["segment"],
                "cat_anchor": cat_anchor.get(c, np.nan),
                "y_cat": teacher_category_score(u, cat_anchor, c)
            })
    return pd.DataFrame(rows)

def build_product_candidates(user_features, products):
    # Precompute category stats (safe)
    cat_stats = {}
    for cat, g in products.groupby("product_category"):
        rmin,rmax = _safe_minmax(g["interest_rate"]) if "interest_rate" in g else (np.nan,np.nan)
        wmin,wmax = _safe_minmax(g["rewards_points"]) if "rewards_points" in g else (np.nan,np.nan)
        cat_stats[cat] = {"rate_min":rmin,"rate_max":rmax,"rew_min":wmin,"rew_max":wmax}

    def teacher_prod(u,p):
        cat = p["product_category"]
        ref = _anchor_amount(p)
        def clos(x): return _normalize_closeness(x, ref)

        u_b  = float(u["predicted_next_spend"])
        u_bal= float(u["current_balance"])
        u_tx = float(u["txn_count_30d"])
        u_av = float(u["avg_amt_30d"])
        u_mx = float(u["max_amt_30d"])
        u_ag = float(u["age"])

        budget_fit = clos(u_b)
        bal_fit    = clos(u_bal)
        activity   = min(1.0,u_tx/30)
        st         = cat_stats.get(cat, {})
        rate,term,rews = p.get("interest_rate"),p.get("loan_term_months"),p.get("rewards_points")

        def _scale(v,vmin,vmax,inv=False):
            if v is None or np.isnan(v) or np.isnan(vmin) or np.isnan(vmax) or vmax<=vmin:
                return 0
            r=(v-vmin)/(vmax-vmin); r=max(0,min(1,r))
            return 1-r if inv else r

        if cat.lower().startswith("loan"):
            rate_s=_scale(rate,st["rate_min"],st["rate_max"],inv=True)
            term_pref=1 - min(1, abs((term or 36)-36)/36)
            amt_fit=0.55*clos(u_mx)+0.45*budget_fit
            return 0.5*amt_fit+0.35*rate_s+0.15*term_pref
        if cat.lower().startswith("investment"):
            return 0.55*bal_fit+0.25*budget_fit+0.20*activity
        if cat.lower().startswith("savings"):
            senior=0.1 if ("senior" in str(p["sub_category"]).lower() and u_ag>=60) else 0
            return 0.55*bal_fit+0.25*activity+0.20*budget_fit+senior
        if cat.lower().startswith("insurance"):
            age_f=min(1,u_ag/70)
            return 0.60*age_f+0.20*activity+0.20*budget_fit
        if cat.lower().startswith("credit card"):
            limit_fit=clos(u_av*25)
            rew_s=_scale(rews,st["rew_min"],st["rew_max"])
            return 0.55*limit_fit+0.45*rew_s
        return 0.6*budget_fit+0.4*activity

    rows = []
    for _, u in user_features.iterrows():
        for _, p in products.iterrows():
            rows.append({
                "customer_id": u["customer_id"],
                "product_category": p["product_category"],
                "sub_category": p["sub_category"],
                "predicted_next_spend": u["predicted_next_spend"],
                "current_balance": u["current_balance"],
                "txn_count_30d": u["txn_count_30d"],
                "avg_amt_30d": u["avg_amt_30d"],
                "total_30d": u["total_30d"],
                "max_amt_30d": u["max_amt_30d"],
                "age": u["age"],
                "gender": u["gender"],
                "city": u["city"],
                "segment": u["segment"],
                "example_amount": p["example_amount"],
                "current_credit_limit": p["current_credit_limit"],
                "account_balance": p["account_balance"],
                "interest_rate": p["interest_rate"],
                "loan_term_months": p["loan_term_months"],
                "rewards_points": p["rewards_points"],
                "eligibility_status": str(p["eligibility_status"]),
                "y_prod": teacher_prod(u,p)
            })
    return pd.DataFrame(rows)

# ================================================================
# 6) TRAIN ML MODELS (with in-pipeline IMPUTERS)
# ================================================================
def train_category_ranker(cat_df, model_path):
    num_cols = ["predicted_next_spend","current_balance","txn_count_30d",
                "avg_amt_30d","total_30d","max_amt_30d","age","cat_anchor"]
    cat_cols = ["product_category","gender","city","segment"]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe())
        ]), cat_cols)
    ])

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )
    pipe = Pipeline([("pre", pre), ("gbr", model)])

    X = cat_df[num_cols + cat_cols]
    y = cat_df["y_cat"].fillna(0.0)

    pipe.fit(X, y)
    joblib.dump(pipe, model_path)
    print("✓ CategoryRanker trained:", model_path)
    return pipe

def train_product_ranker(prod_df, model_path):
    num_cols = ["predicted_next_spend","current_balance","txn_count_30d","avg_amt_30d",
                "total_30d","max_amt_30d","age",
                "example_amount","current_credit_limit","account_balance",
                "interest_rate","loan_term_months","rewards_points"]
    cat_cols = ["product_category","sub_category","gender","city","segment","eligibility_status"]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), [c for c in num_cols if c in prod_df.columns]),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe())
        ]), [c for c in cat_cols if c in prod_df.columns])
    ])

    model = GradientBoostingRegressor(
        n_estimators=250, max_depth=3, learning_rate=0.05, random_state=42
    )
    pipe = Pipeline([("pre", pre), ("gbr", model)])

    X = prod_df[[c for c in num_cols if c in prod_df.columns] + [c for c in cat_cols if c in prod_df.columns]]
    y = prod_df["y_prod"].fillna(0.0)

    pipe.fit(X, y)
    joblib.dump(pipe, model_path)
    print("✓ ProductRanker trained:", model_path)
    return pipe

def load_or_train(user_features, products, force_train=False):
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    cat_model_path = os.path.join(model_dir, "category_ranker.pkl")
    prod_model_path = os.path.join(model_dir, "product_ranker.pkl")

    if force_train or (not os.path.exists(cat_model_path) or not os.path.exists(prod_model_path)):
        print("→ Generating training data (user × category, user × product) ...")
        cat_df = build_cat_candidates(user_features, products)
        prod_df = build_product_candidates(user_features, products)
        print(f"Training pairs: category={len(cat_df):,}, product={len(prod_df):,}")
        cat_model = train_category_ranker(cat_df, cat_model_path)
        prod_model = train_product_ranker(prod_df, prod_model_path)
    else:
        print("→ Loading pre-trained rankers ...")
        cat_model = joblib.load(cat_model_path)
        prod_model = joblib.load(prod_model_path)

    return cat_model, prod_model

# ================================================================
# 7) PREDICT & RANK
# ================================================================
def rank_categories_ml(cat_model, user_features, products):
    pr = products.copy()
    pr["anchor"] = products.apply(_anchor_amount, axis=1)
    cat_anchor = pr.groupby("product_category")["anchor"].median().to_dict()

    rows = []
    cats = sorted(products["product_category"].unique())
    for _, u in user_features.iterrows():
        for c in cats:
            rows.append({
                "customer_id": u["customer_id"],
                "product_category": c,
                "predicted_next_spend": u["predicted_next_spend"],
                "current_balance": u["current_balance"],
                "txn_count_30d": u["txn_count_30d"],
                "avg_amt_30d": u["avg_amt_30d"],
                "total_30d": u["total_30d"],
                "max_amt_30d": u["max_amt_30d"],
                "age": u["age"],
                "gender": u["gender"],
                "city": u["city"],
                "segment": u["segment"],
                "cat_anchor": cat_anchor.get(c,np.nan)
            })
    df = pd.DataFrame(rows)
    X = df[["predicted_next_spend","current_balance","txn_count_30d","avg_amt_30d",
            "total_30d","max_amt_30d","age","cat_anchor",
            "product_category","gender","city","segment"]]
    df["score"] = cat_model.predict(X)
    df["rank"] = df.groupby("customer_id")["score"].rank(method="dense", ascending=False).astype(int)
    return df.sort_values(["customer_id","rank","product_category"])

def rank_products_ml(prod_model, user_features, products, topk=3):
    rows = []
    for _, u in user_features.iterrows():
        for _, p in products.iterrows():
            rows.append({
                "customer_id": u["customer_id"],
                "product_category": p["product_category"],
                "sub_category": p["sub_category"],
                "predicted_next_spend": u["predicted_next_spend"],
                "current_balance": u["current_balance"],
                "txn_count_30d": u["txn_count_30d"],
                "avg_amt_30d": u["avg_amt_30d"],
                "total_30d": u["total_30d"],
                "max_amt_30d": u["max_amt_30d"],
                "age": u["age"],
                "gender": u["gender"],
                "city": u["city"],
                "segment": u["segment"],
                "example_amount": p["example_amount"],
                "current_credit_limit": p["current_credit_limit"],
                "account_balance": p["account_balance"],
                "interest_rate": p["interest_rate"],
                "loan_term_months": p["loan_term_months"],
                "rewards_points": p["rewards_points"],
                "eligibility_status": str(p["eligibility_status"]),
            })
    df = pd.DataFrame(rows)
    X = df[["predicted_next_spend","current_balance","txn_count_30d","avg_amt_30d",
            "total_30d","max_amt_30d","age",
            "example_amount","current_credit_limit","account_balance",
            "interest_rate","loan_term_months","rewards_points",
            "product_category","sub_category","gender","city","segment","eligibility_status"]]
    df["product_score"] = prod_model.predict(X)
    df["rank"] = df.groupby(["customer_id","product_category"])["product_score"]\
                   .rank(method="dense", ascending=False).astype(int)
    if topk:
        df = df[df["rank"] <= topk]
    return df.sort_values(["customer_id","product_category","rank","sub_category"])

# ================================================================
# MAIN
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customers", default="data/raw/customers.csv")
    ap.add_argument("--transactions", default="data/raw/transactions.csv")
    ap.add_argument("--products", default="data/reference/banking_products_sample.csv")
    ap.add_argument("--model", default="data/models/lstm_spend_tuned.pt")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()

    # 1) Forecast
    ckpt, _, _, cfg_dict = load_ckpt(args.model)
    cfg = cfg_from_ckpt(cfg_dict)
    print("→ Forecasting next-period spend per user ...")
    forecasts_df, feats, feature_cols = forecast_per_user(cfg, args.model)
    os.makedirs("data/features", exist_ok=True)
    forecasts_df.to_csv("data/features/user_forecast_spend_tuned.csv", index=False)

    # 2) Update customers
    updated_customers = update_customers_with_budget(args.customers, forecasts_df, "data/raw/customers_with_budget.csv")

    # 3) Load products
    products = load_products(args.products)

    # 4) User features
    user_features = build_user_profile(updated_customers, args.transactions)

    # 5) Load/train rankers
    cat_model, prod_model = load_or_train(user_features, products, force_train=args.train)

    # 6) Predict category ranks
    print("→ Predicting category ranks ...")
    cat_scores = rank_categories_ml(cat_model, user_features, products)
    cat_scores.to_csv("data/features/user_category_rank_ml.csv", index=False)

    # 7) Predict product ranks
    print("→ Predicting product ranks ...")
    prod_scores = rank_products_ml(prod_model, user_features, products, topk=args.topk)
    prod_scores.to_csv("data/features/user_product_rank_ml.csv", index=False)

    print("\n✓ DONE")

if __name__ == "__main__":
    main()