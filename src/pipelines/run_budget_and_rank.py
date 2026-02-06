# src/pipelines/run_budget_and_rank.py
"""
Pipeline:
  1) Load tuned LSTM checkpoint -> per-user forecast (next-period spend)
  2) Write forecasts CSV + add column to customers.csv -> customers_with_budget.csv
  3) Load products catalog (banking_products_sample.csv)
  4) Build clean user features (exclude PII like address/phone/email)
  5) Score & rank product CATEGORIES per user -> user_category_rank.csv

Run:
  python src/pipelines/run_budget_and_rank.py
  # with custom paths:
  python src/pipelines/run_budget_and_rank.py ^
    --customers data/raw/customers.csv ^
    --transactions data/raw/transactions.csv ^
    --products data/reference/banking_products_sample.csv ^
    --model data/models/lstm_spend_tuned.pt
"""

import os
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# --- Import your tuned training helpers (same ones used for training) ---
# If your imports differ, adjust to: from lstm_spend_tune import ...
# Import lstm_spend_tune from src/models when running scripts from repo root
try:
    from src.models.lstm_spend_tune import (
        CFG, set_seed, load_data, build_period_features_daily, LSTMRegressor
    )
except Exception:
    # Fallback: add project root to sys.path, then import via 'models...'
    import os, sys
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src/pipelines
    PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)                           # add repo root

    # Now src is importable as a package; import models.*
    from src.models.lstm_spend_tune import (
        CFG, set_seed, load_data, build_period_features_daily, LSTMRegressor
    )
# -------------------- Utility: load checkpoint with scaler --------------------
def load_ckpt(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    # PyTorch 2.6 safety: weights_only=False allowed here since it is your own file
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    scaler = StandardScaler()
    # stored either as list or numpy arrays; coerce to np.array
    scaler.mean_  = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)

    feature_cols = ckpt["feature_cols"]
    cfg_dict = ckpt.get("cfg", {})  # training-time CFG dict
    return ckpt, scaler, feature_cols, cfg_dict


def cfg_from_ckpt(cfg_dict: dict) -> CFG:
    """Clone CFG with training-time hyperparams saved in checkpoint."""
    c = CFG()
    for k, v in cfg_dict.items():
        if hasattr(c, k):
            setattr(c, k, v)
    return c


# -------------------- Forecast per user --------------------
def forecast_per_user(cfg: CFG, model_path: str):
    """
    Returns:
      forecasts_df: DataFrame[customer_id, predicted_next_spend]
      feats: per-user/day feature frame (used downstream)
      feature_cols: list of feature columns in feats
    """
    set_seed(cfg.seed)

    # Load raw data (reuses your loader; only transactions are required here)
    tx, customers, merchants = load_data(cfg.data_dir)

    # Build daily features with SAME settings used during training
    feats, feature_cols, target_col = build_period_features_daily(
        tx,
        top_k_categories=cfg.top_k_categories,
        freq=cfg.freq,
        outlier_cap_pct=getattr(cfg, "outlier_cap_pct", 0.99),
        global_cap=getattr(cfg, "global_cap", True),
        cap_floor=getattr(cfg, "cap_floor", 25.0),
    )
    feats = feats.sort_values(["customer_id", "period_start"]).reset_index(drop=True)

    # Prepare last window per user
    user_last_X, users = [], []
    for u, g in feats.groupby("customer_id"):
        g = g.sort_values("period_start")
        if len(g) < cfg.lookback:
            continue
        X_last = g[feature_cols].values[-cfg.lookback:].astype(np.float32)  # (T,F)
        user_last_X.append(X_last)
        users.append(u)
    if not user_last_X:
        return pd.DataFrame(columns=["customer_id", "predicted_next_spend"]), feats, feature_cols

    X = np.stack(user_last_X, axis=0)   # (B,T,F)

    # Load checkpoint (model + scaler)
    ckpt, scaler, feature_cols_ckpt, _ = load_ckpt(model_path)
    if feature_cols_ckpt != feature_cols:
        missing = [c for c in feature_cols_ckpt if c not in feature_cols]
        extra   = [c for c in feature_cols if c not in feature_cols_ckpt]
        raise RuntimeError(
            "Feature columns differ from training.\n"
            f"In checkpoint only: {missing}\n"
            f"In current build only: {extra}\n"
            "Make sure productized feature builder matches training config."
        )

    # Scale features
    B, T, F = X.shape
    X = scaler.transform(X.reshape(B*T, F)).reshape(B, T, F).astype(np.float32)

    # Build model with same dims
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

    # Predict
    preds = []
    with torch.no_grad():
        for i in range(0, B, 1024):
            y_log = model(torch.from_numpy(X[i:i+1024]).to(device)).cpu().numpy().ravel()
            y = np.expm1(y_log)  # inverse log1p -> currency
            preds.append(y)
    preds = np.concatenate(preds)

    forecasts_df = pd.DataFrame({"customer_id": users, "predicted_next_spend": preds.round(2)})
    return forecasts_df, feats, feature_cols


# -------------------- Update customers.csv with forecast --------------------
def update_customers_with_budget(customers_path: str, forecasts_df: pd.DataFrame, out_path: str):
    customers = pd.read_csv(customers_path)
    # unify types for merge
    customers["customer_id"] = customers["customer_id"].astype(str)
    forecasts_df["customer_id"] = forecasts_df["customer_id"].astype(str)

    updated = customers.merge(forecasts_df, on="customer_id", how="left")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    updated.to_csv(out_path, index=False)
    return updated


# -------------------- Load products catalog --------------------
def load_products(products_path: str):
    if not os.path.exists(products_path):
        raise FileNotFoundError(f"Products CSV not found: {products_path}")
    products = pd.read_csv(products_path)
    # Normalize column names
    products.columns = [c.strip().lower().replace(" ", "_") for c in products.columns]
    # We’ll keep category and sub-category as strings
    for c in ["product_category", "sub_category"]:
        if c in products.columns:
            products[c] = products[c].astype(str)
    return products


# -------------------- Build compact user features for ranking --------------------
def build_user_profile(customers_with_budget: pd.DataFrame,
                       transactions_path: str,
                       lookback_days: int = 30):
    """
    Returns user_features with columns:
      customer_id, age, gender, city, segment, current_balance, predicted_next_spend,
      txn_count_30d, total_30d, avg_amt_30d, max_amt_30d
    Drops PII-like columns (address, contact_number, email).
    """
    # Keep non-PII columns if present
    keep_cols = ["customer_id", "age", "gender", "city", "segment",
                 "current_balance", "predicted_next_spend"]
    keep_cols = [c for c in keep_cols if c in customers_with_budget.columns]

    users = customers_with_budget[keep_cols].copy()
    users["customer_id"] = users["customer_id"].astype(str)

    # Transactions aggregation (last N days)
    tx = pd.read_csv(transactions_path)
    # normalize types
    if "transaction_ts" in tx.columns:
        tx["transaction_ts"] = pd.to_datetime(tx["transaction_ts"], errors="coerce")
    elif "transaction_date" in tx.columns:
        tx["transaction_ts"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
    else:
        raise ValueError("transactions.csv must have 'transaction_ts' or 'transaction_date'")

    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    tx["customer_id"] = tx["customer_id"].astype(str)

    cutoff = tx["transaction_ts"].max() - pd.Timedelta(days=lookback_days)
    tx30 = tx[tx["transaction_ts"] >= cutoff].copy()

    agg = tx30.groupby("customer_id")["amount"].agg(
        txn_count_30d = "count",
        total_30d = "sum",
        max_amt_30d = "max",
    ).reset_index()
    agg["avg_amt_30d"] = (agg["total_30d"] / agg["txn_count_30d"]).replace([np.inf, -np.inf], 0).fillna(0)

    user_features = users.merge(agg, on="customer_id", how="left")
    for c in ["txn_count_30d","total_30d","max_amt_30d","avg_amt_30d","predicted_next_spend","current_balance"]:
        if c in user_features.columns:
            user_features[c] = pd.to_numeric(user_features[c], errors="coerce").fillna(0.0)

    # Age cleanup if present
    if "age" in user_features.columns:
        user_features["age"] = pd.to_numeric(user_features["age"], errors="coerce").fillna(0.0)

    # Fill missing categoricals
    for c in ["gender","city","segment"]:
        if c in user_features.columns:
            user_features[c] = user_features[c].fillna("unknown").astype(str)

    return user_features


# -------------------- Category ranking model (deterministic scoring) --------------------
def pick_product_amount(row: pd.Series) -> float:
    """
    Choose a monetary anchor from the products row.
    Preference order: example_amount -> account_balance -> current_credit_limit
    """
    for key in ["example_amount", "account_balance", "current_credit_limit"]:
        if key in row.index:
            v = pd.to_numeric(row[key], errors="coerce")
            if pd.notna(v):
                return float(v)
    return np.nan


def normalize_closeness(x, ref, eps=1e-6):
    """Return closeness in [0,1]: 1 - |x-ref|/(|ref|+eps). If ref is nan, return 0."""
    if ref is None or np.isnan(ref):
        return 0.0
    return float(max(0.0, 1.0 - (abs(float(x) - float(ref)) / (abs(float(ref)) + eps))))


def compute_category_scores(user_row: pd.Series, products: pd.DataFrame) -> pd.DataFrame:
    """
    Score each top-level product_category for a single user.
    Heuristic, fast, label-free. Uses user's budget/balance/activity vs product anchors.
    """
    # Pre-calc per category anchor amount (median of available rows)
    products = products.copy()
    for c in ["example_amount","account_balance","current_credit_limit"]:
        if c in products.columns:
            products[c] = pd.to_numeric(products[c], errors="coerce")

    products["anchor_amount"] = products.apply(pick_product_amount, axis=1)
    cat_anchor = products.groupby("product_category")["anchor_amount"].median().to_dict()

    u_budget = float(user_row.get("predicted_next_spend", 0.0) or 0.0)
    u_balance = float(user_row.get("current_balance", 0.0) or 0.0)
    u_txn_count = float(user_row.get("txn_count_30d", 0.0) or 0.0)
    u_avg = float(user_row.get("avg_amt_30d", 0.0) or 0.0)
    u_total = float(user_row.get("total_30d", 0.0) or 0.0)
    u_max = float(user_row.get("max_amt_30d", 0.0) or 0.0)
    u_age = float(user_row.get("age", 0.0) or 0.0)

    rows = []
    for category in sorted(products["product_category"].unique()):
        ref_amt = cat_anchor.get(category, np.nan)

        # Base components
        budget_fit = normalize_closeness(u_budget, ref_amt)
        balance_fit = normalize_closeness(u_balance, ref_amt)
        activity_fit = min(1.0, (u_txn_count / 30.0)) * 0.5 + min(1.0, (u_avg / (ref_amt + 1e-6))) * 0.5

        # Category-specific blend (simple but effective)
        if category.lower().startswith("loan"):
            # Loans: larger amounts align with higher max spend; users with low balance vs anchor also score
            score = 0.45 * normalize_closeness(u_max, ref_amt) + 0.35 * (1 - normalize_closeness(u_balance, ref_amt)) + 0.20 * activity_fit
        elif category.lower().startswith("investment"):
            # Investments: high balance + decent budget
            score = 0.65 * balance_fit + 0.25 * budget_fit + 0.10 * activity_fit
        elif category.lower().startswith("savings"):
            # Savings: reasonable balance & active transactions
            score = 0.50 * balance_fit + 0.30 * activity_fit + 0.20 * budget_fit
        elif category.lower().startswith("insurance"):
            # Insurance: age & recent activity
            age_factor = min(1.0, u_age / 70.0)
            score = 0.60 * age_factor + 0.20 * activity_fit + 0.20 * budget_fit
        elif category.lower().startswith("credit card"):
            # Cards: budget close to credit anchors + average txn size
            score = 0.50 * budget_fit + 0.50 * min(1.0, u_avg / (ref_amt + 1e-6))
        else:
            # Fallback: budget + activity
            score = 0.6 * budget_fit + 0.4 * activity_fit

        rows.append({"product_category": category, "score": float(score)})

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = out["score"].rank(method="dense", ascending=False).astype(int)
    return out


def rank_categories_for_all_users(user_features: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, urow in user_features.iterrows():
        per_cat = compute_category_scores(urow, products)
        per_cat["customer_id"] = urow["customer_id"]
        records.append(per_cat)
    if not records:
        return pd.DataFrame(columns=["customer_id","product_category","score","rank"])
    res = pd.concat(records, ignore_index=True)
    # order columns
    return res[["customer_id","product_category","score","rank"]]


# -------------------- Main CLI --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--customers", default="data/raw/customers.csv")
    p.add_argument("--transactions", default="data/raw/transactions.csv")
    p.add_argument("--products", default="data/reference/banking_products_sample.csv")
    p.add_argument("--model", default="data/models/lstm_spend_tuned.pt")
    args = p.parse_args()

    # 1) Load model cfg + forecast
    ckpt, _, _, cfg_dict = load_ckpt(args.model)
    cfg = cfg_from_ckpt(cfg_dict)

    print("→ Forecasting next-period spend per user ...")
    forecasts_df, feats, feature_cols = forecast_per_user(cfg, args.model)

    # write forecasts for downstream systems
    os.makedirs("data/features", exist_ok=True)
    forecasts_path = "data/features/user_forecast_spend_tuned.csv"
    forecasts_df.to_csv(forecasts_path, index=False)
    print(f"✓ Saved forecasts: {forecasts_path} (rows={len(forecasts_df)})")

    # 2) Update customers with budget column
    print("→ Updating customers.csv with predicted_next_spend ...")
    updated_customers_path = "data/raw/customers_with_budget.csv"
    updated_customers = update_customers_with_budget(args.customers, forecasts_df, updated_customers_path)
    print(f"✓ Saved: {updated_customers_path} (rows={len(updated_customers)})")

    # 3) Products
    products = load_products(args.products)
    print(f"→ Loaded products: {args.products}, categories={products['product_category'].nunique()}")

    # 4) Build compact user features (drop PII, add activity features)
    print("→ Building user features for ranking ...")
    user_features = build_user_profile(updated_customers, args.transactions, lookback_days=30)
    print(f"✓ Users with features: {len(user_features)}")

    # 5) Rank product categories per user
    print("→ Scoring & ranking product categories per user ...")
    ranks = rank_categories_for_all_users(user_features, products)

    out_rank = "data/features/user_category_rank.csv"
    os.makedirs(os.path.dirname(out_rank), exist_ok=True)
    ranks.to_csv(out_rank, index=False)
    print(f"✓ Saved category ranks: {out_rank} (rows={len(ranks)})")

    # Show a small preview
    print("\nTop-3 sample (first user):")
    if len(ranks) > 0:
        first_user = ranks["customer_id"].iloc[0]
        print(ranks[ranks["customer_id"]==first_user].sort_values("rank").head(3))


if __name__ == "__main__":
    main()