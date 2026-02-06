# src/models/lstm_spend.py
import os
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
@dataclass
class LSTMSpendConfig:
    data_dir: str = "data/raw"
    model_dir: str = "data/models"
    # time granularity
    freq: str = "W"              # 'D' for daily, 'W' for weekly
    lookback: int = 12           # timesteps to look back
    horizon: int = 1             # predict next 1 period (can extend to >1)
    # training
    batch_size: int = 128
    epochs: int = 15
    lr: float = 1e-3
    seed: int = 42
    # features
    top_k_categories: int = 10   # keep top K service categories as spend features
    use_static_user_feats: bool = False  # set True to include users' age/segment/city (requires encoding)


CFG = LSTMSpendConfig()

# =========================
# UTILS
# =========================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_items(items):
    """Parses transactions.items JSON field into list[dict]."""
    if items is None or (isinstance(items, float) and math.isnan(items)):
        return []
    if isinstance(items, list):
        return items
    if isinstance(items, dict):
        return [items]
    if isinstance(items, str):
        try:
            val = json.loads(items)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return [val]
        except Exception:
            return []
    return []

def add_time_cyc_features(df, ts_col, freq):
    """Add sin/cos time features based on timestamp and frequency."""
    df = df.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df["dow"] = df[ts_col].dt.dayofweek   # 0..6
    df["month"] = df[ts_col].dt.month     # 1..12
    # sin/cos for dow
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7.0)
    # sin/cos for month
    df["mon_sin"] = np.sin(2*np.pi*df["month"]/12.0)
    df["mon_cos"] = np.cos(2*np.pi*df["month"]/12.0)
    return df

# =========================
# DATA PREP
# =========================
def load_data(data_dir: str):
    tx_path = os.path.join(data_dir, "transactions.csv")
    cust_path = os.path.join(data_dir, "customers.csv")
    merch_path = os.path.join(data_dir, "merchants.csv")

    tx = pd.read_csv(tx_path)
    customers = pd.read_csv(cust_path)
    merchants = pd.read_csv(merch_path)

    # basic normalization
    tx["transaction_ts"] = pd.to_datetime(tx["transaction_ts"], errors="coerce")
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0.0)
    tx["customer_id"] = tx["customer_id"].astype(str)
    customers["customer_id"] = customers["customer_id"].astype(str)

    # filter negative or null amounts if needed (kept as-is here)
    # tx = tx[tx["amount"] > 0]

    # parse items for categories
    tx["items_parsed"] = tx["items"].apply(parse_items) if "items" in tx.columns else [[] for _ in range(len(tx))]

    # get category from items (first item per txn; tailor if needed)
    def get_cat(row):
        items = row["items_parsed"]
        if not items:
            return "unknown"
        cat = items[0].get("category", "unknown")
        return str(cat).lower().strip() if cat else "unknown"

    tx["category"] = tx.apply(get_cat, axis=1)

    return tx, customers, merchants

def build_weekly_user_features(tx: pd.DataFrame, top_k_categories: int = 10, freq: str = "W"):
    """
    Aggregate to user x time (weekly by default):
      - total_amount
      - txn_count
      - avg_amount
      - top-K category spend fractions (per period)
      - time cyclic features
    Robust to timestamp column name and avoids Grouper(key=...) pitfalls.
    """
    df = tx.copy()

    # --- 1) Ensure we have a timestamp column ---
    ts_candidates = ["transaction_ts", "transaction_date", "tx_ts", "ts"]
    ts_col = next((c for c in ts_candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(
            f"No timestamp column found. Tried: {ts_candidates}. "
            f"Columns present: {list(df.columns)[:10]}..."
        )
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    # --- 2) Minimal columns we need ---
    if "amount" not in df.columns:
        raise ValueError("Column 'amount' not found in transactions.")
    if "customer_id" not in df.columns:
        raise ValueError("Column 'customer_id' not found in transactions.")

    # If you already created "category" from items, keep it; otherwise, fallback to 'unknown'
    if "category" not in df.columns:
        df["category"] = "unknown"
    df["category"] = df["category"].fillna("unknown").astype(str).str.lower().str.strip()

    # Keep only needed columns
    df = df[["customer_id", ts_col, "amount", "category"]]

    # --- 3) Pick global top-K categories (rest collapsed into 'other') ---
    top_cats = df["category"].value_counts().head(top_k_categories).index.tolist()

    feats = []
    for cust, grp in df.groupby("customer_id"):
        g = grp.sort_values(ts_col).set_index(ts_col)

        # Base aggregates per period using index-based resample (no key name needed)
        base = (
            g["amount"]
            .resample(freq)
            .agg(["sum", "count", "mean"])
            .rename(columns={"sum": "total_amount", "count": "txn_count", "mean": "avg_amount"})
            .fillna(0.0)
        )

        # Category amount per period → fractions
        g2 = g.assign(cat=np.where(g["category"].isin(top_cats), g["category"], "other"))
        # group on DatetimeIndex with pd.Grouper(freq=...), and category
        cat_pivot = (
            g2.groupby([pd.Grouper(freq=freq), "cat"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
        )

        # Fractions per period (avoid division by zero)
        cat_frac = cat_pivot.div(cat_pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        # Join & finalize this user's frame
        X = base.join(cat_frac, how="outer").fillna(0.0)
        X["customer_id"] = cust
        X = X.reset_index().rename(columns={ts_col: "period_start"})
        feats.append(X)

    if not feats:
        raise RuntimeError(
            "No features were produced. You may need more data or a shorter lookback/frequency."
        )

    feats = pd.concat(feats, axis=0, ignore_index=True)

    # --- 4) Add cyclic time features ---
    feats = add_time_cyc_features(feats, "period_start", freq=freq)

    # --- 5) Decide feature/target columns ---
    non_feat_cols = ["customer_id", "period_start", "dow", "month"]
    # Category fraction columns = all columns created by 'unstack' above (excluding known columns)
    candidate_cols = set(feats.columns) - set(non_feat_cols) - {"total_amount", "txn_count", "avg_amount",
                                                                "dow_sin", "dow_cos", "mon_sin", "mon_cos"}
    cat_cols = [c for c in sorted(candidate_cols)]  # sorted for stability

    feature_cols = ["txn_count", "avg_amount"] + cat_cols + ["dow_sin", "dow_cos", "mon_sin", "mon_cos"]
    target_col = "total_amount"

    feats[feature_cols + [target_col]] = feats[feature_cols + [target_col]].fillna(0.0)
    return feats, feature_cols, target_col

def build_sequences(feats: pd.DataFrame, feature_cols, target_col, lookback: int = 12, horizon: int = 1):
    """
    Build rolling sequences per user. For each user:
      X[t-lookback:t] -> y[t] (total_amount next period)
    """
    seq_X, seq_y, seq_user, seq_time = [], [], [], []
    for cust, grp in feats.groupby("customer_id"):
        g = grp.sort_values("period_start").reset_index(drop=True)
        X_mat = g[feature_cols].values.astype(np.float32)
        y_vec = g[target_col].values.astype(np.float32)

        for t in range(lookback, len(g)-horizon+1):
            seq_X.append(X_mat[t-lookback: t, :])
            seq_y.append(y_vec[t: t+horizon])  # can be >1 horizon
            seq_user.append(cust)
            seq_time.append(g.loc[t, "period_start"])

    X = np.stack(seq_X, axis=0) if seq_X else np.zeros((0, lookback, len(feature_cols)), dtype=np.float32)
    y = np.stack(seq_y, axis=0) if seq_y else np.zeros((0, horizon), dtype=np.float32)
    return X, y, np.array(seq_user), np.array(seq_time)

def time_based_split(seq_time, train_ratio=0.8):
    """
    Split sequences by time to prevent leakage:
    pick a time threshold (per global) and split.
    """
    if len(seq_time) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    times = pd.to_datetime(seq_time)
    thresh = np.quantile(times.view(np.int64), train_ratio)
    mask_train = times.view(np.int64) <= thresh
    idx_train = np.where(mask_train)[0]
    idx_val = np.where(~mask_train)[0]
    return idx_train, idx_val

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)        # (N, T, F)
        self.y = torch.from_numpy(y)        # (N, H)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# =========================
# MODEL
# =========================
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)          # (B, T, H)
        last = out[:, -1, :]           # use last hidden
        yhat = self.head(last)         # (B, horizon)
        return yhat

# =========================
# TRAIN / EVAL
# =========================
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / max(1, len(loader.dataset))

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_abs, total_sq, n = 0.0, 0.0, 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        total_loss += loss.item() * Xb.size(0)
        # MAE / RMSE
        total_abs += torch.sum(torch.abs(pred - yb)).item()
        total_sq  += torch.sum((pred - yb)**2).item()
        n += yb.numel()
    mae = total_abs / max(1, n)
    rmse = math.sqrt(total_sq / max(1, n))
    return total_loss / max(1, len(loader.dataset)), mae, rmse

def main():
    set_seed(CFG.seed)

    # 1) Load
    tx, customers, merchants = load_data(CFG.data_dir)

    # 2) Build period features
    feats, feature_cols, target_col = build_weekly_user_features(
        tx, top_k_categories=CFG.top_k_categories, freq=CFG.freq
    )

    # 3) Normalize **features + target** using train stats only
    #    We'll normalize features; target we keep in real units (or normalize and de-normalize later).
    scaler_X = StandardScaler()
    feats_sorted = feats.sort_values(["customer_id", "period_start"]).reset_index(drop=True)

    # IMPORTANT: we must avoid leakage; we’ll scale after building sequences using only train portion.
    # So first build sequences with raw values:
    X_raw, y_raw, seq_user, seq_time = build_sequences(
        feats_sorted, feature_cols, target_col, CFG.lookback, CFG.horizon
    )
    if X_raw.shape[0] == 0:
        raise RuntimeError("Not enough data to build sequences. Check frequency/lookback/horizon settings.")

    # Split by time
    idx_train, idx_val = time_based_split(seq_time, train_ratio=0.8)
    X_train_raw, X_val_raw = X_raw[idx_train], X_raw[idx_val]
    y_train, y_val = y_raw[idx_train], y_raw[idx_val]

    # Fit scaler on train features only (flatten over time steps)
    B,T,F = X_train_raw.shape
    scaler_X.fit(X_train_raw.reshape(B*T, F))

    # Transform both train/val
    def transform_X(X):
        b,t,f = X.shape
        Xf = scaler_X.transform(X.reshape(b*t, f)).reshape(b,t,f).astype(np.float32)
        return Xf

    X_train = transform_X(X_train_raw)
    X_val   = transform_X(X_val_raw)

    # 4) Dataloaders
    train_ds = SeqDataset(X_train, y_train)
    val_ds   = SeqDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, drop_last=False)

    # 5) Model / Optim
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(n_features=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.2, horizon=CFG.horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    loss_fn = nn.L1Loss()  # MAE (robust for spend)

    # 6) Train
    best_val = float("inf")
    os.makedirs(CFG.model_dir, exist_ok=True)
    save_path = os.path.join(CFG.model_dir, "lstm_spend.pt")
    for epoch in range(1, CFG.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, loss_fn, device)
        print(f"[{epoch:02d}/{CFG.epochs}] train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
              f"MAE={val_mae:.2f} | RMSE={val_rmse:.2f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "scaler_X_mean": scaler_X.mean_,
                "scaler_X_scale": scaler_X.scale_,
                "feature_cols": feature_cols,
                "cfg": CFG.__dict__
            }, save_path)
            print(f"  ↳ saved best model → {save_path}")

    # 7) Quick inference example:
    #    Take last sequence from val and predict next period spend
    if len(val_ds) > 0:
        model.eval()
        with torch.no_grad():
            xb, yb = val_ds[0]
            pred = model(xb.unsqueeze(0).to(device)).cpu().numpy().ravel()
            print(f"Example — true next spend: {yb.numpy().ravel()[0]:.2f} | predicted: {pred[0]:.2f}")

if __name__ == "__main__":
    main()