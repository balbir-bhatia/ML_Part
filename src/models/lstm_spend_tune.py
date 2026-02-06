# src/models/lstm_spend_tune.py
# Fine-tuned LSTM for user spend forecasting with user-balanced batching.

import os, json, math, random
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from sklearn.preprocessing import StandardScaler

# ---- Import existing helpers (keep as is if this works in your env) ----
# If your imports differ, you can switch to: from src.models.lstm_spend import ...
from lstm_spend import load_data, add_time_cyc_features, set_seed


# =========================
# Config
# =========================
@dataclass
class CFG:
    # data / io
    data_dir: str = "data/raw"
    model_dir: str = "data/models"

    # time & sequence
    freq: str = "D"          # daily aggregation
    lookback: int = 28       # last 28 days -> predict next day
    horizon: int = 1

    # training
    batch_size: int = 128
    epochs: int = 30
    lr: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42

    # features
    top_k_categories: int = 10
    outlier_cap_pct: float = 0.99      # cap daily total at 99th pct (and add flag)

    # model
    hidden_size: int = 96
    num_layers: int = 2
    dropout: float = 0.25

    # early stop / lr schedule
    es_patience: int = 6
    lr_patience: int = 3
    min_delta: float = 1e-3

    # ---- NEW: User-balanced batching controls ----
    sampler: str = "weighted"          # "weighted" | "grouped" | "none"
    cap_train_sequences_per_user: int | None = None  # e.g., 1000 to cap; None to disable

    # grouped sampler params (used when sampler=="grouped")
    users_per_batch: int = 8
    seqs_per_user: int = 16            # users_per_batch * seqs_per_user == batch_size


# =========================
# Feature engineering
# =========================
def parse_items(items):
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

def ensure_ts(df: pd.DataFrame):
    # robust to column name differences
    for c in ["transaction_ts", "transaction_date", "tx_ts", "ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return c
    raise ValueError("No timestamp column found in transactions.")

def build_period_features_daily(tx: pd.DataFrame, top_k_categories: int = 10, freq: str = "D", outlier_cap_pct: float = 0.99):
    """
    Aggregate to (user, day):
      - base: total_amount, txn_count, avg_amount
      - category share (top-K + other)
      - lags/EMAs/std: lag1, lag7, ema3, ema7, std7
      - optional outlier cap + flag (improves stability)
      - time cyclic features
    """
    ts_col = ensure_ts(tx)
    df = tx[["customer_id", ts_col, "amount", "items"]].copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # derive category from items (first item per txn)
    df["items_parsed"] = df["items"].apply(parse_items) if "items" in df.columns else [[] for _ in range(len(df))]

    def get_cat(items):
        if not items:
            return "unknown"
        return str(items[0].get("category", "unknown")).lower().strip() or "unknown"

    df["category"] = df["items_parsed"].apply(get_cat)

    top_cats = df["category"].value_counts().head(top_k_categories).index.tolist()

    feats = []
    for u, g in df.groupby("customer_id"):
        g = g.sort_values(ts_col).set_index(ts_col)

        base = g["amount"].resample(freq).agg(["sum", "count", "mean"]).rename(
            columns={"sum": "total_amount", "count": "txn_count", "mean": "avg_amount"}
        ).fillna(0.0)

        # outlier cap + flag on daily total
        if len(base) > 0:
            q = base["total_amount"].quantile(outlier_cap_pct)
        else:
            q = 0.0
        base["outlier_day"] = (base["total_amount"] > q).astype(float)
        base["total_amount"] = np.minimum(base["total_amount"], q if q > 0 else base["total_amount"])

        # category shares
        g2 = g.assign(cat=lambda x: np.where(x["category"].isin(top_cats), x["category"], "other"))
        cat_amt = g2.groupby([pd.Grouper(freq=freq), "cat"])["amount"].sum().unstack(fill_value=0.0)
        cat_frac = cat_amt.div(cat_amt.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        X = base.join(cat_frac, how="outer").fillna(0.0)

        # recency lags and moving stats
        X["lag1"] = X["total_amount"].shift(1).fillna(0.0)
        X["lag7"] = X["total_amount"].shift(7).fillna(0.0)
        X["ema3"] = X["total_amount"].ewm(span=3, adjust=False).mean().fillna(0.0)
        X["ema7"] = X["total_amount"].ewm(span=7, adjust=False).mean().fillna(0.0)
        X["std7"] = X["total_amount"].rolling(7).std().fillna(0.0)

        X["customer_id"] = u
        X = X.reset_index().rename(columns={ts_col: "period_start"})
        feats.append(X)

    if not feats:
        raise RuntimeError("No daily features produced. Check input data.")

    feats = pd.concat(feats, ignore_index=True)
    feats = add_time_cyc_features(feats, "period_start", freq=freq)

    # finalize feature columns
    non_feat = ["customer_id", "period_start", "dow", "month", "total_amount"]
    known = set(non_feat + ["txn_count", "avg_amount", "outlier_day",
                            "lag1", "lag7", "ema3", "ema7", "std7",
                            "dow_sin", "dow_cos", "mon_sin", "mon_cos"])
    cat_cols = [c for c in feats.columns if c not in known]
    feature_cols = ["txn_count", "avg_amount", "outlier_day",
                    "lag1", "lag7", "ema3", "ema7", "std7"] + cat_cols + \
                   ["dow_sin", "dow_cos", "mon_sin", "mon_cos"]
    target_col = "total_amount"

    feats[feature_cols + [target_col]] = feats[feature_cols + [target_col]].fillna(0.0)
    return feats, feature_cols, target_col


def build_sequences(feats, feature_cols, target_col, lookback=28, horizon=1, log_target=True):
    """
    Build rolling sequences per user.
    Returns X, y_log (for training), y_orig (for metrics), users, times.
    """
    Xs, ys_log, ys_orig, users, times = [], [], [], [], []
    for u, g in feats.groupby("customer_id"):
        g = g.sort_values("period_start").reset_index(drop=True)
        if len(g) < lookback + horizon:
            continue
        X_mat = g[feature_cols].values.astype(np.float32)
        y_vec = g[target_col].values.astype(np.float32)
        y_log = np.log1p(y_vec) if log_target else y_vec

        for t in range(lookback, len(g) - horizon + 1):
            Xs.append(X_mat[t - lookback: t, :])
            ys_log.append(y_log[t: t + horizon])
            ys_orig.append(y_vec[t: t + horizon])
            users.append(u)
            times.append(g.loc[t, "period_start"])

    if not Xs:
        return (np.zeros((0, lookback, len(feature_cols)), np.float32),
                np.zeros((0, horizon), np.float32),
                np.zeros((0, horizon), np.float32),
                np.array([]), np.array([]))
    return (np.stack(Xs), np.stack(ys_log), np.stack(ys_orig), np.array(users), np.array(times))


def time_split(seq_time, train_ratio=0.8):
    if len(seq_time) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    ts = pd.to_datetime(seq_time).view(np.int64)
    thresh = np.quantile(ts, train_ratio)
    idx_tr = np.where(ts <= thresh)[0]
    idx_va = np.where(ts > thresh)[0]
    return idx_tr, idx_va


class SeqDS(Dataset):
    def __init__(self, X, y_log, y_orig):
        self.X = torch.from_numpy(X)
        self.y_log = torch.from_numpy(y_log)
        self.y_orig = torch.from_numpy(y_orig)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y_log[i], self.y_orig[i]


# =========================
# User-balanced batching
# =========================
class UserBalancedBatchSampler(Sampler):
    """
    Yields batches with a fixed number of users per batch and approx.
    a fixed number of sequences per user (with replacement if needed).
    """
    def __init__(self, users_array, users_per_batch=8, seqs_per_user=16, drop_last=False, seed=42):
        self.users_array = np.asarray(users_array)
        self.users_per_batch = users_per_batch
        self.seqs_per_user = seqs_per_user
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        self.by_user = defaultdict(list)
        for i, u in enumerate(self.users_array):
            self.by_user[u].append(i)
        self.user_list = list(self.by_user.keys())

        self.batch_size = users_per_batch * seqs_per_user
        self.num_batches = max(1, math.ceil(len(self.users_array) / self.batch_size))

    def __iter__(self):
        for _ in range(self.num_batches):
            self.rng.shuffle(self.user_list)
            picked = self.user_list[:self.users_per_batch]
            batch = []
            for u in picked:
                idxs = self.by_user[u]
                # with replacement to avoid short users starving
                for _ in range(self.seqs_per_user):
                    batch.append(self.rng.choice(idxs))
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# =========================
# Model
# =========================
class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden=96, layers=2, dropout=0.25, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon)   # predicts log spend if log_target=True
        )
    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, H)
        last = out[:, -1, :]
        return self.head(last)


# =========================
# Train / Eval
# =========================
def train_epoch(m, loader, opt, loss_fn, device, grad_clip=1.0):
    m.train()
    tot = 0.0
    for xb, yb_log, _ in loader:
        xb, yb_log = xb.to(device), yb_log.to(device)
        opt.zero_grad()
        pred_log = m(xb)
        loss = loss_fn(pred_log, yb_log)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
        opt.step()
        tot += loss.item() * xb.size(0)
    return tot / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(m, loader, loss_fn, device):
    m.eval()
    tot, mae, mse, n = 0.0, 0.0, 0.0, 0
    for xb, yb_log, yb_orig in loader:
        xb = xb.to(device)
        pred_log = m(xb).cpu()
        # optimize on log scale
        loss = loss_fn(pred_log, yb_log)
        tot += loss.item() * xb.size(0)

        # metrics on original currency scale
        pred = torch.expm1(pred_log)
        mae += torch.sum(torch.abs(pred - yb_orig)).item()
        mse += torch.sum((pred - yb_orig) ** 2).item()
        n += yb_orig.numel()
    return (tot / max(1, len(loader.dataset)),
            mae / max(1, n),
            math.sqrt(mse / max(1, n)))


def build_train_val_loaders(cfg: CFG, tr_ds: SeqDS, va_ds: SeqDS, train_users_array: np.ndarray):
    """
    Construct DataLoaders with user-balanced strategies.
    """
    if cfg.cap_train_sequences_per_user:
        # cap sequences per user to limit dominance
        user_counts = defaultdict(int)
        keep_idx = []
        for i, u in enumerate(train_users_array):
            if user_counts[u] < cfg.cap_train_sequences_per_user:
                keep_idx.append(i)
                user_counts[u] += 1
        # filter dataset tensors by keep_idx
        tr_ds.X = tr_ds.X[keep_idx]
        tr_ds.y_log = tr_ds.y_log[keep_idx]
        tr_ds.y_orig = tr_ds.y_orig[keep_idx]
        train_users_array = train_users_array[keep_idx]
        print(f"Applied cap: kept {len(keep_idx)} train sequences.")

    if cfg.sampler == "weighted":
        # inverse-frequency weights per user
        uniq, counts = np.unique(train_users_array, return_counts=True)
        c_map = {u: c for u, c in zip(uniq, counts)}
        weights = np.array([1.0 / c_map[u] for u in train_users_array], dtype=np.float64)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=False)
    elif cfg.sampler == "grouped":
        # user-balanced grouped sampler
        assert cfg.users_per_batch * cfg.seqs_per_user == cfg.batch_size, \
            "users_per_batch * seqs_per_user must equal batch_size"
        ubs = UserBalancedBatchSampler(
            users_array=train_users_array,
            users_per_batch=cfg.users_per_batch,
            seqs_per_user=cfg.seqs_per_user,
            seed=cfg.seed
        )
        train_loader = DataLoader(tr_ds, batch_sampler=ubs)
    else:
        # default shuffling
        train_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    val_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def run_once(hparams=None):
    hp = hparams or {}
    cfg = CFG()
    for k, v in (hp.items() if isinstance(hp, dict) else []):
        setattr(cfg, k, v)

    set_seed(cfg.seed)

    # 1) Load data
    tx, customers, merchants = load_data(cfg.data_dir)

    # 2) Build daily features
    feats, feature_cols, target_col = build_period_features_daily(
        tx, top_k_categories=cfg.top_k_categories, freq=cfg.freq, outlier_cap_pct=cfg.outlier_cap_pct
    )
    feats = feats.sort_values(["customer_id", "period_start"]).reset_index(drop=True)

    # 3) Build sequences (log target)
    X_raw, y_log, y_orig, users, times = build_sequences(
        feats, feature_cols, target_col, cfg.lookback, cfg.horizon, log_target=True
    )
    if X_raw.shape[0] == 0:
        raise RuntimeError("Not enough sequences. Reduce lookback or ensure daily aggregation produced enough rows.")

    # 4) Time split
    idx_tr, idx_va = time_split(times, 0.8)
    X_tr, X_va = X_raw[idx_tr], X_raw[idx_va]
    y_tr_log, y_va_log = y_log[idx_tr], y_log[idx_va]
    y_tr, y_va = y_orig[idx_tr], y_orig[idx_va]
    train_users_array = np.array(users)[idx_tr]

    # 5) Scale features using train only
    B, T, F = X_tr.shape
    scaler = StandardScaler().fit(X_tr.reshape(B * T, F))

    def tfm(X):
        b, t, f = X.shape
        return scaler.transform(X.reshape(b * t, f)).reshape(b, t, f).astype(np.float32)

    X_tr, X_va = tfm(X_tr), tfm(X_va)

    # 6) DataLoaders (with user-balanced strategy)
    tr_ds = SeqDS(X_tr, y_tr_log, y_tr)
    va_ds = SeqDS(X_va, y_va_log, y_va)
    train_loader, val_loader = build_train_val_loaders(cfg, tr_ds, va_ds, train_users_array)

    print(f"Train sequences: {len(tr_ds)} | Val sequences: {len(va_ds)} | Features: {len(feature_cols)}")
    if cfg.sampler != "none":
        print(f"Sampler: {cfg.sampler} | "
              f"users_per_batch={cfg.users_per_batch} | seqs_per_user={cfg.seqs_per_user}")

    # 7) Model / Optim / Loss / Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(n_features=len(feature_cols),
                          hidden=cfg.hidden_size,
                          layers=cfg.num_layers,
                          dropout=cfg.dropout,
                          horizon=cfg.horizon).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.1)  # smoother than MAE on log scale

    # ReduceLROnPlateau (no verbose arg for broad compatibility)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=cfg.lr_patience)

    # 8) Train loop with early stopping
    best_val, bad_epochs = float("inf"), 0
    os.makedirs(cfg.model_dir, exist_ok=True)
    save_path = os.path.join(cfg.model_dir, "lstm_spend_tuned.pt")

    for e in range(1, cfg.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn, device, grad_clip=cfg.grad_clip)
        val_loss, val_mae, val_rmse = evaluate(model, val_loader, loss_fn, device)
        print(f"[{e:02d}/{cfg.epochs}] tr(log-loss)={tr_loss:.4f} | "
              f"val(log-loss)={val_loss:.4f} | MAE={val_mae:.2f} | RMSE={val_rmse:.2f}")

        # LR schedule + notify on LR drops
        prev_lr = opt.param_groups[0]['lr']
        lr_sched.step(val_loss)
        new_lr = opt.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"  ↳ LR reduced: {prev_lr:.2e} → {new_lr:.2e}")

        if val_loss + cfg.min_delta < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save({
                "state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "feature_cols": feature_cols,
                "cfg": asdict(cfg)
            }, save_path)
            print(f"  ↳ saved best → {save_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.es_patience:
                print("Early stopping.")
                break

    return save_path


def main():
    # Optional tiny sweep; comment out entries to make it quicker.
    trials = [
        {},  # default CFG
        {"hidden_size": 128, "dropout": 0.2, "lr": 3e-4},
        {"hidden_size": 64,  "dropout": 0.3, "lr": 7e-4, "weight_decay": 5e-5,
         "sampler": "grouped", "users_per_batch": 8, "seqs_per_user": 16},
    ]
    best = None
    for i, hp in enumerate(trials, 1):
        print(f"\n=== Trial {i}/{len(trials)}: {hp} ===")
        path = run_once(hp)
        if best is None:
            best = path
    print(f"\nBest checkpoint: {best}")


if __name__ == "__main__":
    main()