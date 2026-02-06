# src/models/lstm_spend_test.py
"""
Test runner for the fine-tuned LSTM spend model (checkpoint-aware).
- Loads tuned checkpoint saved by lstm_spend_tune.py (data/models/lstm_spend_tuned.pt)
- Rebuilds daily features & sequences using the SAME config as training (from checkpoint)
- Evaluates MAE/RMSE on a time-based validation split
- Optional: predict for one user and/or write batch forecasts to CSV

Usage:
  python src/models/lstm_spend_test.py
  python src/models/lstm_spend_test.py --user_id C003
  python src/models/lstm_spend_test.py --write_csv
"""

import os
import json
import math
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ---- import from tuned training script ----
# If imports differ in your project, adjust to: from src.models.lstm_spend_tune import ...
from lstm_spend_tune import (
    CFG, set_seed, load_data,
    build_period_features_daily, build_sequences, time_split,
    SeqDS, LSTMRegressor
)

# ----------------- Checkpoint loading -----------------
def _load_ckpt(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # PyTorch 2.6+ ships weights_only=True default; for your own trusted file use False:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # restore scaler
    scaler = StandardScaler()
    # Some older saves may have lists; coerce to np.array to be safe
    scaler.mean_  = np.array(ckpt["scaler_mean"], dtype=np.float64)
    scaler.scale_ = np.array(ckpt["scaler_scale"], dtype=np.float64)
    feature_cols = ckpt["feature_cols"]
    cfg_dict = ckpt.get("cfg", {})  # dict of training CFG
    return ckpt, scaler, feature_cols, cfg_dict

def _cfg_from_ckpt(cfg_dict: dict) -> CFG:
    """Create a CFG instance that mirrors training hyper-params from checkpoint."""
    cfg_eval = CFG()
    for k, v in cfg_dict.items():
        # only set attributes that exist on CFG
        if hasattr(cfg_eval, k):
            setattr(cfg_eval, k, v)
    return cfg_eval

# ----------------- Evaluation / Prediction -----------------
def evaluate_split(cfg: CFG, feats: pd.DataFrame, feature_cols: list[str], target_col: str,
                   ckpt, scaler: StandardScaler, show_examples: int = 5):
    """Evaluate on time-based validation split using the SAME model config as training."""
    feats_sorted = feats.sort_values(["customer_id", "period_start"]).reset_index(drop=True)

    # sequences with log1p target (as in training)
    X_raw, y_log, y_orig, users, times = build_sequences(
        feats_sorted, feature_cols, target_col,
        lookback=cfg.lookback, horizon=cfg.horizon, log_target=True
    )
    if X_raw.shape[0] == 0:
        raise RuntimeError("Not enough sequences. Try reducing lookback or confirm daily features are populated.")

    idx_tr, idx_va = time_split(times, 0.8)
    X_tr, X_va = X_raw[idx_tr], X_raw[idx_va]
    y_tr_log, y_va_log = y_log[idx_tr], y_log[idx_va]
    y_tr, y_va = y_orig[idx_tr], y_orig[idx_va]
    users_va = np.array(users)[idx_va]
    times_va = pd.to_datetime(np.array(times)[idx_va])

    # scale features with TRAIN scaler
    B, T, F = X_tr.shape
    def tfm(X):
        b, t, f = X.shape
        return scaler.transform(X.reshape(b*t, f)).reshape(b, t, f).astype(np.float32)

    X_tr, X_va = tfm(X_tr), tfm(X_va)

    # datasets
    va_ds = SeqDS(X_va, y_va_log, y_va)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)

    # model with SAME dimensions as training
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

    loss_fn = torch.nn.SmoothL1Loss(beta=0.1)  # same as training
    tot_log, mae, mse, n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for xb, yb_log, yb_orig in va_loader:
            xb = xb.to(device)
            pred_log = model(xb).cpu()
            # log-space loss
            loss = loss_fn(pred_log, yb_log)
            tot_log += loss.item() * xb.size(0)
            # original-space metrics
            pred = torch.expm1(pred_log)
            mae += torch.sum(torch.abs(pred - yb_orig)).item()
            mse += torch.sum((pred - yb_orig)**2).item()
            n += yb_orig.numel()

    val_log = tot_log / max(1, len(va_ds))
    val_mae = mae / max(1, n)
    val_rmse = math.sqrt(mse / max(1, n))

    print("\n================ Validation Metrics ================")
    print(f" val(log-loss) = {val_log:.4f} (SmoothL1 on log1p(y))")
    print(f" val(MAE)      = {val_mae:.2f}")
    print(f" val(RMSE)     = {val_rmse:.2f}")

    # show a few validation predictions
    if show_examples > 0 and len(va_ds) > 0:
        print("\n------------- Example Predictions (val) -------------")
        step = max(1, len(va_ds)//show_examples)
        shown = 0
        for i in range(0, len(va_ds), step):
            xb, yb_log, yb_orig = va_ds[i]
            with torch.no_grad():
                pred_log = model(xb.unsqueeze(0).to(device)).cpu().numpy().ravel()
            pred = float(np.expm1(pred_log)[0])
            true = float(yb_orig.numpy().ravel()[0])
            print(f" user={users_va[i]:<8} period={str(times_va[i])} | true={true:8.2f} | pred={pred:8.2f}")
            shown += 1
            if shown >= show_examples:
                break
        print("----------------------------------------------------\n")

def predict_one_user(cfg: CFG, feats: pd.DataFrame, feature_cols: list[str], ckpt, scaler: StandardScaler, user_id: str):
    """Build last lookback window for a user and print next-period prediction."""
    g = feats[feats["customer_id"].astype(str) == str(user_id)].sort_values("period_start")
    if len(g) < cfg.lookback:
        print(f"[WARN] Not enough history for user {user_id}. Need >= {cfg.lookback}, have {len(g)}.")
        return

    X_last = g[feature_cols].values[-cfg.lookback:].astype(np.float32)[None, :, :]
    X_last = scaler.transform(X_last.reshape(cfg.lookback, len(feature_cols))).reshape(1, cfg.lookback, len(feature_cols)).astype(np.float32)

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

    with torch.no_grad():
        pred_log = model(torch.from_numpy(X_last).to(device)).cpu().numpy().ravel()
        pred = float(np.expm1(pred_log)[0])
    print(f"Next-period spend prediction for user {user_id}: {pred:.2f}")

def batch_write_forecasts(cfg: CFG, feats: pd.DataFrame, feature_cols: list[str], ckpt, scaler: StandardScaler,
                          out_path="data/features/user_forecast_spend_tuned.csv"):
    """Write one-step-ahead forecast per user from their latest window."""
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

    feats_sorted = feats.sort_values(["customer_id", "period_start"])
    rows = []
    for u, g in feats_sorted.groupby("customer_id"):
        if len(g) < cfg.lookback:
            continue
        X_last = g[feature_cols].values[-cfg.lookback:].astype(np.float32)[None, :, :]
        X_last = scaler.transform(X_last.reshape(cfg.lookback, len(feature_cols))).reshape(1, cfg.lookback, len(feature_cols)).astype(np.float32)
        with torch.no_grad():
            pred_log = model(torch.from_numpy(X_last).to(device)).cpu().numpy().ravel()
            pred = float(np.expm1(pred_log)[0])
        rows.append({"customer_id": u, "predicted_next_spend": round(pred, 2)})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved forecasts → {out_path}  (rows={len(rows)})")



# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, default=None, help="Print next-period prediction for this customer_id")
    parser.add_argument("--write_csv", action="store_true", help="Write per-user next-period predictions to CSV")
    parser.add_argument("--examples", type=int, default=5, help="How many example predictions to print from validation")
    args = parser.parse_args()

    # Base config + seed (we'll override from checkpoint)
    cfg_base = CFG()
    set_seed(cfg_base.seed)

    # 1) Load checkpoint early to get the EXACT training config
    ckpt_path = os.path.join(cfg_base.model_dir, "lstm_spend_tuned.pt")
    ckpt, scaler, feature_cols_ckpt, cfg_dict = _load_ckpt(ckpt_path)
    cfg = _cfg_from_ckpt(cfg_dict)  # this cfg mirrors training hyper-params

    # 2) Load raw and build DAILY features using training config (freq/top_k/outlier_cap)
    tx, customers, merchants = load_data(cfg.data_dir)
    feats, feature_cols_now, target_col = build_period_features_daily(
        tx,
        top_k_categories=cfg.top_k_categories,
        freq=cfg.freq,
        outlier_cap_pct=getattr(cfg, "outlier_cap_pct", 0.99)
    )

    # 3) Ensure feature columns match training (should, if configs match)
    if feature_cols_now != feature_cols_ckpt:
        # helpful message to reconcile diffs
        missing = [c for c in feature_cols_ckpt if c not in feature_cols_now]
        extra   = [c for c in feature_cols_now if c not in feature_cols_ckpt]
        raise RuntimeError(
            "Feature columns differ from training!\n"
            f"- In checkpoint only: {missing}\n"
            f"- In current build only: {extra}\n"
            "Make sure top_k_categories/freq/outlier_cap/etc. match training."
        )

    # 4) Evaluate on validation
    evaluate_split(cfg, feats, feature_cols_now, target_col, ckpt, scaler, show_examples=args.examples)

    # 5) Single user (optional)
    if args.user_id:
        predict_one_user(cfg, feats, feature_cols_now, ckpt, scaler, args.user_id)

    # 6) Batch CSV (optional)
    if args.write_csv:
        out_csv = "data/features/user_forecast_spend_tuned.csv"
        batch_write_forecasts(cfg, feats, feature_cols_now, ckpt, scaler, out_csv)

if __name__ == "__main__":
    main()