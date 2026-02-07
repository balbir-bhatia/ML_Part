# ================================================================
# FastAPI server for ML ranking pipeline
# - Provides endpoints to rank categories/products for a customer
# - Uses the existing pipeline functions you've implemented
# ================================================================

import os
import sys
import time
import traceback
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd

# Ensure project root on sys.path (mirrors your pipeline import pattern)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---- Import your pipeline functions ----
from src.pipelines.run_budget_and_rank import (
    load_ckpt, cfg_from_ckpt,
    forecast_per_user, update_customers_with_budget,
    load_products, build_user_profile,
    load_or_train,
    rank_categories_ml, rank_products_ml
)

# ----------------------------
# Configuration via env vars
# ----------------------------
CUSTOMERS_PATH      = os.getenv("CUSTOMERS_PATH",      "data/raw/customers.csv")
TRANSACTIONS_PATH   = os.getenv("TRANSACTIONS_PATH",   "data/raw/transactions.csv")
PRODUCTS_PATH       = os.getenv("PRODUCTS_PATH",       "data/reference/banking_products_sample.csv")
LSTM_MODEL_PATH     = os.getenv("LSTM_MODEL_PATH",     "data/models/lstm_spend_tuned.pt")
DEFAULT_TOPK        = int(os.getenv("DEFAULT_TOPK",    "3"))
LOOKBACK_DAYS       = int(os.getenv("LOOKBACK_DAYS",   "30"))
FORCE_TRAIN         = os.getenv("FORCE_TRAIN",         "0") == "1"

app = FastAPI(
    title="Budget & Rank API",
    description="Serve category and product rankings for a given customer using your ML pipeline.",
    version="1.0.0"
)

# Optional CORS (adjust in real deployments)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ----------------------------
# Global runtime state
# ----------------------------
STATE: Dict[str, Any] = {
    "loaded_at": None,
    "products": None,          # pd.DataFrame
    "updated_customers": None, # pd.DataFrame
    "user_features": None,     # pd.DataFrame
    "cat_model": None,         # sklearn Pipeline
    "prod_model": None,        # sklearn Pipeline
    "cfg_dict": None,          # dict
}

def _require(condition: bool, msg: str):
    if not condition:
        raise HTTPException(status_code=500, detail=msg)

def _refresh_state(force_train: bool = False):
    """
    Builds/loads all runtime artifacts once.
    """
    try:
        # 1) Load LSTM checkpoint + cfg
        ckpt, _, _, cfg_dict = load_ckpt(LSTM_MODEL_PATH)
        cfg = cfg_from_ckpt(cfg_dict)

        # 2) Forecast next-period spend (per-user)
        forecasts_df, feats, feature_cols = forecast_per_user(cfg, LSTM_MODEL_PATH)
        os.makedirs("data/features", exist_ok=True)
        forecasts_df.to_csv("data/features/user_forecast_spend_tuned.csv", index=False)

        # 3) Merge into customers_with_budget.csv
        updated_customers = update_customers_with_budget(
            CUSTOMERS_PATH, forecasts_df, "data/raw/customers_with_budget.csv"
        )

        # 4) Load products
        products = load_products(PRODUCTS_PATH)

        # 5) Build non-PII user features
        user_features = build_user_profile(updated_customers, TRANSACTIONS_PATH, lookback_days=LOOKBACK_DAYS)

        # 6) Load or train rankers
        cat_model, prod_model = load_or_train(user_features, products, force_train=force_train)

        # Save into global STATE
        STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        STATE["products"] = products
        STATE["updated_customers"] = updated_customers
        STATE["user_features"] = user_features
        STATE["cat_model"] = cat_model
        STATE["prod_model"] = prod_model
        STATE["cfg_dict"] = cfg_dict

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Required file not found: {e}. Ensure data paths & LSTM checkpoint exist."
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize runtime state: {e}"
        )

# Initialize on startup
@app.on_event("startup")
def startup_event():
    _refresh_state(force_train=FORCE_TRAIN)

# ----------------------------
# Schemas
# ----------------------------
class RankingRequest(BaseModel):
    customer_id: str
    topk: Optional[int] = None

class CategoryRanking(BaseModel):
    product_category: str
    score: float
    rank: int

class ProductRanking(BaseModel):
    product_category: str
    sub_category: str
    product_score: float
    rank: int

class RankingResponse(BaseModel):
    customer_id: str
    topk: int
    categories: List[CategoryRanking]
    products: List[ProductRanking]

# ----------------------------
# Helpers
# ----------------------------
def _get_user_row(customer_id: str) -> pd.DataFrame:
    uf = STATE["user_features"]
    _require(uf is not None and not uf.empty, "User features are not loaded.")
    row = uf[uf["customer_id"].astype(str) == str(customer_id)]
    return row

def _format_category_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Expect columns: product_category, score, rank
    out = []
    for _, r in df[["product_category", "score", "rank"]].iterrows():
        out.append({
            "product_category": str(r["product_category"]),
            "score": float(r["score"]),
            "rank": int(r["rank"])
        })
    return out

def _format_product_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Expect columns: product_category, sub_category, product_score, rank
    out = []
    for _, r in df[["product_category", "sub_category", "product_score", "rank"]].iterrows():
        out.append({
            "product_category": str(r["product_category"]),
            "sub_category": str(r["sub_category"]),
            "product_score": float(r["product_score"]),
            "rank": int(r["rank"])
        })
    return out

# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_at": STATE["loaded_at"],
        "customers": int(0 if STATE["updated_customers"] is None else len(STATE["updated_customers"])),
        "user_features": int(0 if STATE["user_features"] is None else len(STATE["user_features"])),
        "products": int(0 if STATE["products"] is None else len(STATE["products"])),
        "models": {
            "category_ranker_loaded": STATE["cat_model"] is not None,
            "product_ranker_loaded": STATE["prod_model"] is not None
        }
    }

@app.get("/customers/sample")
def sample_customers(limit: int = Query(10, ge=1, le=100)):
    uf = STATE["user_features"]
    _require(uf is not None and not uf.empty, "User features are not loaded.")
    ids = uf["customer_id"].astype(str).head(limit).tolist()
    return {"count": len(ids), "customer_ids": ids}

@app.get("/refresh")
def refresh(force_train: bool = Query(False, description="Force retraining of rankers.")):
    _refresh_state(force_train=force_train)
    return {"status": "refreshed", "loaded_at": STATE["loaded_at"], "force_trained": force_train}

@app.get("/rankings/{customer_id}", response_model=RankingResponse)
def get_rankings(customer_id: str, topk: Optional[int] = Query(None, ge=1)):
    # Resolve topk
    k = int(topk) if topk is not None else DEFAULT_TOPK

    # Validate state
    _require(STATE["cat_model"] is not None and STATE["prod_model"] is not None, "Models are not loaded.")
    _require(STATE["products"] is not None, "Products are not loaded.")

    # Fetch user row
    row = _get_user_row(customer_id)
    if row is None or row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in user features.")

    # Rank categories (for this single user)
    cat_df = rank_categories_ml(STATE["cat_model"], row, STATE["products"])
    # Rank products (for this single user)
    prod_df = rank_products_ml(STATE["prod_model"], row, STATE["products"], topk=k)

    # Optionally filter category topk as well for brevity (comment out if you want all)
    cat_df = cat_df.sort_values(["rank", "product_category"])
    cat_df = cat_df[cat_df["rank"] <= k].reset_index(drop=True)

    return RankingResponse(
        customer_id=str(customer_id),
        topk=k,
        categories=_format_category_rows(cat_df),
        products=_format_product_rows(prod_df)
    )