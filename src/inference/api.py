# src/inference/api.py
from __future__ import annotations
import os
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------
# Configuration (paths)
# -----------------------------
BASE_DIR = Path(os.getenv("REC_SYS_BASE_DIR", Path(__file__).resolve().parents[2]))
CUSTOMERS_WITH_BUDGET = BASE_DIR / "data" / "raw" / "customers_with_budget.csv"
CAT_RANKS_ML         = BASE_DIR / "data" / "features" / "user_category_rank_ml.csv"
PROD_RANKS_ML        = BASE_DIR / "data" / "features" / "user_product_rank_ml.csv"
PRODUCTS_CATALOG     = BASE_DIR / "data" / "reference" / "banking_products_sample.csv"

# -----------------------------
# In-memory cache with mtime checks
# -----------------------------
class CsvCache:
    def __init__(self):
        self.cache: Dict[Path, Dict[str, Any]] = {}

    def load(self, path: Path, dtype: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(str(path))
        mtime = path.stat().st_mtime
        item = self.cache.get(path)
        if item and item["mtime"] == mtime:
            return item["df"]

        df = pd.read_csv(path, dtype=dtype)
        self.cache[path] = {"mtime": mtime, "df": df}
        return df

    def snapshot(self) -> Dict[str, Any]:
        out = {}
        for p, it in self.cache.items():
            out[str(p)] = {"last_loaded": it["mtime"], "last_loaded_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(it["mtime"]))}
        return out

    def clear(self):
        self.cache.clear()

csv_cache = CsvCache()

# -----------------------------
# Pydantic Schemas
# -----------------------------
class UserSummary(BaseModel):
    customer_id: str
    predicted_next_spend: float = Field(..., description="One-step-ahead spend forecast from LSTM")
    age: Optional[float] = 0
    gender: Optional[str] = "unknown"
    city: Optional[str] = "unknown"
    segment: Optional[str] = "unknown"
    current_balance: Optional[float] = 0
    txn_count_30d: Optional[float] = 0
    total_30d: Optional[float] = 0
    avg_amt_30d: Optional[float] = 0
    max_amt_30d: Optional[float] = 0

class CategoryRank(BaseModel):
    product_category: str
    score: float
    rank: int

class ProductRank(BaseModel):
    product_category: str
    sub_category: str
    product_score: float
    rank: int
    # Optional enrichment
    example_amount: Optional[float] = None
    current_credit_limit: Optional[float] = None
    account_balance: Optional[float] = None
    interest_rate: Optional[float] = None
    loan_term_months: Optional[float] = None
    rewards_points: Optional[float] = None
    eligibility_status: Optional[str] = None

class CatalogItem(BaseModel):
    product_category: str
    sub_category: str
    example_amount: Optional[float] = None
    current_credit_limit: Optional[float] = None
    account_balance: Optional[float] = None
    interest_rate: Optional[float] = None
    loan_term_months: Optional[float] = None
    rewards_points: Optional[float] = None
    eligibility_status: Optional[str] = None

class Snapshots(BaseModel):
    files: Dict[str, Any]

# -----------------------------
# App
# -----------------------------
app = FastAPI(
    title="Recommendation System API",
    version="1.0.0",
    description="Serves ML-ranked categories and products for each user."
)

# CORS (allow all; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
# -----------------------------
def _ensure_customer_exists(cust_id: str, users_df: pd.DataFrame):
    if cust_id not in set(users_df["customer_id"].astype(str)):
        raise HTTPException(status_code=404, detail=f"customer_id '{cust_id}' not found")

def _load_users() -> pd.DataFrame:
    df = csv_cache.load(CUSTOMERS_WITH_BUDGET)
    # normalize expected columns if present
    keep_cols = ["customer_id", "predicted_next_spend", "age", "gender", "city", "segment",
                 "current_balance", "txn_count_30d", "total_30d", "avg_amt_30d", "max_amt_30d"]
    for c in keep_cols:
        if c not in df.columns:
            # fill empty columns if pipeline didn't write them (older runs)
            df[c] = 0 if c not in ("gender","city","segment") else "unknown"
    df["customer_id"] = df["customer_id"].astype(str)
    # numeric coercion
    for c in ["predicted_next_spend", "age", "current_balance", "txn_count_30d", "total_30d", "avg_amt_30d", "max_amt_30d"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["gender","city","segment"]:
        df[c] = df[c].fillna("unknown").astype(str)
    return df

def _load_cat_ranks() -> pd.DataFrame:
    df = csv_cache.load(CAT_RANKS_ML)
    df["customer_id"] = df["customer_id"].astype(str)
    # normalize col names if needed
    if "score" not in df.columns:
        # backward compat if column named differently
        if "category_score" in df.columns: df.rename(columns={"category_score":"score"}, inplace=True)
    return df

def _load_prod_ranks() -> pd.DataFrame:
    df = csv_cache.load(PROD_RANKS_ML)
    df["customer_id"] = df["customer_id"].astype(str)
    return df

def _load_products() -> pd.DataFrame:
    df = csv_cache.load(PRODUCTS_CATALOG)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    # numeric coercion for enrichments
    for c in ["example_amount","current_credit_limit","account_balance","interest_rate","loan_term_months","rewards_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "product_category" in df.columns:
        df["product_category"] = df["product_category"].astype(str)
    if "sub_category" in df.columns:
        df["sub_category"] = df["sub_category"].astype(str)
    return df

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status":"ok", "ts": int(time.time())}

@app.get("/meta/snapshots", response_model=Snapshots)
def snapshots():
    # also include current mtime even if not cached yet
    files = {}
    for p in [CUSTOMERS_WITH_BUDGET, CAT_RANKS_ML, PROD_RANKS_ML, PRODUCTS_CATALOG]:
        if p.exists():
            files[str(p)] = {
                "mtime": p.stat().st_mtime,
                "mtime_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
            }
        else:
            files[str(p)] = {"error": "file_not_found"}
    # merge cache info
    files.update({f"[cache] {k}": v for k, v in csv_cache.snapshot().items()})
    return {"files": files}

@app.get("/users/{customer_id}/summary", response_model=UserSummary)
def get_user_summary(customer_id: str):
    try:
        users = _load_users()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"'{CUSTOMERS_WITH_BUDGET}' not found. Please run the pipeline first.")
    _ensure_customer_exists(customer_id, users)
    row = users[users["customer_id"] == str(customer_id)].iloc[0].to_dict()
    return UserSummary(**row)

@app.get("/users/{customer_id}/categories", response_model=List[CategoryRank])
def get_user_categories(customer_id: str, topk: int = Query(5, ge=1, le=50)):
    try:
        cats = _load_cat_ranks()
        users = _load_users()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"{e}")
    _ensure_customer_exists(customer_id, users)
    df = cats[cats["customer_id"] == str(customer_id)].copy()
    if df.empty:
        return []
    df = df.sort_values(["rank","score"], ascending=[True, False]).head(topk)
    return [CategoryRank(product_category=r["product_category"], score=float(r["score"]), rank=int(r["rank"])) 
            for _, r in df.iterrows()]

@app.get("/users/{customer_id}/products", response_model=List[ProductRank])
def get_user_products(
    customer_id: str,
    category: str = Query(..., alias="category", description="Product category (e.g., 'Credit Card', 'Loan')"),
    topk: int = Query(5, ge=1, le=100)
):
    try:
        prods = _load_prod_ranks()
        users = _load_users()
        catalog = _load_products()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"{e}")
    _ensure_customer_exists(customer_id, users)

    df = prods[(prods["customer_id"] == str(customer_id)) & (prods["product_category"].str.lower() == category.lower())].copy()
    if df.empty:
        return []

    # Enrich with product catalog attributes (optional)
    cat_min = catalog[["product_category","sub_category",
                       "example_amount","current_credit_limit","account_balance","interest_rate","loan_term_months",
                       "rewards_points","eligibility_status"]].copy()
    cat_min.columns = [c.strip().lower().replace(" ", "_") for c in cat_min.columns]
    df = df.merge(cat_min,
                  left_on=["product_category","sub_category"],
                  right_on=["product_category","sub_category"],
                  how="left")

    df = df.sort_values(["rank","product_score"], ascending=[True, False]).head(topk)

    out: List[ProductRank] = []
    for _, r in df.iterrows():
        out.append(ProductRank(
            product_category=r["product_category"],
            sub_category=r["sub_category"],
            product_score=float(r["product_score"]),
            rank=int(r["rank"]),
            example_amount=float(r["example_amount"]) if pd.notna(r.get("example_amount")) else None,
            current_credit_limit=float(r["current_credit_limit"]) if pd.notna(r.get("current_credit_limit")) else None,
            account_balance=float(r["account_balance"]) if pd.notna(r.get("account_balance")) else None,
            interest_rate=float(r["interest_rate"]) if pd.notna(r.get("interest_rate")) else None,
            loan_term_months=float(r["loan_term_months"]) if pd.notna(r.get("loan_term_months")) else None,
            rewards_points=float(r["rewards_points"]) if pd.notna(r.get("rewards_points")) else None,
            eligibility_status=str(r["eligibility_status"]) if pd.notna(r.get("eligibility_status")) else None
        ))
    return out

@app.get("/products/catalog", response_model=List[CatalogItem])
def get_catalog():
    try:
        df = _load_products()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"'{PRODUCTS_CATALOG}' not found")
    cols = ["product_category","sub_category","example_amount","current_credit_limit","account_balance",
            "interest_rate","loan_term_months","rewards_points","eligibility_status"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    # coerce numerics for clean JSON
    for c in ["example_amount","current_credit_limit","account_balance","interest_rate","loan_term_months","rewards_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return [CatalogItem(**r._asdict()) if hasattr(r, "_asdict") else CatalogItem(**r.to_dict())
            for _, r in df.iterrows()]

@app.post("/admin/reload")
def reload_all():
    """Drop in-memory cache; next call reloads from disk."""
    csv_cache.clear()
    return {"status": "reloaded"}