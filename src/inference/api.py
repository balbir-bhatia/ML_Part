from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

app = FastAPI()

# -----------------------------
# Configuration (paths)
# -----------------------------
BASE_DIR = Path(os.getenv("REC_SYS_BASE_DIR", Path(__file__).resolve().parents[2]))
CUSTOMERS_WITH_BUDGET = BASE_DIR / "data" / "raw" / "customers_with_budget.csv"
CAT_RANKS_ML = BASE_DIR / "data" / "features" / "user_category_rank_ml.csv"
PROD_RANKS_ML = BASE_DIR / "data" / "features" / "user_product_rank_ml.csv"
PRODUCTS_CATALOG = BASE_DIR / "data" / "reference" / "banking_products_sample.csv"

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

csv_cache = CsvCache()

# -----------------------------
# Recommendation Endpoint
# -----------------------------
from src.pipelines.run_budget_and_rank import rank_categories_ml, rank_products_ml
from src.models.lstm_spend_tune import CFG
from src.utils.logging import get_logger

logger = get_logger(__name__)

@app.post("/recommend")
async def recommend_users(customer_ids: List[str], topk: int = 3):
    logger.info(f"Generating recommendations for {len(customer_ids)} customers")
    
    # Load required data
    customers = csv_cache.load(CUSTOMERS_WITH_BUDGET)
    products = csv_cache.load(PRODUCTS_CATALOG)
    
    # Filter target customers
    target_users = customers[customers["customer_id"].isin(customer_ids)]
    if target_users.empty:
        raise HTTPException(status_code=404, detail="No matching customers found")
    
    # Generate recommendations
    category_ranks = rank_categories_ml(target_users, products)
    product_ranks = rank_products_ml(target_users, products, topk=topk)
    
    # # Format response
    # recommendations = {
    #     customer_id: {
    #         "categories": list(category_ranks[category_ranks["customer_id"] == customer_id]["product_category"])"
    #         "products": product_ranks[product_ranks["customer_id"] == customer_id].to_dict(orient="records")
    # }
    
    # return {
    #     "status": "success",
    #     "recommendations": recommendations
    # }
