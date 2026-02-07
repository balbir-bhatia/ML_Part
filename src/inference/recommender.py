import os
import joblib
import pandas as pd
from config import settings
from .ranking_pipeline import RankingPipeline

MODEL_FILE = os.path.join(settings.DATA_MODELS, "service_pipeline.pkl")
CAT_FILE = os.path.join(settings.DATA_FEATURES, "service_catalog.csv")
USR_FILE = os.path.join(settings.DATA_FEATURES, "user_features.csv")
USR_SVC_FILE = os.path.join(settings.DATA_FEATURES, "user_service_interactions.csv")

def load_pipeline() -> RankingPipeline:
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    service_catalog = pd.read_csv(CAT_FILE)
    user_features = pd.read_csv(USR_FILE)
    user_service = pd.read_csv(USR_SVC_FILE)
    pipe = RankingPipeline().fit(service_catalog, user_features, user_service)
    return pipe

def save_pipeline(pipe: RankingPipeline):
    os.makedirs(settings.DATA_MODELS, exist_ok=True)
    joblib.dump(pipe, MODEL_FILE)

def recommend_for_user(pipe: RankingPipeline, customer_id: str, recent_service_ids: list[str] | None = None, top_k: int = 10):
    return pipe.recommend(customer_id, recent_service_ids, top_k=top_k)