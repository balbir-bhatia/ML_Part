import os
import pandas as pd

from config.settings import settings
from src.data_loading.load_postgres import load_postgres_tables
from src.data_loading.load_mongodb import load_mongodb_collections
from src.data_loading.load_csvs import load_csvs
from src.data_loading.merge_data import merge_core

from src.preprocessing.cleaning import clean_transactions, clean_users, clean_merchants
from src.feature_engineering.user_features import build_user_features
from src.feature_engineering.service_features import build_service_catalog
from src.feature_engineering.interaction_features import build_user_service_interactions
from src.feature_engineering.feature_store import ensure_dirs, save_csv

from src.inference.ranking_pipeline import RankingPipeline
from src.inference.recommender import save_pipeline

def main(source: str = "csv", limit: int | None = None):
    ensure_dirs()

    # 1) Load
    if source == "postgres":
        users, merchants, tx = load_postgres_tables(limit=limit)
    elif source == "mongo":
        users, merchants, tx = load_mongodb_collections(limit=limit)
    else:
        users, merchants, tx = load_csvs(
            customers_path=os.path.join(settings.DATA_RAW, "customers.csv"),
            merchants_path=os.path.join(settings.DATA_RAW, "merchants.csv"),
            transactions_path=os.path.join(settings.DATA_RAW, "transactions.csv")
        )

    # 2) Clean
    users_c = clean_users(users)
    merchants_c = clean_merchants(merchants)
    tx_c = clean_transactions(tx)

    # 3) Merge (context only)
    master = merge_core(users_c, merchants_c, tx_c)

    # 4) Features
    user_feat = build_user_features(master)
    service_catalog = build_service_catalog(master)
    user_service = build_user_service_interactions(master)

    # Save features
    save_csv(user_feat, os.path.join(settings.DATA_FEATURES, "user_features.csv"))
    save_csv(service_catalog, os.path.join(settings.DATA_FEATURES, "service_catalog.csv"))
    save_csv(user_service, os.path.join(settings.DATA_FEATURES, "user_service_interactions.csv"))

    # 5) Train pipeline
    pipe = RankingPipeline().fit(service_catalog, user_feat, user_service)

    # 6) Save model
    save_pipeline(pipe)
    print("Pipeline trained and saved.")

if __name__ == "__main__":
    main(source=os.getenv('\SOURCE', '\csv'), limit=None)