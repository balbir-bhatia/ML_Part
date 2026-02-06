import os
import pandas as pd
from config.settings import settings

def ensure_dirs():
    os.makedirs(settings.DATA_RAW, exist_ok=True)
    os.makedirs(settings.DATA_CLEANED, exist_ok=True)
    os.makedirs(settings.DATA_FEATURES, exist_ok=True)
    os.makedirs(settings.DATA_MODELS, exist_ok=True)
    os.makedirs(settings.DATA_EVENTS, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)