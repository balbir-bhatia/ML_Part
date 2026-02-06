import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

def build_training_frame(user_features: pd.DataFrame, service_catalog: pd.DataFrame, user_service: pd.DataFrame, negatives_per_positive: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positives = user_service[["customer_id","service_id"]].drop_duplicates().copy()
    positives["label"] = 1

    all_users = positives["customer_id"].unique()
    all_services = service_catalog["service_id"].unique()
    neg_rows = []
    for u in all_users:
        user_services = set(positives[positives["customer_id"]==u]["service_id"])
        choices = [s for s in all_services if s not in user_services]
        if not choices: continue
        n_pos = max(1, len(user_services))
        n_neg = min(len(choices), n_pos * negatives_per_positive)
        neg_sample = rng.choice(choices, size=n_neg, replace=False)
        for s in neg_sample:
            neg_rows.append((u, s, 0))
    negatives = pd.DataFrame(neg_rows, columns=["customer_id","service_id","label"])

    Xy = pd.concat([positives, negatives], ignore_index=True)
    Xy = Xy.merge(user_features, on="customer_id", how="left")
    Xy = Xy.merge(service_catalog, on="service_id", how="left")

    Xy["recency_days"] = Xy["recency_days"].fillna(Xy["recency_days"].median())
    Xy["frequency"] = Xy["frequency"].fillna(0)
    Xy["monetary"] = Xy["monetary"].fillna(0)
    Xy["avg_amount"] = Xy["avg_amount"].fillna(Xy["avg_amount"].median())
    Xy["price_sensitivity"] = Xy["price_sensitivity"].fillna(1.0)
    for col in ["category","subcategory","top_category"]:
        if col in Xy.columns:
            Xy[col] = Xy[col].fillna("unknown")

    return Xy

def train_ranker(Xy: pd.DataFrame):
    y = Xy["label"].astype(int)
    numeric = [c for c in ["recency_days","frequency","monetary","avg_amount","price_sensitivity","popularity","avg_price"] if c in Xy.columns]
    categorical = [c for c in ["category","subcategory","top_category"] if c in Xy.columns]
    X = Xy[numeric + categorical]

    ct = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough"
    )

    if LGBM_AVAILABLE:
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)

    pipe = Pipeline(steps=[("prep", ct), ("clf", clf)])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, val_pred)
    return pipe, {"val_auc": auc, "features_numeric": numeric, "features_categorical": categorical}