import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np

class ServiceEmbeddings:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.embeddings = None
        self.service_ids = None

    @staticmethod
    def _textify(df: pd.DataFrame) -> pd.Series:
        return (df["service_name"].fillna("") + " " + df["category"].fillna("") + " " + df["subcategory"].fillna("")).str.lower().str.strip()

    def fit(self, service_catalog: pd.DataFrame):
        self.service_ids = service_catalog["service_id"].tolist()
        text = self._textify(service_catalog)
        X = self.vectorizer.fit_transform(text)
        self.embeddings = normalize(X)
        return self

    def candidates_from_recent(self, service_catalog: pd.DataFrame, recent_service_ids: list[str], top_k: int = 50):
        if not recent_service_ids:
            return service_catalog.sort_values("popularity", ascending=False)["service_id"].head(top_k).tolist()

        idxs = [self.service_ids.index(s) for s in recent_service_ids if s in self.service_ids]
        if not idxs:
            return service_catalog.sort_values("popularity", ascending=False)["service_id"].head(top_k).tolist()

        user_vec = self.embeddings[idxs].mean(axis=0)
        sims = self.embeddings.dot(user_vec.T).toarray().ravel()
        top_idx = np.argsort(-sims)[:top_k]
        return [self.service_ids[i] for i in top_idx]