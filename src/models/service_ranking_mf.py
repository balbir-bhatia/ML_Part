import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

class SimpleMFRecommender:
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_factors = None
        self.item_factors = None

    def fit(self, interactions: pd.DataFrame):
        data = interactions.copy()
        data["customer_id_enc"] = self.user_encoder.fit_transform(data["customer_id"].astype(str))
        data["service_id_enc"] = self.item_encoder.fit_transform(data["service_id"].astype(str))
        weights = data["interactions"].astype(float).values
        rows = data["customer_id_enc"].values
        cols = data["service_id_enc"].values
        n_users = data["customer_id_enc"].nunique()
        n_items = data["service_id_enc"].nunique()

        mat = csr_matrix((weights, (rows, cols)), shape=(n_users, n_items))
        uv = self.svd.fit_transform(mat)  # users x k
        vt = self.svd.components_.T       # items x k
        self.user_factors = uv
        self.item_factors = vt
        return self

    def recommend(self, customer_id: str, top_k: int = 50, exclude: list[str] | None = None):
        if exclude is None: exclude = []
        if customer_id not in self.user_encoder.classes_:
            return []
        uidx = int(self.user_encoder.transform([customer_id])[0])
        scores = self.item_factors @ self.user_factors[uidx, :]
        order = np.argsort(-scores)
        result = []
        for idx in order:
            sid = self.item_encoder.inverse_transform([idx])[0]
            if sid in exclude: continue
            result.append(sid)
            if len(result) >= top_k: break
        return result