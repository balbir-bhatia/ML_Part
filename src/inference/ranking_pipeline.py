import pandas as pd
from src.models.embeddings import ServiceEmbeddings
from src.models.service_ranking_mf import SimpleMFRecommender
from src.models.service_ranking_lgbm import train_ranker, build_training_frame

class RankingPipeline:
    def __init__(self):
        self.emb = None
        self.mf = None
        self.rank_model = None
        self.train_meta = None
        self.service_catalog = None
        self.user_features = None

    def fit(self, service_catalog: pd.DataFrame, user_features: pd.DataFrame, user_service: pd.DataFrame):
        self.service_catalog = service_catalog
        self.user_features = user_features
        self.emb = ServiceEmbeddings(max_features=5000).fit(service_catalog)
        self.mf = SimpleMFRecommender(n_components=50).fit(user_service)
        Xy = build_training_frame(user_features, service_catalog, user_service)
        self.rank_model, self.train_meta = train_ranker(Xy)
        return self

    def recommend(self, customer_id: str, recent_service_ids: list[str] | None = None, top_k: int = 10):
        if recent_service_ids is None: recent_service_ids = []
        # Candidates
        c1 = self.emb.candidates_from_recent(self.service_catalog, recent_service_ids, top_k=200)
        c2 = self.mf.recommend(customer_id, top_k=200, exclude=recent_service_ids)
        seen, candidates = set(), []
        for s in c1 + c2:
            if s not in seen:
                seen.add(s)
                candidates.append(s)
            if len(candidates) >= 300:
                break

        cand_df = pd.DataFrame({"service_id": candidates})
        u = self.user_features[self.user_features["customer_id"] == customer_id]
        if u.empty:
            sc = self.service_catalog.copy()
            sc["score"] = sc["popularity"].rank(method="first", ascending=False)
            sc = sc.sort_values("score").head(top_k)
            return sc[["service_id","service_name","category","subcategory"]].to_dict(orient="records")

        u_row = u.iloc[0:1].drop(columns=["customer_id"])
        X = cand_df.merge(self.service_catalog, on="service_id", how="left")
        for col in u_row.columns:
            X[col] = u_row[col].values[0]

        for col in ["recency_days","frequency","monetary","avg_amount","price_sensitivity","popularity","avg_price","category","subcategory","top_category"]:
            if col not in X.columns:
                X[col] = "unknown" if col in ["category","subcategory","top_category"] else 0

        numeric = self.train_meta["features_numeric"]
        categorical = self.train_meta["features_categorical"]
        X_infer = X[numeric + categorical]
        scores = self.rank_model.predict_proba(X_infer)[:,1]
        X["score"] = scores
        X = X.sort_values("score", ascending=False).head(top_k)
        return X[["service_id","service_name","category","subcategory","score"]].to_dict(orient="records")