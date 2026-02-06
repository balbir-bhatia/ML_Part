from fastapi import FastAPI, Request, Response
from pydantic import BaseModel
from datetime import datetime
import os, json, uuid

from config.settings import settings
from .recommender import load_pipeline, recommend_for_user

app = FastAPI(title="Service Recommendation API", version="1.0")

class RecommendRequest(BaseModel):
    customer_id: str
    recent_service_ids: list[str] | None = None
    context: dict | None = None
    top_k: int = 10

class EventsRequest(BaseModel):
    customer_id: str
    session_id: str | None = None
    context: dict | None = None
    events: list[dict]

@app.on_event("startup")
def _load():
    os.makedirs(settings.DATA_EVENTS, exist_ok=True)
    global PIPE
    PIPE = load_pipeline()

@app.post("/v1/recommend")
def recommend(req: RecommendRequest):
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    recs = recommend_for_user(PIPE, req.customer_id, req.recent_service_ids, req.top_k)
    resp = {
        "user": req.customer_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "recommendations": recs,
        "debug": {
            "candidate_sources": ["embeddings","mf"],
            "model_version": settings.MODEL_VERSION
        }
    }
    from fastapi.responses import JSONResponse
    r = JSONResponse(resp)
    r.headers["X-Req-Id"] = req_id
    r.headers["X-Model-Version"] = settings.MODEL_VERSION
    r.headers["Cache-Control"] = "private, max-age=30"
    return r

@app.post("/v1/events")
def track_events(req: EventsRequest):
    # Append events to JSONL for hackathon simplicity
    path = os.path.join(settings.DATA_EVENTS, "events.jsonl")
    payload = req.dict()
    payload["received_at"] = datetime.utcnow().isoformat() + "Z"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    return {"status": "ok", "accepted": len(req.events)}

@app.get("/v1/popular-services")
def popular_services(city: str | None = None, category: str | None = None, limit: int = 10):
    import pandas as pd
    sc_path = os.path.join(settings.DATA_FEATURES, "service_catalog.csv")
    if not os.path.exists(sc_path):
        return {"services": []}
    sc = pd.read_csv(sc_path)
    df = sc.copy()
    if category:
        df = df[df["category"] == category]
    df = df.sort_values("popularity", ascending=False).head(limit)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "services": df[["service_id","service_name","category","subcategory","popularity"]].to_dict(orient="records")
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": settings.MODEL_VERSION}