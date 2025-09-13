from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import HealthResponse, RecommendRequest, RecommendResponse, MovieItem
from api.model_server import ModelServer

app = FastAPI(title="Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_server = ModelServer()

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "Movie Recommender API is running!"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        recs = model_server.recommend_for_user(req.user_id, req.candidate_movie_ids, req.top_k)
        return {"user_id": req.user_id, "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    