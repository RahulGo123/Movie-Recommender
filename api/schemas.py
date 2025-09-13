# api/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str

class RecommendRequest(BaseModel):
    user_id: int
    top_k: Optional[int] = 10
    candidate_movie_ids: Optional[List[int]] = None  # if None -> consider all movies

class MovieItem(BaseModel):
    movieId: int
    title: str
    score: float

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[MovieItem]
