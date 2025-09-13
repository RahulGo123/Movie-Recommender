import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional
from spicy import sparse
from movierecommender.exception.exception import MovieRecommenderException
from movierecommender.logging.logger import logging

MODEL_PATH = os.getenv("MODEL_PATH", "final_model/model.pkl")
MOVIES_CSV = os.getenv("MODEL_CSV", "Movie_data/movies.csv")


class ModelServer:
    def __init__(self, model_path=MODEL_PATH, movies_csv=MOVIES_CSV):
        self.model_obj = joblib.load(model_path)
        self.movies = pd.read_csv(movies_csv)
        self.movie_map = dict(zip(self.movies.movieId, self.movies.title))

    def predict_score(self, input_df):
        preds = self.model_obj.predict(input_df)
        try:
            transformed = self.model_obj.preprocessor.transform(input_df)
            if hasattr(self.model_obj.model, "predict_proba"):
                probs = self.model_obj.model.predict_proba(transformed)
                scores = np.max(probs, axis=1)
                return scores
        except Exception as e:
            pass

        try:
            return np.asarray(preds, dtype=float)
        except Exception as e:
            logging.exception("Error in predict_score")
            return np.zeros(len(preds))

    def recommend_for_user(self,user_id: int,candidate_movie_ids: Optional[List[int]] = None,top_k: int = 10,batch_size: int = 10000,  ):
        try:
            if candidate_movie_ids is None:
                candidate_movie_ids = list(self.movie_map.keys())
                candidate_movie_ids = np.random.choice(candidate_movie_ids, 5000, replace=False).tolist()

            all_scores = []
            all_movie_ids = []

            # ✅ process movies in batches to avoid memory errors
            for start in range(0, len(candidate_movie_ids), batch_size):
                batch_ids = candidate_movie_ids[start:start+batch_size]

                rows = []
                for mid in batch_ids:
                    title = self.movie_map.get(mid, "")
                    genres = self.movies.loc[self.movies["movieId"] == mid, "genres"].values
                    genres = genres[0] if len(genres) > 0 else ""

                    rows.append({
                        "userId": int(user_id),
                        "movieId": int(mid),
                        "rating": 0,
                        "timestamp": 0,
                        "title": title,
                        "genres": genres,
                    })

                input_df = pd.DataFrame(rows)
                batch_scores = self.predict_score(input_df)

                # store batch results
                all_scores.extend(batch_scores)
                all_movie_ids.extend(batch_ids)

            # ✅ now pick the top_k overall
            all_scores = np.array(all_scores)
            idx_sorted = np.argsort(all_scores)[::-1][:top_k]

            recs = []
            for idx in idx_sorted:
                mid = int(all_movie_ids[idx])
                recs.append({
                    "movieId": mid,
                    "title": self.movie_map.get(mid, ""),
                    "score": float(all_scores[idx]),
                })

            return {"Recommendations": recs}
        except Exception as e:
            logging.exception("Error inside recommend_for_user")
            raise MovieRecommenderException(e, sys)