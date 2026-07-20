import pandas as pd
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

class HybridRecommender:
    def __init__(self, als=None, bert=None, kg=None, movies_df=None):
        self.als = als
        self.bert = bert
        self.kg = kg
        self.movies_df = movies_df
        self.weights = {"als": 0.40, "bert": 0.25, "kg": 0.20, "popularity": 0.15}
        self.candidate_pool = 50

    def recommend(self, user_id: int = None, tmdb_id: int = None, n: int = 10) -> list[dict]:
        candidates: dict[int, float] = {}
        active_weight = 0.0

        if user_id and self.als:
            als_recs = self.als.recommend(user_id, n=self.candidate_pool)
            if als_recs:
                max_s = max(r["score"] for r in als_recs) or 1
                for r in als_recs:
                    tid = r["tmdb_id"]
                    candidates[tid] = candidates.get(tid, 0) + (r["score"] / max_s) * self.weights["als"]
                active_weight += self.weights["als"]

        if tmdb_id and self.bert:
            bert_recs = self.bert.recommend_from_movie(tmdb_id, n=self.candidate_pool)
            if bert_recs:
                max_s = max(r["score"] for r in bert_recs) or 1
                for r in bert_recs:
                    tid = r["tmdb_id"]
                    candidates[tid] = candidates.get(tid, 0) + (r["score"] / max_s) * self.weights["bert"]
                active_weight += self.weights["bert"]

        if tmdb_id and self.kg:
            kg_recs = self.kg.recommend(tmdb_id, n=self.candidate_pool)
            if kg_recs:
                max_s = max(r["score"] for r in kg_recs) or 1
                for r in kg_recs:
                    tid = r["tmdb_id"]
                    candidates[tid] = candidates.get(tid, 0) + (r["score"] / max_s) * self.weights["kg"]
                active_weight += self.weights["kg"]

        if not candidates:
            return self._popular_movies(n)

        if active_weight > 0:
            for tid in candidates:
                candidates[tid] /= active_weight

        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{"tmdb_id": tid, "score": round(s, 4)} for tid, s in ranked]

    def _popular_movies(self, n: int = 10) -> list[dict]:
        if self.movies_df is None:
            return []
        top = self.movies_df.nlargest(n, "popularity")
        return [{"tmdb_id": int(r["tmdb_id"]), "score": 1.0} for _, r in top.iterrows()]
