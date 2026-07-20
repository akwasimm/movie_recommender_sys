import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares

_ROOT = Path(__file__).resolve().parent.parent.parent

class ALSRecommender:
    def __init__(self):
        self.model: AlternatingLeastSquares | None = None
        self.user_map: dict[int, int] = {}
        self.movie_map: dict[int, int] = {}
        self.reverse_movie_map: dict[int, int] = {}
        self.interaction_matrix = None

    def load(self) -> bool:
        try:
            base = _ROOT / "models" / "collaborative"

            meta_path = base / "als_meta.pkl"
            npz_path = base / "als_meta.npz"
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
            elif npz_path.exists():
                data = np.load(npz_path)
                meta = {
                    "user_map": dict(zip(data["user_ids"].tolist(), data["user_vals"].tolist())),
                    "movie_map": dict(zip(data["movie_ids"].tolist(), data["movie_vals"].tolist())),
                }
                meta["reverse_movie_map"] = {v: k for k, v in meta["movie_map"].items()}
            else:
                return False

            self.user_map = meta["user_map"]
            self.movie_map = meta["movie_map"]
            self.reverse_movie_map = meta["reverse_movie_map"]

            npz_matrix = base / "als_matrix.npz"
            pkl_matrix = base / "als_matrix.pkl"
            if npz_matrix.exists():
                self.interaction_matrix = np.load(npz_matrix, allow_pickle=True)["arr_0"]
            elif pkl_matrix.exists():
                with open(pkl_matrix, "rb") as f:
                    self.interaction_matrix = pickle.load(f)

            factors_npy = base / "als_user_factors.npy"
            items_npy = base / "als_item_factors.npy"
            model_pkl = base / "als_model.pkl"
            if factors_npy.exists() and items_npy.exists():
                from implicit.cpu.als import AlternatingLeastSquares as ALS
                self.model = ALS(factors=128, regularization=0.01, iterations=1)
                self.model.user_factors = np.load(factors_npy)
                self.model.item_factors = np.load(items_npy)
            elif model_pkl.exists():
                with open(model_pkl, "rb") as f:
                    self.model = pickle.load(f)

            return True
        except Exception:
            return False

    def recommend(self, user_id: int, n: int = 10) -> list[dict]:
        if self.model is None:
            return []
        if user_id not in self.user_map:
            return []
        try:
            inner_uid = self.user_map[user_id]
            ids, scores = self.model.recommend(
                inner_uid,
                self.interaction_matrix[inner_uid],
                N=n,
                filter_already_liked_items=True,
            )
            return [
                {"tmdb_id": int(self.reverse_movie_map[i]), "score": float(s)}
                for i, s in zip(ids, scores)
            ]
        except Exception:
            return []

    def similar_items(self, tmdb_id: int, n: int = 10) -> list[dict]:
        if self.model is None or tmdb_id not in self.movie_map:
            return []
        try:
            inner_mid = self.movie_map[tmdb_id]
            ids, scores = self.model.similar_items(inner_mid, N=n + 1)
            return [
                {"tmdb_id": int(self.reverse_movie_map[i]), "score": float(s)}
                for i, s in zip(ids, scores)
                if i != inner_mid
            ][:n]
        except Exception:
            return []
