import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

class BERTRecommender:
    def __init__(self):
        self.embeddings: np.ndarray | None = None
        self.movie_ids: np.ndarray | None = None
        self._id_to_idx: dict[int, int] = {}

    def load(self) -> bool:
        try:
            base = _ROOT / "data" / "embeddings"
            self.embeddings = np.load(str(base / "bert_embeddings.npy"))
            self.movie_ids = np.load(str(base / "bert_movie_ids.npy"))
            self._id_to_idx = {
                int(mid): i for i, mid in enumerate(self.movie_ids)
            }
            return True
        except Exception:
            return False

    def recommend_from_movie(self, tmdb_id: int, n: int = 10) -> list[dict]:
        if self.embeddings is None or tmdb_id not in self._id_to_idx:
            return []
        try:
            idx = self._id_to_idx[tmdb_id]
            query = self.embeddings[idx : idx + 1]
            scores = self.embeddings @ query.T
            scores = scores.flatten()
            top_indices = np.argpartition(scores, -(n + 1))[-(n + 1) :]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            results = []
            for i in top_indices:
                mid = int(self.movie_ids[i])
                if mid != tmdb_id:
                    results.append({"tmdb_id": mid, "score": float(scores[i])})
                if len(results) >= n:
                    break
            return results
        except Exception:
            return []

    def recommend_from_text(self, query: str, n: int = 10) -> list[dict]:
        if self.embeddings is None:
            return []
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            scores = self.embeddings @ q_emb
            top_indices = np.argpartition(scores, -(n + 1))[-(n + 1) :]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            return [
                {"tmdb_id": int(self.movie_ids[i]), "score": float(scores[i])}
                for i in top_indices[:n]
            ]
        except ImportError:
            return []
        except Exception:
            return []
