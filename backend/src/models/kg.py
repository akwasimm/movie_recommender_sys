import pickle
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

_ROOT = Path(__file__).resolve().parent.parent.parent

class KnowledgeGraphRecommender:
    def __init__(self):
        self.entity_names: list[str] = []
        self.movie_id_array: np.ndarray | None = None
        self.entity_movie_matrix: csr_matrix | None = None
        self.movie_entities: dict[int, set[str]] = {}
        self._id_to_idx: dict[int, int] = {}

    def load(self) -> bool:
        try:
            path = _ROOT / "models" / "graph" / "knowledge_graph.pkl"
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.entity_names = data["entity_names"]
            self.movie_id_array = data["movie_id_array"]
            self.entity_movie_matrix = data["entity_movie_matrix"]
            self.movie_entities = data["movie_entities"]
            self._id_to_idx = {
                int(mid): i for i, mid in enumerate(self.movie_id_array)
            }
            return True
        except Exception:
            return False

    def recommend(self, tmdb_id: int, n: int = 10) -> list[dict]:
        if self.entity_movie_matrix is None or tmdb_id not in self._id_to_idx:
            return []
        try:
            idx = self._id_to_idx[tmdb_id]
            query_vec = self.entity_movie_matrix[idx : idx + 1]
            scores = cosine_similarity(query_vec, self.entity_movie_matrix).flatten()
            top_indices = np.argpartition(scores, -(n + 1))[-(n + 1) :]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            results = []
            for i in top_indices:
                mid = int(self.movie_id_array[i])
                if mid != tmdb_id:
                    results.append({"tmdb_id": mid, "score": float(scores[i])})
                if len(results) >= n:
                    break
            return results
        except Exception:
            return []

    def explain(self, tmdb_id_a: int, tmdb_id_b: int) -> dict:
        a_entities = self.movie_entities.get(tmdb_id_a, set())
        b_entities = self.movie_entities.get(tmdb_id_b, set())
        shared = a_entities & b_entities
        return {
            "shared_entities": sorted(shared),
            "entity_count": len(shared),
        }
