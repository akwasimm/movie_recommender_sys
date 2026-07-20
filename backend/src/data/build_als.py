"""
CineScope ALS Model Builder
Trains ALS collaborative filtering model from interactions.parquet
"""
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("als_builder")

ROOT = Path(__file__).resolve().parent.parent.parent


def run():
    log.info("=" * 60)
    log.info("CineScope ALS Model Builder")
    log.info("=" * 60)

    interactions_path = ROOT / "data" / "processed" / "interactions.parquet"
    movies_path = ROOT / "data" / "processed" / "movies_clean.parquet"

    if not interactions_path.exists():
        log.error(f"interactions.parquet not found at {interactions_path}")
        return
    if not movies_path.exists():
        log.error(f"movies_clean.parquet not found at {movies_path}")
        return

    log.info("Loading interactions...")
    interactions = pd.read_parquet(interactions_path)
    log.info(f"  Total interactions: {len(interactions)}")

    log.info("Loading clean movies...")
    movies = pd.read_parquet(movies_path)
    valid_ids = set(movies["tmdb_id"].tolist())
    log.info(f"  Valid movie IDs: {len(valid_ids)}")

    interactions = interactions[interactions["tmdb_id"].isin(valid_ids)].copy()
    log.info(f"  Interactions with valid movies: {len(interactions)}")

    interactions["userId"] = interactions["userId"].astype(int)
    interactions["tmdb_id"] = interactions["tmdb_id"].astype(int)

    user_ids = interactions["userId"].unique()
    movie_ids = interactions["tmdb_id"].unique()

    user_map = {uid: i for i, uid in enumerate(user_ids)}
    movie_map = {mid: i for i, mid in enumerate(movie_ids)}
    reverse_movie_map = {i: mid for mid, i in movie_map.items()}

    log.info(f"  Unique users: {len(user_ids)}")
    log.info(f"  Unique movies: {len(movie_ids)}")

    log.info("Building interaction matrix...")
    rows = interactions["userId"].map(user_map).values
    cols = interactions["tmdb_id"].map(movie_map).values
    ratings = interactions["rating"].values.astype(np.float32)

    matrix = csr_matrix((ratings, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    log.info(f"  Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")

    log.info("Training ALS model (factors=128, iterations=50, alpha=40)...")
    model = AlternatingLeastSquares(factors=128, iterations=50, regularization=0.01)
    model.fit(matrix * 40)

    out_dir = ROOT / "models" / "collaborative"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "als_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(out_dir / "als_meta.pkl", "wb") as f:
        pickle.dump({
            "user_map": user_map,
            "movie_map": movie_map,
            "reverse_movie_map": reverse_movie_map,
            "alpha": 40,
        }, f)

    with open(out_dir / "als_matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)

    log.info("Saved: als_model.pkl, als_meta.pkl, als_matrix.pkl")
    log.info("=" * 60)
    log.info("ALS build complete!")


if __name__ == "__main__":
    run()
