import pandas as pd
import numpy as np
import requests
from pathlib import Path
from functools import lru_cache

_ROOT = Path(__file__).resolve().parent.parent.parent


class MovieData:
    def __init__(self):
        self.movies: pd.DataFrame | None = None
        self._id_to_idx: dict[int, int] = {}

    def load(self) -> bool:
        try:
            path = _ROOT / "data" / "processed" / "movies_clean.parquet"
            self.movies = pd.read_parquet(path)
            self._id_to_idx = {
                int(tid): i for i, tid in enumerate(self.movies["tmdb_id"])
            }
            return True
        except Exception:
            return False

    def get_movie(self, tmdb_id: int) -> dict | None:
        if tmdb_id not in self._id_to_idx:
            return None
        row = self.movies.iloc[self._id_to_idx[tmdb_id]]
        return _row_to_dict(row)

    def search(self, query: str, limit: int = 10) -> list[dict]:
        if self.movies is None:
            return []
        q = query.lower()
        mask = self.movies["title"].str.lower().str.contains(q, na=False, regex=False)
        return [_row_to_dict(r) for _, r in self.movies[mask].head(limit).iterrows()]

    def trending(self, n: int = 20) -> list[dict]:
        if self.movies is None:
            return []
        return [_row_to_dict(r) for _, r in self.movies.nlargest(n, "popularity").iterrows()]

    def top_rated(self, n: int = 20) -> list[dict]:
        if self.movies is None:
            return []
        return [_row_to_dict(r) for _, r in self.movies.nlargest(n, "vote_average").iterrows()]

    def by_genre(self, genre: str, n: int = 20) -> list[dict]:
        if self.movies is None or "genres_clean" not in self.movies.columns:
            return []
        genre_lower = genre.lower().strip()
        mask = self.movies["genres_clean"].str.contains(genre_lower, na=False, regex=False)
        results = self.movies[mask].nlargest(n, "popularity")
        return [_row_to_dict(r) for _, r in results.iterrows()]

    def by_decade(self, decade: int, n: int = 20) -> list[dict]:
        if self.movies is None or "decade" not in self.movies.columns:
            return []
        mask = self.movies["decade"] == decade
        results = self.movies[mask].nlargest(n, "popularity")
        return [_row_to_dict(r) for _, r in results.iterrows()]

    def get_all_ids(self) -> list[int]:
        if self.movies is None:
            return []
        return self.movies["tmdb_id"].tolist()

    def random_by_genre(self, genre: str, n: int = 15) -> list[dict]:
        if self.movies is None or "genres_clean" not in self.movies.columns:
            return []
        genre_lower = genre.lower().strip()
        mask = self.movies["genres_clean"].str.contains(genre_lower, na=False, regex=False)
        pool = self.movies[mask]
        if len(pool) == 0:
            return []
        sample = pool.sample(n=min(n, len(pool)))
        return [_row_to_dict(r) for _, r in sample.iterrows()]

    def random_top(self, n: int = 15, min_votes: int = 200) -> list[dict]:
        if self.movies is None:
            return []
        pool = self.movies[self.movies["vote_count"] >= min_votes]
        sample = pool.sample(n=min(n, len(pool)))
        return [_row_to_dict(r) for _, r in sample.iterrows()]

    def random_by_decade(self, decade: int, n: int = 15) -> list[dict]:
        if self.movies is None or "decade" not in self.movies.columns:
            return []
        mask = self.movies["decade"] == decade
        pool = self.movies[mask]
        if len(pool) == 0:
            return []
        sample = pool.sample(n=min(n, len(pool)))
        return [_row_to_dict(r) for _, r in sample.iterrows()]


@lru_cache(maxsize=512)
def fetch_credits(tmdb_id: int, api_key: str) -> list[dict]:
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits"
        resp = requests.get(url, params={"api_key": api_key, "language": "en-US"}, timeout=8)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [
            {"name": c["name"], "character": c["character"], "profile_path": c.get("profile_path")}
            for c in data.get("cast", [])[:6]
        ]
    except Exception:
        return []


def _row_to_dict(row) -> dict:
    genres_raw = row.get("genres_clean", "")
    if isinstance(genres_raw, str) and genres_raw:
        genre_list = [g.strip() for g in genres_raw.split("|") if g.strip()]
    else:
        genre_list = []

    year = None
    rd = row.get("release_date", "")
    if pd.notna(rd):
        rd_str = str(rd)
        if len(rd_str) >= 4:
            try:
                year = int(rd_str[:4])
            except ValueError:
                pass
    else:
        rd_str = ""

    return {
        "tmdb_id": int(row["tmdb_id"]),
        "title": str(row.get("title", "")),
        "overview": str(row.get("overview", "")),
        "genres": genre_list,
        "vote_average": float(row.get("vote_average", 0) or 0),
        "vote_count": int(row.get("vote_count", 0) or 0),
        "popularity": float(row.get("popularity", 0) or 0),
        "release_date": rd_str,
        "year": year,
        "runtime": int(row.get("runtime", 0) or 0),
        "poster_path": f"https://image.tmdb.org/t/p/w500{row.get('poster_path', '')}" if row.get("poster_path") else None,
        "backdrop_path": f"https://image.tmdb.org/t/p/w1280{row.get('backdrop_path', '')}" if row.get("backdrop_path") else None,
    }
